"""Checks on the word 13-gram overlap statistic, and the datastore comparison in words.

For one domain's training data, computes the fraction of each val/test document's word n-grams
(n in --ns; words and hashes as in prepare_cc_news.ngram_hashes) found in
  - full_train: the whole training split (n=13 reproduces ngram_overlap.py / prepare_cc_news.py),
  - datastore: the text of the kNN-LM datastore, i.e. the training tokens of the windows selected
    as in dstore_ngram_overlap.py, decoded back to text one contiguous run at a time.
Documents scored are the domain's own val/test and, as a chance baseline, another domain's val/test,
which share no source documents with this training data.

Words cut by a run boundary are dropped, so no datastore n-gram spans two runs or holds a partial word.
Python's hash() is salted per process, so every hash is compared within this one process.
Run from the directory that holds data/ and out/, inside a job.
"""
import os
import json
import argparse

import numpy as np

import data
from data import load_pile
from model import LanguageModel
from prepare_cc_news import HASH_MULT
from dstore_ngram_overlap import load_flat, window_target_ranges, contains

THRESHOLDS = [0.5, 0.8, 0.99]

class Corpus(object):
    """Word hashes of a list of documents (each a list of words), concatenated."""

    def __init__(self, word_lists):
        chunks, lengths = [], []
        for words in word_lists:
            chunks.append(np.fromiter((hash(w) for w in words), dtype=np.int64, count=len(words)))
            lengths.append(len(words))
        self.ids = np.concatenate(chunks).view(np.uint64)
        self.lengths = np.array(lengths, dtype=np.int64)
        # words from each position to the end of its document
        self.left = np.repeat(np.cumsum(self.lengths), self.lengths) - np.arange(len(self.ids))

    def ngrams(self, n):
        """Hash of every n-gram starting at each position, and whether it lies within one document."""
        m = max(len(self.ids) - n + 1, 0)
        h = np.zeros(m, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(n):
                h *= HASH_MULT
                h += self.ids[j:j+m]
        return h, self.left[:m] >= n

    def reference(self, n):
        h, valid = self.ngrams(n)
        return np.unique(h[valid])

    def overlap(self, ref, n):
        """Per document, the fraction of its n-grams found in ref (NaN if it has none); found and total counts."""
        h, valid = self.ngrams(n)
        docs = np.repeat(np.arange(len(self.lengths)), self.lengths)[:len(h)][valid]
        found = np.bincount(docs, weights=contains(ref, h[valid]), minlength=len(self.lengths))
        total = np.bincount(docs, minlength=len(self.lengths))
        fracs = np.full(len(self.lengths), np.nan)
        fracs[total > 0] = found[total > 0] / total[total > 0]
        return fracs, int(found.sum()), int(total.sum())

def summarize(fracs, common, n_found, n_total):
    valid = fracs[~np.isnan(fracs)]
    stats = {"n_docs_with_ngrams": int(len(valid)), "frac_all_ngrams_found": n_found / n_total,
             "median_doc_overlap": float(np.median(valid)), "frac_docs_any_found": float(np.mean(valid > 0))}
    for t in THRESHOLDS:
        stats["frac_docs_overlap>={}".format(t)] = float(np.mean(valid >= t))
    # on the documents long enough for every n, so values are comparable across n
    stats["frac_docs_overlap>=0.5_common"] = float(np.mean(fracs[common] >= 0.5))
    return stats

def datastore_word_lists(args, tokenizer, pad_token):
    """Words of each contiguous run of datastore tokens within a training document, and a decoded
    sample of whole training documents for checking the decoding."""
    flat, lengths = load_flat(os.path.join(args.output_dir, "neoX", "train", "{}-tokenized.pkl".format(args.subset)))
    starts, ends = window_target_ranges(len(flat), args.max_seq_length, args.stride)
    np.random.seed(2023)  # as in main.py's get_input_ids_and_targets
    perm = np.random.permutation(len(starts))
    selected = perm[:min(args.n_shards * args.max_n_sequences * 1000, len(starts))]
    token_stats = json.load(open(os.path.join("data", "{}-dstore-overlap".format(args.subset), "duplicate_stats.json")))
    assert len(selected) == token_stats["n_datastore_windows"] and len(flat) == token_stats["n_train_tokens"]

    in_dstore = np.zeros(len(flat) + 1, dtype=np.int64)
    np.add.at(in_dstore, starts[selected], 1)
    np.add.at(in_dstore, ends[selected] + 1, -1)
    in_dstore = (np.cumsum(in_dstore)[:len(flat)] > 0) & (flat != pad_token)

    offsets = np.cumsum(lengths) - lengths
    doc_start = np.zeros(len(flat) + 1, dtype=bool)
    doc_start[offsets] = True
    doc_start[len(flat)] = True
    prev = np.concatenate([[False], in_dstore[:-1]])
    nxt = np.concatenate([in_dstore[1:], [False]])
    run_starts = np.flatnonzero(in_dstore & (~prev | doc_start[:-1]))
    run_ends = np.flatnonzero(in_dstore & (~nxt | doc_start[1:])) + 1
    assert len(run_starts) == len(run_ends)

    # a run edge falls between words if it is a document edge or whitespace is next to it
    vocab = tokenizer.batch_decode([[i] for i in range(len(tokenizer))], clean_up_tokenization_spaces=False)
    starts_ws = np.array([t[:1].isspace() for t in vocab])
    ends_ws = np.array([t[-1:].isspace() for t in vocab])
    assert flat.max() < len(vocab)
    clean_start = doc_start[run_starts] | starts_ws[flat[run_starts]] | ends_ws[flat[np.maximum(run_starts - 1, 0)]]
    clean_end = doc_start[run_ends] | ends_ws[flat[run_ends - 1]] | starts_ws[flat[np.minimum(run_ends, len(flat) - 1)]]

    word_lists, n_dropped = [], 0
    for b in range(0, len(run_starts), 10000):
        texts = tokenizer.batch_decode([flat[s:e].tolist() for s, e in zip(run_starts[b:b+10000], run_ends[b:b+10000])],
                                       clean_up_tokenization_spaces=False)
        for text, cs, ce in zip(texts, clean_start[b:b+10000], clean_end[b:b+10000]):
            words = text.lower().split()
            n_words = len(words)
            words = words[(0 if cs else 1):(n_words if ce else n_words - 1)]
            n_dropped += n_words - len(words)
            word_lists.append(words)

    rng = np.random.RandomState(0)
    sample = rng.choice(len(lengths), size=min(1000, len(lengths)), replace=False)
    decoded = tokenizer.batch_decode([flat[offsets[i]:offsets[i] + lengths[i]].tolist() for i in sample],
                                     clean_up_tokenization_spaces=False)
    info = {"n_train_docs_tokenized": int(len(lengths)), "n_datastore_windows": int(len(selected)),
            "n_datastore_tokens": int(in_dstore.sum()), "n_runs": int(len(run_starts)),
            "n_words_kept": int(sum(len(w) for w in word_lists)), "n_words_dropped_at_run_edges": int(n_dropped)}
    return word_lists, info, dict(zip(sample.tolist(), decoded))

def load_eval(subset, work):
    saved, data.DATA_DIR = data.DATA_DIR, os.path.join(work, "data")
    try:
        return {split: load_pile(split, subset=subset) for split in ["val", "test"]}
    finally:
        data.DATA_DIR = saved

def cell(stats, splits, key, fmt="{:.3f}"):
    return " / ".join(fmt.format(stats[s][key]) for s in splits)

def report(args, stats, token_stats):
    splits = ["val", "test"]
    md = ["# {}: overlap checks (word n-grams)\n".format(args.subset),
          "References: *full_train* = the whole {} training split; *datastore* = the text of the kNN-LM "
          "datastore ({} of {} windows, {} tokens), decoded run by run, words cut at run edges dropped. "
          "Cells are val / test. *docs >= {} words* restricts to documents long enough for every n.\n".format(
              args.subset, stats["datastore"]["n_datastore_windows"], token_stats["n_windows"],
              stats["datastore"]["n_datastore_tokens"], max(args.ns)),
          "Token 13-grams (dstore_ngram_overlap.py), docs >= 50%: datastore {}, full_train {}.\n".format(
              " / ".join("{:.3f}".format(token_stats[s]["datastore"]["frac_docs_13gram_overlap>=0.5"]) for s in splits),
              " / ".join("{:.3f}".format(token_stats[s]["full_train"]["frac_docs_13gram_overlap>=0.5"]) for s in splits)),
          "Checks: word 13-gram full_train overlap vs stored arrays, max abs diff {}; decoded train docs with the "
          "same words as the text: {} of {}.\n".format(
              " / ".join("{:.2g}".format(stats["check"]["max_abs_diff_vs_stored"][s]) for s in splits),
              stats["check"]["decoded_docs_same_words"], stats["check"]["decoded_docs_checked"])]
    for name, by_split in stats["evals"].items():
        md += ["\n## {} val/test vs {} train\n".format(name, args.subset),
               "| reference | n | docs | >= 50% | >= 50%, docs >= {} words | >= 80% | >= 99% | any n-gram found |".format(max(args.ns)),
               "|---|---|---|---|---|---|---|---|"]
        for ref in ["full_train", "datastore"]:
            for n in args.ns:
                s = {split: by_split[split][ref][str(n)] for split in splits}
                md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    ref, n, cell(s, splits, "n_docs_with_ngrams", "{}"), cell(s, splits, "frac_docs_overlap>=0.5"),
                    cell(s, splits, "frac_docs_overlap>=0.5_common"), cell(s, splits, "frac_docs_overlap>=0.8"),
                    cell(s, splits, "frac_docs_overlap>=0.99"), cell(s, splits, "frac_docs_any_found")))
    return "\n".join(md) + "\n"

def main(args):
    lm = LanguageModel(args.lm)
    ds_words, ds_info, decoded_sample = datastore_word_lists(args, lm.tokenizer, lm.pad_token)
    datastore = Corpus(ds_words)
    del ds_words
    print ("datastore: {}".format(ds_info), flush=True)

    train_texts = load_pile("train", subset=args.subset)
    assert len(train_texts) == ds_info["n_train_docs_tokenized"], len(train_texts)
    same = sum(decoded.lower().split() == train_texts[i].lower().split() for i, decoded in decoded_sample.items())
    train = Corpus(t.lower().split() for t in train_texts)
    del train_texts
    print ("train: {} docs, {} words".format(len(train.lengths), len(train.ids)), flush=True)

    evals = {args.subset: load_eval(args.subset, "."), args.other_subset: load_eval(args.other_subset, args.other_work)}
    evals = {name: {split: Corpus(t.lower().split() for t in texts) for split, texts in by_split.items()}
             for name, by_split in evals.items()}

    stats = {"datastore": ds_info, "n_train_docs": int(len(train.lengths)), "n_train_words": int(len(train.ids)),
             "evals": {name: {split: {"full_train": {}, "datastore": {}} for split in by_split} for name, by_split in evals.items()}}
    arrays = {(name, split): {} for name, by_split in evals.items() for split in by_split}
    for n in args.ns:
        refs = {"full_train": train.reference(n), "datastore": datastore.reference(n)}
        for name, by_split in evals.items():
            for split, corpus in by_split.items():
                common = corpus.lengths >= max(args.ns)
                for ref_name, ref in refs.items():
                    fracs, n_found, n_total = corpus.overlap(ref, n)
                    stats["evals"][name][split][ref_name][str(n)] = summarize(fracs, common, n_found, n_total)
                    arrays[(name, split)]["{}_{}".format(ref_name, n)] = fracs
        del refs
        print ("n={} done".format(n), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stats["check"] = {"decoded_docs_checked": len(decoded_sample), "decoded_docs_same_words": int(same),
                      "max_abs_diff_vs_stored": {}}
    for split in ["val", "test"]:
        stored = np.load(os.path.join(args.stored_overlap_dir, "{}_13gram_overlap.npy".format(split)))
        ours = arrays[(args.subset, split)]["full_train_13"]
        assert np.array_equal(np.isnan(stored), np.isnan(ours))
        stats["check"]["max_abs_diff_vs_stored"][split] = float(np.nanmax(np.abs(stored - ours)))
        # same layout as the token-level per-document arrays, for report_knn.py --overlap_dir
        np.save(os.path.join(args.out_dir, "{}_13gram_overlap.npy".format(split)), arrays[(args.subset, split)]["datastore_13"])
    for (name, split), a in arrays.items():
        np.savez(os.path.join(args.out_dir, "{}_{}_overlap.npz".format(name, split)), **a)

    token_stats = json.load(open(os.path.join("data", "{}-dstore-overlap".format(args.subset), "duplicate_stats.json")))
    with open(os.path.join(args.out_dir, "overlap_checks.md"), "w") as f:
        f.write(report(args, stats, token_stats))
    print (json.dumps(stats["check"], indent=2))
    # written last: its presence means the outputs are complete
    with open(os.path.join(args.out_dir, "overlap_checks.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--other_subset", type=str, required=True, help="domain whose val/test serve as the chance baseline")
    parser.add_argument("--other_work", type=str, required=True, help="directory holding the other domain's data/")
    parser.add_argument("--stored_overlap_dir", type=str, required=True,
                        help="holds the word-level {val,test}_13gram_overlap.npy from ngram_overlap.py / prepare_cc_news.py")
    parser.add_argument("--n_shards", type=int, required=True, help="number of shards in the datastore")
    parser.add_argument("--ns", type=int, nargs="+", default=[8, 13, 20, 50])
    parser.add_argument("--lm", type=str, default="pythia-1.4B")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--max_n_sequences", default=10, type=int)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
