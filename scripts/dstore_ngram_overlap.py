"""Measure how much of each val/test document is duplicated in the kNN-LM datastore itself,
as token 13-gram overlap.

A datastore built with --max_n_sequences N and shards 0..S-1 holds the targets of the first
S*N*1000 windows of main.py's seed-2023 window permutation (positions 1..L-1 of the concatenated
training documents, each in exactly one window, minus pad-token targets). A val/test 13-gram counts
as found if the same 13 tokens occur at 13 consecutive datastore positions. Overlap with the whole
training data is computed the same way, for comparison.

The reconstruction is checked against the shard target files written by main.py
({subset}-tokenized_[aK-bK]_flatten.npy). Run from the directory that holds out/, inside a job.
"""
import os
import json
import argparse
import pickle as pkl

import numpy as np

from model import LanguageModel

NGRAM = 13
HASH_MULT = np.uint64(0x9E3779B97F4A7C15)

def ngram_hashes(ids):
    """Hash of ids[s:s+NGRAM] for every start s (length len(ids)-NGRAM+1)."""
    ids = ids.astype(np.uint64)
    n = len(ids) - NGRAM + 1
    h = np.zeros(max(n, 0), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(NGRAM):
            h = h * HASH_MULT + ids[j:j+n]
    return h

def contains(sorted_hashes, h):
    pos = np.minimum(np.searchsorted(sorted_hashes, h), len(sorted_hashes) - 1)
    return sorted_hashes[pos] == h

def load_flat(tokenized_path):
    with open(tokenized_path, "rb") as f:
        input_ids = pkl.load(f)
    lengths = np.array([len(ids) for ids in input_ids])
    return np.concatenate([np.asarray(ids, dtype=np.int64) for ids in input_ids]), lengths

def window_target_ranges(n_tokens, max_seq_length, stride):
    """[start, end] (inclusive) target positions of every window, as in model.batch_merged."""
    starts, ends, prev_end = [], [], 0
    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + max_seq_length, n_tokens - 1)
        starts.append(prev_end + 1)
        ends.append(end)
        prev_end = end
        if end == n_tokens - 1:
            break
    return np.array(starts), np.array(ends)

def doc_overlap(flat, lengths, references):
    """Per document, the fraction of its 13-grams (within the document) found in each reference."""
    doc_ids = np.repeat(np.arange(len(lengths)), lengths)
    h = ngram_hashes(flat)
    same_doc = doc_ids[:len(h)] == doc_ids[NGRAM-1:]
    h, h_docs = h[same_doc], doc_ids[:len(same_doc)][same_doc]
    n_ngrams = np.bincount(h_docs, minlength=len(lengths))
    results = {}
    for name, ref in references.items():
        found = np.bincount(h_docs, weights=contains(ref, h), minlength=len(lengths))
        fracs = np.full(len(lengths), np.nan)
        fracs[n_ngrams > 0] = found[n_ngrams > 0] / n_ngrams[n_ngrams > 0]
        valid = fracs[~np.isnan(fracs)]
        results[name] = (fracs, {
            "n_docs": len(lengths),
            "n_docs_with_13grams": int(len(valid)),
            "frac_docs_13gram_overlap>=0.5": float(np.mean(valid >= 0.5)),
            "frac_docs_13gram_overlap>=0.8": float(np.mean(valid >= 0.8)),
            "frac_docs_13gram_overlap>=0.99": float(np.mean(valid >= 0.99)),
            "median_doc_13gram_overlap": float(np.median(valid)),
            "frac_all_13grams_found": float(found.sum() / n_ngrams.sum()),
        })
    return results

def main(args):
    pad_token = LanguageModel(args.lm).pad_token
    tok_dir = os.path.join(args.output_dir, "neoX")

    flat, _ = load_flat(os.path.join(tok_dir, "train", "{}-tokenized.pkl".format(args.subset)))
    starts, ends = window_target_ranges(len(flat), args.max_seq_length, args.stride)
    np.random.seed(2023)  # as in main.py's get_input_ids_and_targets
    perm = np.random.permutation(len(starts))
    n_selected = min(args.n_shards * args.max_n_sequences * 1000, len(starts))
    selected = perm[:n_selected]

    # check against the shard target files written by main.py
    for i in range(args.n_shards):
        shard = selected[i * args.max_n_sequences * 1000:(i + 1) * args.max_n_sequences * 1000]
        positions = np.concatenate([np.arange(starts[w], ends[w] + 1) for w in shard])
        targets = flat[positions]
        targets = targets[targets != pad_token]
        saved = np.load(os.path.join(tok_dir, "train", "{}-tokenized_[{}K-{}K]_flatten.npy".format(
            args.subset, args.max_n_sequences * i, args.max_n_sequences * (i + 1))))
        assert np.array_equal(targets, saved), "shard %d does not match main.py's targets" % i
    print ("Datastore windows reproduced: %d of %d windows, %d shards checked" % (n_selected, len(starts), args.n_shards))

    in_dstore = np.zeros(len(flat) + 1, dtype=np.int64)
    np.add.at(in_dstore, starts[selected], 1)
    np.add.at(in_dstore, ends[selected] + 1, -1)
    in_dstore = (np.cumsum(in_dstore)[:len(flat)] > 0) & (flat != pad_token)
    h = ngram_hashes(flat)
    run = np.convolve(in_dstore.astype(np.int64), np.ones(NGRAM, dtype=np.int64), mode="valid")
    references = {"datastore": np.unique(h[run == NGRAM]), "full_train": np.unique(h)}
    del h, run, in_dstore

    stats = {"n_train_tokens": int(len(flat)), "n_windows": int(len(starts)), "n_datastore_windows": int(n_selected),
             "n_unique_13grams": {name: int(len(ref)) for name, ref in references.items()}}
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["val", "test"]:
        flat_eval, lengths = load_flat(os.path.join(tok_dir, split, "{}-tokenized.pkl".format(args.subset)))
        results = doc_overlap(flat_eval, lengths, references)
        stats[split] = {name: s for name, (_, s) in results.items()}
        np.save(os.path.join(args.out_dir, "{}_13gram_overlap.npy".format(split)), results["datastore"][0])
        np.save(os.path.join(args.out_dir, "{}_13gram_overlap_full_train.npy".format(split)), results["full_train"][0])

    print (json.dumps(stats, indent=2))
    with open(os.path.join(args.out_dir, "duplicate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--lm", type=str, default="pythia-1.4B")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--max_n_sequences", default=10, type=int)
    parser.add_argument("--n_shards", type=int, required=True, help="number of shards in the datastore")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="where to write {val,test}_13gram_overlap.npy (vs the datastore) and duplicate_stats.json")
    args = parser.parse_args()
    main(args)
