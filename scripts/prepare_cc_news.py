"""Write data/cc-news/{train,val,test}.jsonl the way download_data.download_cc_news does, and
measure how much of val/test is duplicated in train.

The `cc_news` dataset used by download_cc_news is now hosted as parquet files at
vblagoje/cc_news (same 708,241 articles); `load_dataset('cc_news')` no longer works with
current versions of `datasets`, so this reads the parquet files directly, in order.

Duplicate statistics (the split is random and nothing is deduplicated, as in SILO):
    - exact: val/test articles whose text also appears verbatim in train
    - near: fraction of each article's word 13-grams that also occur somewhere in train
      (the n-gram length commonly used for contamination checks)
"""
import os
import json
import argparse

import numpy as np
import pyarrow.parquet as pq

N_EXPECTED = 708241
NGRAM = 13
HASH_MULT = np.uint64(1000003)

def ngram_hashes(text):
    words = text.lower().split()
    if len(words) < NGRAM:
        return np.zeros(0, dtype=np.uint64)
    ids = np.fromiter((hash(w) for w in words), dtype=np.int64, count=len(words)).view(np.uint64)
    n = len(ids) - NGRAM + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(NGRAM):
            h = h * HASH_MULT + ids[j:j+n]
    return h

def overlap_stats(texts, train_texts, train_ngrams):
    """Returns aggregate statistics, and each document's fraction of 13-grams found in train
    (NaN for documents shorter than 13 words)."""
    exact = sum(t in train_texts for t in texts)
    doc_fracs, n_found, n_total = np.full(len(texts), np.nan), 0, 0
    for i, text in enumerate(texts):
        h = ngram_hashes(text)
        if len(h) == 0:
            continue
        pos = np.minimum(np.searchsorted(train_ngrams, h), len(train_ngrams) - 1)
        found = int(np.sum(train_ngrams[pos] == h))
        doc_fracs[i] = found / len(h)
        n_found += found
        n_total += len(h)
    fracs = doc_fracs[~np.isnan(doc_fracs)]
    return doc_fracs, {
        "n_docs": len(texts),
        "n_exact_duplicate_in_train": exact,
        "n_docs_with_13grams": len(fracs),
        "frac_docs_13gram_overlap>=0.5": float(np.mean(fracs >= 0.5)),
        "frac_docs_13gram_overlap>=0.8": float(np.mean(fracs >= 0.8)),
        "frac_docs_13gram_overlap>=0.99": float(np.mean(fracs >= 0.99)),
        "median_doc_13gram_overlap": float(np.median(fracs)),
        "frac_all_13grams_in_train": n_found / n_total,
    }

def save_overlap(splits, out_dir):
    train_texts = set(splits["train"])
    stats = {"n_train_docs": len(splits["train"]), "n_unique_train_docs": len(train_texts)}
    train_ngrams = np.unique(np.concatenate([ngram_hashes(t) for t in splits["train"]]))
    stats["n_unique_train_13grams"] = len(train_ngrams)
    for split in ["val", "test"]:
        doc_fracs, stats[split] = overlap_stats(splits[split], train_texts, train_ngrams)
        np.save(os.path.join(out_dir, "{}_13gram_overlap.npy".format(split)), doc_fracs)
    return stats

def main(args):
    if args.overlap_only:
        # recompute the duplicate statistics from the existing split files
        splits = {}
        for split in ["train", "val", "test"]:
            with open(os.path.join(args.out_dir, "{}.jsonl".format(split))) as f:
                splits[split] = [json.loads(line)["text"] for line in f]
        stats = save_overlap(splits, args.out_dir)
        print (json.dumps(stats, indent=2))
        with open(os.path.join(args.out_dir, "duplicate_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        return

    fns = sorted(fn for fn in os.listdir(args.raw_dir) if fn.endswith(".parquet"))
    assert len(fns) == 5, fns
    paragraphs = []
    for fn in fns:
        paragraphs += pq.read_table(os.path.join(args.raw_dir, fn), columns=["text"]).column("text").to_pylist()
    assert len(paragraphs) == N_EXPECTED, len(paragraphs)

    # identical to download_data.download_cc_news
    n_test_lines = 2000
    np.random.seed(2023)
    indices = np.random.permutation(range(len(paragraphs)))
    valid_indices = set(indices[:n_test_lines])
    test_indices = set(indices[n_test_lines:2*n_test_lines])

    splits = {"train": [], "val": [], "test": []}
    for i, text in enumerate(paragraphs):
        if i in valid_indices:
            splits["val"].append(text.strip())
        elif i in test_indices:
            splits["test"].append(text.strip())
        else:
            splits["train"].append(text.strip())
    del paragraphs

    os.makedirs(args.out_dir, exist_ok=True)
    for split, texts in splits.items():
        with open(os.path.join(args.out_dir, "{}.jsonl".format(split)), "w") as f:
            for text in texts:
                f.write(json.dumps({"text": text})+"\n")

    stats = save_overlap(splits, args.out_dir)
    print (json.dumps(stats, indent=2))

    with open(os.path.join(args.out_dir, "duplicate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # written last: its presence means the split files are complete
    open(os.path.join(args.out_dir, ".done"), "w").close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, help="directory with the 5 parquet files of vblagoje/cc_news")
    parser.add_argument("--out_dir", type=str, required=True, help="data/cc-news")
    parser.add_argument("--overlap_only", action="store_true",
                        help="only (re)compute duplicate statistics, incl. per-article overlap, from the existing split files")
    args = parser.parse_args()
    main(args)
