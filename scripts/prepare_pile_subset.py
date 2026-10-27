"""Build the training file of a Pile subset expected by data.load_pile_train.

The original release of SILO's Pile training subsets is no longer available, so this
rebuilds an equivalent one from the Pile's training documents of that subset
(ArmelR/the-pile-splitted; both of its train/ and test/ folders come from the Pile train split):
    1. drop exact duplicates (the Pile upsamples some subsets, e.g., Enron 2x and Wikipedia 3x)
    2. drop documents that also appear in the Pile val or test split
    3. shuffle and keep documents until the Pythia token count reaches --target_tokens

For large subsets (Wikipedia, Github) only the first --n_files arrow files of each folder are read.
The Pile train split is shuffled, so these are a random sample of the subset.
"""
import os
import io
import re
import sys
import json
import gzip
import argparse

import numpy as np
import pyarrow as pa
import zstandard as zstd
from transformers import AutoTokenizer

def read_arrow_texts(path):
    with pa.memory_map(path, "r") as source:
        try:
            table = pa.ipc.open_stream(source).read_all()
        except pa.lib.ArrowInvalid:
            table = pa.ipc.open_file(source).read_all()
    texts = table.column("text").to_pylist()
    names = [m["pile_set_name"] for m in table.column("meta").to_pylist()]
    return texts, names

def read_pile_split(path, pile_set_name):
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    texts = []
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            dp = json.loads(line)
            if dp["meta"]["pile_set_name"] == pile_set_name:
                texts.append(dp["text"].strip())
    return texts

def list_arrow_files(split_dir, n_files):
    # expects exactly the first n_files of data-XXXXX-of-NNNNN.arrow (all of them if n_files is None)
    fns = sorted(fn for fn in os.listdir(split_dir) if fn.endswith(".arrow"))
    assert len(fns) > 0, split_dir
    n_total = int(re.match(r"data-\d{5}-of-(\d{5})\.arrow$", fns[0]).group(1))
    n_expected = n_total if n_files is None else min(n_files, n_total)
    expected = ["data-%05d-of-%05d.arrow" % (i, n_total) for i in range(n_expected)]
    assert fns == expected, (split_dir, len(fns), n_expected)
    return fns

def main(args):
    stats = {"pile_set_name": args.pile_set_name}

    texts, n_other = [], 0
    for split in args.splits.split(","):
        split_dir = os.path.join(args.raw_dir, split)
        fns = list_arrow_files(split_dir, args.n_files)
        stats["n_{}_files".format(split)] = len(fns)
        for fn in fns:
            _texts, _names = read_arrow_texts(os.path.join(split_dir, fn))
            for text, name in zip(_texts, _names):
                if name == args.pile_set_name:
                    texts.append(text.strip())
                else:
                    n_other += 1
    stats["n_raw_docs"] = len(texts)
    stats["n_other_docs_dropped"] = n_other

    unique_texts = list(dict.fromkeys(t for t in texts if t))
    stats["n_unique_docs"] = len(unique_texts)
    del texts

    eval_texts = set()
    for split in ["val", "test"]:
        split_texts = read_pile_split(os.path.join(args.pile_dir, "{}.jsonl.zst".format(split)), args.pile_set_name)
        stats["n_{}_docs".format(split)] = len(split_texts)
        eval_texts.update(split_texts)
    assert len(eval_texts) > 0
    candidates = [t for t in unique_texts if t not in eval_texts]
    stats["n_docs_overlapping_val_test"] = len(unique_texts) - len(candidates)
    del unique_texts

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e12)  # we only count tokens here

    np.random.seed(args.seed)
    order = np.random.permutation(len(candidates))
    selected, n_tokens = [], 0
    chunk_size = 10000
    for chunk_start in range(0, len(order), chunk_size):
        chunk = [candidates[i] for i in order[chunk_start:chunk_start+chunk_size]]
        lengths = [len(ids) for ids in tokenizer(chunk)["input_ids"]]
        for text, length in zip(chunk, lengths):
            selected.append(text)
            n_tokens += length
            if n_tokens >= args.target_tokens:
                break
        if n_tokens >= args.target_tokens:
            break
    stats["n_candidate_docs"] = len(candidates)
    stats["n_selected_docs"] = len(selected)
    stats["n_selected_tokens"] = n_tokens
    print (json.dumps(stats, indent=2))
    if n_tokens < args.target_tokens:
        sys.exit("Only %d tokens available (target %d); increase --n_files" % (n_tokens, args.target_tokens))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with gzip.open(args.out_path, "wt", encoding="utf-8") as f:
        for text in selected:
            f.write(json.dumps({"text": text, "meta": {"pile_set_name": args.pile_set_name}}) + "\n")

    with open(args.out_path + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    # written last: its presence means the output file is complete
    open(args.out_path + ".done", "w").close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="directory with the arrow folders (see --splits) of one ArmelR/the-pile-splitted subset")
    parser.add_argument("--pile_dir", type=str, required=True,
                        help="directory with the Pile val.jsonl.zst and test.jsonl.zst")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--pile_set_name", type=str, default="Enron Emails")
    parser.add_argument("--splits", type=str, default="train,test",
                        help="comma-separated folders of --raw_dir to read")
    parser.add_argument("--n_files", type=int, default=None,
                        help="read only the first n arrow files of each folder (default: all)")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-1.4b-deduped-v0")
    parser.add_argument("--target_tokens", type=int, default=45000000)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    main(args)
