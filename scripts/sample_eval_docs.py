"""Random sample of whole val/test documents of a subset, for kNN-LM evaluation of large domains.

Wikipedia's val and test splits are ~12M tokens each; searching all of them at k=4096 would take
about a day per split and ~400 GB of neighbor files. This keeps a seeded random set of documents
(in their original order) until --target_tokens is reached and writes them as
{out_root}/out/neoX/{split}/{subset}-tokenized.pkl, so main.py and ensemble/ can run unchanged on
a work directory that holds the sample. The chosen document ids are saved next to it.

Inside a job:
    python3 scripts/sample_eval_docs.py --subset 'Wikipedia_(en)' \
        --src_root /work/hdd/bcyi/eergun/silo-knn --out_root /work/hdd/bcyi/eergun/silo-knn-wiki
"""
import os
import json
import argparse
import pickle as pkl

import numpy as np

def main(args):
    for split in ["val", "test"]:
        src = os.path.join(args.src_root, "out/neoX", split, "{}-tokenized.pkl".format(args.subset))
        out_dir = os.path.join(args.out_root, "out/neoX", split)
        out = os.path.join(out_dir, "{}-tokenized.pkl".format(args.subset))
        assert not os.path.exists(out), out
        with open(src, "rb") as f:
            docs = pkl.load(f)
        lengths = np.array([len(d) for d in docs])
        order = np.random.RandomState(args.seed).permutation(len(docs))
        n_keep = int(np.searchsorted(np.cumsum(lengths[order]), args.target_tokens)) + 1
        keep = np.sort(order[:n_keep])

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "{}-sampled_doc_ids.npy".format(args.subset)), keep)
        summary = {"source": src, "seed": args.seed, "target_tokens": args.target_tokens,
                   "n_docs": len(keep), "n_tokens": int(lengths[keep].sum()),
                   "n_docs_total": len(docs), "n_tokens_total": int(lengths.sum())}
        with open(os.path.join(out_dir, "{}-sample.json".format(args.subset)), "w") as f:
            json.dump(summary, f, indent=2)
        # written last: main.py only looks for the pkl
        with open(out, "wb") as f:
            pkl.dump([docs[i] for i in keep], f)
        print (split, json.dumps(summary))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--target_tokens", type=int, default=1100000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
