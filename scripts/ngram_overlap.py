"""Measure how much of each val/test document is duplicated in a domain's training data
(the datastore pool), as word 13-gram overlap; see prepare_cc_news.py for the definition.

Documents are loaded with data.load_pile, exactly as for tokenization, so the saved
per-document arrays line up with the evaluated tokens (scripts/report_knn.py --overlap_dir).
Run from the directory that holds data/.
"""
import os
import json
import argparse

from data import load_pile
from prepare_cc_news import save_overlap

def main(args):
    splits = {split: load_pile(split, subset=args.subset) for split in ["train", "val", "test"]}
    os.makedirs(args.out_dir, exist_ok=True)
    stats = save_overlap(splits, args.out_dir)
    print (json.dumps(stats, indent=2))
    with open(os.path.join(args.out_dir, "duplicate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="where to write {val,test}_13gram_overlap.npy and duplicate_stats.json")
    args = parser.parse_args()
    main(args)
