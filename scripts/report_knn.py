"""Summarize a kNN-LM evaluation run by scripts/main.py --task inference (on val and test).

Follows the paper's protocol: (k, temperature, lambda) are chosen on the validation data and
the test perplexity is reported with those values. With --overlap_dir, perplexities are also broken
down by how much of each evaluated document is duplicated in the training data
(scripts/ngram_overlap.py, or scripts/prepare_cc_news.py --overlap_only for cc-news).

Run from the directory that holds out/ and data/, inside a job (it loads probability arrays).
"""
import os
import json
import argparse
import pickle as pkl

import numpy as np

from model import LanguageModel

# Pythia-1.4B numbers from the paper: param-only from the test and validation results tables,
# kNN-LM read off the appendix datastore-scaling plot (figs/baseline_CC_News_Pythia_scale.pdf)
PAPER = {
    "cc-news": {"param_only_val": 10.9, "param_only_test": 12.0,
                "knn_val_51.2M_tokens (approx., from plot)": 8.3},
    "Enron_Emails": {"param_only_val": 7.6, "param_only_test": 6.9},
}

def load_grid(result_base):
    with open(result_base + ".grid.json") as f:
        grid = json.load(f)
    grid["ppl"] = {(d["k"], d["temperature"], d["lambda"]): d["ppl"] for d in grid["ppl"]}
    return grid

def token_doc_ids(tokenized_path, pad_token):
    """Article id of every evaluated token, in the order of main.py's flattened targets:
    targets are positions 1..L-1 of the concatenated articles, minus those equal to the pad token."""
    with open(tokenized_path, "rb") as f:
        input_ids = pkl.load(f)
    flatten_input_ids = np.concatenate([np.array(ids) for ids in input_ids])
    doc_ids = np.concatenate([np.full(len(ids), i) for i, ids in enumerate(input_ids)])
    keep = flatten_input_ids[1:] != pad_token
    return doc_ids[1:][keep]

def ppl(losses):
    return float(np.exp(np.mean(losses)))

def main(args):
    lm = LanguageModel(args.lm)
    out_dir = os.path.join(args.output_dir, args.lm, "train")
    postfix = "{}-{}-[0K-{}K]".format(args.max_seq_length, args.stride, args.max_n_sequences * (args.embed_idx + 1))
    report = {"subset": args.subset, "datastore": postfix, "paper": PAPER.get(args.subset, {})}

    grids, result_bases = {}, {}
    for split in ["val", "test"]:
        result_bases[split] = os.path.join(out_dir, "{}-{}-{}-k={}-p={}{}".format(
            args.subset, split, postfix, args.k, args.probe, "_approx" if args.approximate else ""))
        grids[split] = load_grid(result_bases[split])
        report["param_only_" + split] = grids[split]["param_only_ppl"]
        report["n_tokens_" + split] = grids[split]["n_tokens"]

    best = min(grids["val"]["ppl"], key=grids["val"]["ppl"].get)
    report["selected_on_val"] = {"k": best[0], "temperature": best[1], "lambda": best[2]}
    for split in ["val", "test"]:
        report["knn_" + split] = grids[split]["ppl"][best]

    if args.overlap_dir is not None:
        report["by_overlap"] = {}
        k, temperature, lambda_ = best
        for split in ["val", "test"]:
            grid = grids[split]
            p_knn = np.load(result_bases[split] + ".pknn.npy", mmap_mode="r")
            p_knn = np.asarray(p_knn[grid["ks"].index(k), grid["temperatures"].index(temperature)], dtype=np.float64)
            losses = np.load(os.path.join(args.output_dir, args.lm, split,
                                          "{}-{}-{}-loss.npy".format(args.subset, args.max_seq_length, args.stride)))
            knn_losses = -np.log(np.exp(-losses) * (1 - lambda_) + p_knn * lambda_)

            doc_ids = token_doc_ids(os.path.join(args.output_dir, "neoX", split, "{}-tokenized.pkl".format(args.subset)),
                                    lm.pad_token)
            assert len(doc_ids) == len(losses) == len(p_knn), (len(doc_ids), len(losses), len(p_knn))
            token_overlap = np.load(os.path.join(args.overlap_dir, "{}_13gram_overlap.npy".format(split)))[doc_ids]

            groups = {"docs <50% duplicated": token_overlap < 0.5,
                      "docs >=50% duplicated": token_overlap >= 0.5,
                      "docs <13 words": np.isnan(token_overlap)}
            report["by_overlap"][split] = {
                name: {"n_tokens": int(mask.sum()),
                       "param_only": ppl(losses[mask]) if mask.any() else None,
                       "knn": ppl(knn_losses[mask]) if mask.any() else None}
                for name, mask in groups.items()}

    report_path = os.path.join(out_dir, "{}-{}.json".format(args.subset, args.report_name))
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# kNN-LM evaluation: {} (Pythia-1.4B, datastore {})".format(args.subset, postfix), "",
             "Selected on val: k={k}, temperature={temperature}, lambda={lambda}".format(**report["selected_on_val"]), "",
             "| | val | test |", "|---|---|---|",
             "| Param-only | {:.3f} | {:.3f} |".format(report["param_only_val"], report["param_only_test"]),
             "| kNN-LM | {:.3f} | {:.3f} |".format(report["knn_val"], report["knn_test"])]
    if report["paper"]:
        lines += ["", "Paper: " + ", ".join("{}={}".format(key, value) for key, value in report["paper"].items())]
    for split, groups in report.get("by_overlap", {}).items():
        lines += ["", "{} by duplication in train (13-gram overlap):".format(split), "",
                  "| | tokens | param-only | kNN-LM |", "|---|---|---|---|"]
        for name, g in groups.items():
            if g["n_tokens"]:
                lines.append("| {} | {} | {:.3f} | {:.3f} |".format(name, g["n_tokens"], g["param_only"], g["knn"]))
    with open(report_path.replace(".json", ".md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print ("\n".join(lines))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--lm", type=str, default="pythia-1.4B")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--max_n_sequences", default=10, type=int)
    parser.add_argument("--embed_idx", type=int, required=True)
    parser.add_argument("--k", default=4096, type=int)
    parser.add_argument("--probe", default=8, type=int)
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--report_name", type=str, default="knn_eval_report",
                        help="written to out/{lm}/train/{subset}-{report_name}.{json,md}")
    parser.add_argument("--overlap_dir", type=str, default=None,
                        help="directory with {val,test}_13gram_overlap.npy (from scripts/ngram_overlap.py)")
    args = parser.parse_args()
    main(args)
