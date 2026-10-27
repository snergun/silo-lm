"""Test perplexity of every finished ensemble/train.py run, split by how much of each test document is
duplicated in the kNN-LM datastore (token 13-gram overlap, scripts/dstore_ngram_overlap.py).

For each group: PPL of Pythia alone and of each run, the change against the fixed kNN-LM in that group,
and the group's share of the run's total NLL reduction over the fixed kNN-LM. Writes
{work}/out/pythia-1.4B/ensemble/{subset}/overlap_breakdown.json, which ensemble/report.py reads.

Inside a job (it loads per-token NLL arrays):
    PYTHONPATH=$REPO python3 -m ensemble.overlap_breakdown --work $WORK --subset Enron_Emails --last_idx 8
"""
import argparse
import glob
import json
import os

import numpy as np

from ensemble.data import Paths

def groups_for(overlap):
    return {"<50% in datastore": overlap < 0.5, ">=50% in datastore": overlap >= 0.5,
            "<13 tokens": np.isnan(overlap)}

def main(args):
    paths = Paths(args.work, args.subset, args.last_idx, n=args.n)
    overlap_path = os.path.join(args.work, "data", "{}-dstore-overlap".format(args.subset), "test_13gram_overlap.npy")
    overlap = np.load(overlap_path)[np.load(paths.cached("test", "doc_ids"))]
    lm_nll = np.load(paths.cached("test", "lm_loss")).astype(np.float64)
    groups = {g: m for g, m in groups_for(overlap).items() if m.any()}

    runs = {}
    for path in sorted(glob.glob(os.path.join(paths.out, "*", "test_nll.npy"))):
        name = os.path.basename(os.path.dirname(path))
        if not name.startswith("smoke") and os.path.exists(os.path.join(os.path.dirname(path), ".done")):
            runs[name] = np.load(path).astype(np.float64)
            assert len(runs[name]) == len(lm_nll), name
    assert "fixed" in runs, "the fixed kNN-LM run is the reference"
    ref = runs["fixed"]

    out = {"overlap_file": overlap_path, "n_tokens": len(lm_nll),
           "groups": {g: {"n_tokens": int(m.sum()), "frac_tokens": float(m.mean()),
                          "lm_only_ppl": float(np.exp(lm_nll[m].mean())),
                          "fixed_ppl": float(np.exp(ref[m].mean()))} for g, m in groups.items()},
           "runs": {}}
    for name, nll in runs.items():
        gain = ref - nll   # per-token NLL reduction over the fixed kNN-LM
        out["runs"][name] = {
            "test_ppl": float(np.exp(nll.mean())),
            "groups": {g: {"ppl": float(np.exp(nll[m].mean())),
                           "change_vs_fixed": float(np.exp(nll[m].mean() - ref[m].mean()) - 1),
                           "share_of_gain": float(gain[m].sum() / gain.sum()) if name != "fixed" and gain.sum() != 0 else None}
                       for g, m in groups.items()}}
    with open(os.path.join(paths.out, "overlap_breakdown.json"), "w") as f:
        json.dump(out, f, indent=2)

    print ("%s, test by datastore overlap (%s)" % (args.subset, overlap_path))
    print ("  " + "  ".join("%s: %d tokens, LM %.3f, fixed %.3f" % (g, d["n_tokens"], d["lm_only_ppl"], d["fixed_ppl"])
                            for g, d in out["groups"].items()))
    for name, r in out["runs"].items():
        print ("  %-16s %s" % (name, "  ".join("%s %.3f (%+.1f%%)" % (g, d["ppl"], 100 * d["change_vs_fixed"])
                                               for g, d in r["groups"].items())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--last_idx", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    main(parser.parse_args())
