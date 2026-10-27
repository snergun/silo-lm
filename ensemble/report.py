# login-node-ok: reads prep.json, */metrics.json and overlap_breakdown.json (a few hundred KB), writes one .md; ~1 s
"""Collects the ensemble/train.py runs of every domain into one markdown report.

Reads only {work}/out/pythia-1.4B/ensemble/{subset}/prep.json and */metrics.json.

    python3 ensemble/report.py --out results/pythia_knn_ensemble.md
"""
import argparse
import glob
import json
import os

DOMAINS = [("Enron_Emails", "/work/hdd/bcyi/eergun/silo-knn"),
           ("cc-news", "/work/nvme/bcyi/eergun/silo-knn-ccnews")]
ARMS = [("fixed", "Fixed λ (kNN-LM, refit on fit slice)"), ("pct", "Fixed λ + PCT"),
        ("dawin", "DAWIN (b, τ_m)"), ("dawin_cov", "DAWIN (b, τ_m, c·top-1 score)"),
        ("adaptive", "Adaptive linear (MLP)"), ("nonlinear", "Non-linear")]

def load_runs(out_dir):
    runs = {}
    for path in glob.glob(os.path.join(out_dir, "*", "metrics.json")):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("smoke") or not os.path.exists(os.path.join(os.path.dirname(path), ".done")):
            continue
        with open(path) as f:
            runs[name] = json.load(f)
    return runs

def pct_change(ppl, ref):
    return "{:+.1f}%".format(100 * (ppl / ref - 1))

def domain_section(subset, work):
    out_dir = os.path.join(work, "out/pythia-1.4B/ensemble", subset)
    if not os.path.exists(os.path.join(out_dir, "prep.json")):
        return ["## {}".format(subset), "", "Not prepared yet.", ""]
    with open(os.path.join(out_dir, "prep.json")) as f:
        prep = json.load(f)
    runs = load_runs(out_dir)
    lines = ["## {}".format(subset), ""]
    fx, mb = prep["fixed"], prep["main_py_val_selected"]
    lines += ["Datastore {}; val {} positions, fit slice val[:{}] ({:.1%}), early stopping on the rest. "
              "Members: k={}, τ={} (chosen with λ on the fit slice). Neighbors: approximate (IVFPQ) scores.".format(
                  prep["datastore"], prep["n_val"], prep["cut"], prep["cut"] / prep["n_val"], fx["k"], fx["temperature"]), ""]
    if "fixed" not in runs:
        lines += ["No finished runs yet.", ""]
        return lines
    ref = runs["fixed"]["test"]["ppl"]
    lm_only = runs["fixed"]["test_lm_only_ppl"]
    p = prep["parity"]["test"]
    lines += ["Test perplexity (Δ against the fixed kNN-LM, {:.3f}):".format(ref), "",
              "| Method | test | Δ | +PCT test | Δ |", "|---|---:|---:|---:|---:|",
              "| Pythia-1.4B only | {:.3f} | {} | | |".format(lm_only, pct_change(lm_only, ref)),
              "| kNN-LM as in scripts/main.py (k={}, τ={}, λ={}, all of val) | {:.3f} | {} | | |".format(
                  mb["k"], mb["temperature"], mb["lambda"], p["knn_lm_ppl_main_py"], pct_change(p["knn_lm_ppl_main_py"], ref))]
    for arm, label in ARMS:
        if arm == "pct":
            continue
        base = runs.get(arm)
        withp = runs.get("pct") if arm == "fixed" else runs.get(arm + "+pct")
        cells = []
        for r in [base, withp]:
            cells += ["{:.3f}".format(r["test"]["ppl"]), pct_change(r["test"]["ppl"], ref)] if r else ["–", ""]
        lines.append("| {} | {} |".format(label, " | ".join(cells)))
    lines += ["", "<details><summary>Fit / early-stop perplexity, best step, mean kNN weight</summary>", "",
              "| Run | fit | early-stop | test | best step | mean kNN weight (test) |", "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(runs, key=lambda n: ([a for a, _ in ARMS].index(runs[n]["arm"]), n)):
        r = runs[name]
        w = r["test"].get("mean_knn_weight")
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            name, r["fit"]["ppl"], r["early_stop"]["ppl"], r["test"]["ppl"], r["best_step"],
            "{:.3f}".format(w) if w is not None else "–"))
    lines += ["", "</details>", ""]
    breakdown_path = os.path.join(out_dir, "overlap_breakdown.json")
    if os.path.exists(breakdown_path):
        with open(breakdown_path) as f:
            bd = json.load(f)
        names = [g for g in bd["groups"] if g != "<13 tokens"]
        lines += ["Test perplexity by how much of each document is in the datastore (token 13-gram overlap; "
                  "ensemble/overlap_breakdown.py). Δ is against the fixed kNN-LM in the same group; "
                  "share = the group's part of the run's total NLL reduction over the fixed kNN-LM.", "",
                  "| Run | " + " | ".join("{} ({:.0%} of tokens): PPL | Δ | share".format(g, bd["groups"][g]["frac_tokens"])
                                         for g in names) + " |",
                  "|---|" + "---:|---:|---:|" * len(names),
                  "| Pythia-1.4B only | " + " | ".join("{:.3f} | {} | ".format(
                      bd["groups"][g]["lm_only_ppl"], pct_change(bd["groups"][g]["lm_only_ppl"], bd["groups"][g]["fixed_ppl"]))
                      for g in names) + " |"]
        order = lambda n: ([a for a, _ in ARMS].index(runs[n]["arm"]) if n in runs else 99, n)
        for name in sorted(bd["runs"], key=order):
            cells = []
            for g in names:
                d = bd["runs"][name]["groups"][g]
                cells.append("{:.3f} | {:+.1f}% | {}".format(d["ppl"], 100 * d["change_vs_fixed"],
                             "{:.0%}".format(d["share_of_gain"]) if d["share_of_gain"] is not None else ""))
            lines.append("| {} | {} |".format(name, " | ".join(cells)))
        lines.append("")
    return lines

def main(args):
    lines = ["# Learned LM + kNN combiners: Pythia-1.4B + kNN-LM", "",
             "Generated by `ensemble/report.py`. All combiners are trained on the first ~90% of val (split at a "
             "document boundary), early-stopped on the remaining ~10%, and reported on test. PCT = piecewise "
             "temperature on both members, fitted jointly with λ on the fit slice and then frozen for the "
             "+PCT runs. Single seed (0).", "",
             "Reference (WikiText-103, ADT + kNN, test): fixed 16.11 → adaptive linear 15.30 (−5.1%); "
             "ADT + MEGA + SF + kNN: fixed 13.75 → +PCT 13.61 → adaptive 12.81 → non-linear 12.74.", ""]
    for subset, work in DOMAINS:
        lines += domain_section(subset, work)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print ("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/pythia_knn_ensemble.md")
    main(parser.parse_args())
