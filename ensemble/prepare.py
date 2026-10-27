"""Prepare a domain for ensemble/train.py, and check that this code reproduces scripts/main.py.

1. Picks the fixed kNN-LM (k, temperature, lambda) on the fit slice (the first ~90% of val, cut at a
   document boundary) from main.py's val p_kNN grid, with lambda on a 0.01 grid.
2. Writes {work}/out/pythia-1.4B/ensemble/{subset}/cache/{split}_*.npy: targets, LM losses, document
   starts, and the top-K neighbors' token ids and scores (K >= every k used here); then {split}.done.
3. Saves Pythia's output embedding (embed_out.pt), so full-vocabulary logits are h @ embed_out^T.
4. Parity (GPU): LM NLL recomputed from the saved hidden states against main.py's losses, and the fixed
   kNN-LM at main.py's val-selected configuration against its grid.json, on val and test.

Run from anywhere, inside a GPU job:
    PYTHONPATH=$REPO:$REPO/scripts python3 -m ensemble.prepare --work $WORK --subset Enron_Emails --last_idx 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ensemble.combiners import Fixed, Members
from ensemble.data import Paths, Split, load_embed, split_at_doc_boundary
from model import LanguageModel
from report_knn import token_doc_ids

FIT_RATIO = 0.9

def load_grid(paths, split):
    with open(paths.result_base(split) + ".grid.json") as f:
        return json.load(f)

def choose_fixed(paths, cut):
    """Best (k, temperature, lambda) on val[:cut]; also main.py's coarse-lambda choice on the same slice."""
    grid = load_grid(paths, "val")
    p_knn = np.load(paths.result_base("val") + ".pknn.npy", mmap_mode="r")
    assert p_knn.shape[:2] == (len(grid["ks"]), len(grid["temperatures"])), p_knn.shape
    p_lm = np.exp(-np.load(paths.loss("val"))[:cut].astype(np.float64))
    lambdas = np.round(np.arange(0, 1.0, 0.01), 2)
    best, best_coarse = None, None
    for i, k in enumerate(grid["ks"]):
        for j, temperature in enumerate(grid["temperatures"]):
            pk = np.asarray(p_knn[i, j, :cut], dtype=np.float64)
            nll = np.array([-np.mean(np.log(p_lm * (1 - l) + pk * l)) for l in lambdas])
            for l, v in zip(lambdas, nll):
                cand = (float(v), int(k), float(temperature), float(l))
                if best is None or cand < best:
                    best = cand
                if l in grid["lambdas"] and (best_coarse is None or cand < best_coarse):
                    best_coarse = cand
    to_dict = lambda c: {"k": c[1], "temperature": c[2], "lambda": c[3], "fit_ppl": float(np.exp(c[0]))}
    return to_dict(best), to_dict(best_coarse)

def write_cache(paths, split, dstore_targets, K, pad_token):
    done = os.path.join(paths.cache, split + ".done")
    if os.path.exists(done):
        print ("%s: cache exists" % split)
        return
    targets = np.load(paths.targets(split))
    losses = np.load(paths.loss(split))
    scores = np.load(paths.result_base(split) + ".scores.npy", mmap_mode="r")
    indices = np.load(paths.result_base(split) + ".indices.npy", mmap_mode="r")
    n = len(targets)
    assert n == len(losses) == len(scores) == len(indices) == len(load_embed(paths.embed(split))), split
    doc_ids = token_doc_ids(paths.tokenized(split), pad_token)
    assert len(doc_ids) == n
    doc_starts = np.concatenate([[0], np.flatnonzero(np.diff(doc_ids)) + 1])

    np.save(paths.cached(split, "targets"), targets)
    np.save(paths.cached(split, "lm_loss"), losses.astype(np.float32))
    np.save(paths.cached(split, "doc_starts"), doc_starts)
    np.save(paths.cached(split, "doc_ids"), doc_ids)
    vals_out = np.lib.format.open_memmap(paths.cached(split, "knn_vals"), mode="w+", dtype=np.int32, shape=(n, K))
    scores_out = np.lib.format.open_memmap(paths.cached(split, "knn_scores"), mode="w+", dtype=np.float32, shape=(n, K))
    t0 = time.time()
    for start in range(0, n, 65536):
        end = min(start + 65536, n)
        idx = np.asarray(indices[start:end, :K])
        vals_out[start:end] = np.where(idx >= 0, dstore_targets[np.maximum(idx, 0)], -1)
        scores_out[start:end] = scores[start:end, :K]
    vals_out.flush()
    scores_out.flush()
    del vals_out, scores_out
    open(done, "w").close()
    print ("%s: cached %d positions x %d neighbors (%d docs) in %ds" % (split, n, K, len(doc_starts), time.time() - t0))

@torch.no_grad()
def parity(paths, split, members, lambda_, k, batch_size=2048):
    data = Split(paths, split, k)
    head = Fixed(lambda_).cuda()
    lm_nll, mix_nll = [], []
    for start in range(0, len(data), batch_size):
        batch = data.batch(slice(start, start + batch_size), "cuda")
        lm_logp, knn_p, feats = members(batch)
        lm_nll.append(-lm_logp.gather(1, batch["targets"].unsqueeze(1)).squeeze(1).cpu())
        mix_nll.append(-head(lm_logp, knn_p, feats, batch["targets"])[0].cpu())
    lm_nll, mix_nll = torch.cat(lm_nll).double(), torch.cat(mix_nll).double()
    ref = data.lm_loss.double()
    return {"lm_ppl_recomputed": float(lm_nll.mean().exp()), "lm_ppl_main_py": float(ref.mean().exp()),
            "lm_nll_mean_abs_diff": float((lm_nll - ref).abs().mean()),
            "lm_nll_max_abs_diff": float((lm_nll - ref).abs().max()),
            "knn_lm_ppl": float(mix_nll.mean().exp())}

def main(args):
    paths = Paths(args.work, args.subset, args.last_idx, n=args.n)
    os.makedirs(paths.cache, exist_ok=True)
    lm = LanguageModel("pythia-1.4B")

    val_doc_ids = token_doc_ids(paths.tokenized("val"), lm.pad_token)
    val_doc_starts = np.concatenate([[0], np.flatnonzero(np.diff(val_doc_ids)) + 1])
    cut = split_at_doc_boundary(val_doc_starts, len(val_doc_ids), FIT_RATIO)
    fixed, fixed_coarse = choose_fixed(paths, cut)
    with open(paths.result_base("val") + ".grid.json") as f:
        val_grid = json.load(f)
    main_best = min(val_grid["ppl"], key=lambda d: d["ppl"])
    K = max(args.K, fixed["k"], fixed_coarse["k"], main_best["k"])
    print ("fit slice: val[:%d] of %d; fixed kNN-LM on it: %s (main.py's lambda grid: %s); K=%d" % (
        cut, len(val_doc_ids), fixed, fixed_coarse, K))

    dstore_targets = np.concatenate([np.load(p) for p in paths.dstore_target_paths])
    for split in ["val", "test"]:
        write_cache(paths, split, dstore_targets, K, lm.pad_token)
    del dstore_targets

    if not os.path.exists(paths.embed_out):
        lm.load_model(cuda=False)
        assert lm.model.embed_out.bias is None
        torch.save(lm.model.embed_out.weight.detach().float().clone(), paths.embed_out)
        lm.model = None

    torch.backends.cuda.matmul.allow_tf32 = False
    members = Members(torch.load(paths.embed_out), main_best["temperature"]).cuda()
    checks = {}
    for split in ["val", "test"]:
        checks[split] = parity(paths, split, members, main_best["lambda"], main_best["k"])
        grid = load_grid(paths, split)
        checks[split]["knn_lm_ppl_main_py"] = next(d["ppl"] for d in grid["ppl"] if
            (d["k"], d["temperature"], d["lambda"]) == (main_best["k"], main_best["temperature"], main_best["lambda"]))
        print (split, json.dumps(checks[split]))

    prep = {"subset": args.subset, "datastore": paths.datastore, "K": K, "fit_ratio": FIT_RATIO, "cut": cut,
            "n_val": len(val_doc_ids), "fixed": fixed, "fixed_main_py_lambdas": fixed_coarse,
            "main_py_val_selected": main_best, "parity": checks}
    with open(paths.prep, "w") as f:
        json.dump(prep, f, indent=2)
    print ("wrote %s" % paths.prep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, help="directory holding out/ and data/")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--last_idx", type=int, required=True, help="index of the datastore's last shard")
    parser.add_argument("--n", type=int, default=10, help="--max_n_sequences used to build the datastore")
    parser.add_argument("--K", type=int, default=1024, help="neighbors kept per position (at least every k used)")
    main(parser.parse_args())
