"""Train one LM + kNN combiner on the fit slice of val, early-stop on the rest of val, report test.

Arms (ensemble/combiners.py): fixed (the kNN-LM chosen in prepare.py; nothing is trained), pct (piecewise
temperatures on both members trained jointly with the fixed lambda; saves pct.pt), dawin, dawin_cov,
adaptive, nonlinear. --pct PATH loads a pct.pt into the members, frozen.

Writes {work}/out/pythia-1.4B/ensemble/{subset}/{name}/: metrics.json, test_nll.npy, best.pt
(+ pct.pt for the pct arm), and .done last.

    PYTHONPATH=$REPO:$REPO/scripts python3 -m ensemble.train --work $WORK --subset Enron_Emails --last_idx 8 --arm adaptive
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from ensemble.combiners import Members, Normalizer, build_head, head_forward, make_pct
from ensemble.data import Paths, Split, load_prep

DEFAULTS = {  # arm: (lr, batch size); the pos_lm_v2 stage_b / pile_diverse recipe, and lr 0.008 for PCT
    "fixed": (0.0, 2048), "pct": (8e-3, 1024), "dawin": (1e-2, 1024), "dawin_cov": (1e-2, 1024),
    "adaptive": (1e-4, 1024), "nonlinear": (1e-4, 512)}

def pct_ranges(members, data, idx, n_rows=256):
    """Knot ranges for the PCTs: LM logits from their 0.1% quantile to their max, kNN scores / tau
    between their 1% and 99% quantiles, on a sample of fit-slice rows."""
    with torch.no_grad():
        batch = data.batch(idx[:n_rows], "cuda")
        logits = (batch["h"].float() @ members.embed_out.t()).flatten()
        s = (batch["knn_scores"] / members.tau).flatten()
        sample = logits[torch.randperm(len(logits), device=logits.device)[:1000000]]
        return {"lm": [float(sample.quantile(0.001)), float(logits.max())],
                "knn": [float(s.quantile(0.01)), float(s.quantile(0.99))]}

def evaluate(members, head, data, idx, batch_size, return_nll=False):
    head.eval()
    nll, w_knn = [], []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = data.batch(idx[start:start + batch_size], "cuda")
            lm_logp, knn_p, feats = members(batch)
            t_logp, w = head_forward(head, lm_logp, knn_p, feats, batch)
            nll.append(-t_logp.double().cpu())
            if w is not None:
                w_knn.append(w.double().cpu())
    head.train()
    nll = torch.cat(nll)
    out = {"nll": float(nll.mean()), "ppl": float(nll.mean().exp())}
    if w_knn:
        out["mean_knn_weight"] = float(torch.cat(w_knn).mean())
    return (out, nll.float().numpy()) if return_nll else out

def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = Paths(args.work, args.subset, args.last_idx, n=args.n)
    prep = load_prep(paths)
    fixed = prep["fixed"]
    k, tau = fixed["k"], fixed["temperature"]
    lr, batch_size = DEFAULTS[args.arm]
    lr = args.lr if args.lr is not None else lr
    batch_size = args.batch_size or batch_size
    name = args.name or (args.arm + ("+pct" if args.pct else ""))
    out_dir = os.path.join(paths.out, name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    val = Split(paths, "val", k)
    cut = prep["cut"]
    fit_idx, es_idx = torch.arange(cut), torch.arange(cut, len(val))
    print ("loaded val (%d positions; fit %d, early-stop %d) in %ds; k=%d tau=%s lambda=%s" % (
        len(val), len(fit_idx), len(es_idx), time.time() - t0, k, tau, fixed["lambda"]))

    members = Members(torch.load(paths.embed_out), tau).cuda()
    lambda_ = fixed["lambda"]
    pct_config = None
    if args.arm == "pct":
        pct_config = {"lm_pieces": args.lm_pieces, "knn_pieces": args.knn_pieces,
                      **{key + "_range": r for key, r in pct_ranges(members, val, fit_idx).items()}}
        members.pct_lm = make_pct(args.lm_pieces, *pct_config["lm_range"]).cuda()
        members.pct_knn = make_pct(args.knn_pieces, *pct_config["knn_range"]).cuda()
    elif args.pct:
        saved = torch.load(args.pct)
        pct_config = saved["config"]
        members.pct_lm = make_pct(pct_config["lm_pieces"], *pct_config["lm_range"]).cuda()
        members.pct_knn = make_pct(pct_config["knn_pieces"], *pct_config["knn_range"]).cuda()
        members.pct_lm.load_state_dict(saved["pct_lm"])
        members.pct_knn.load_state_dict(saved["pct_knn"])
        lambda_ = saved["lambda"]   # the lambda fitted jointly with these temperatures
        for p in members.parameters():
            p.requires_grad_(False)
    print ("PCT: %s" % (json.dumps(pct_config) if pct_config else "none"))

    normalizer = Normalizer().cuda()
    with torch.no_grad():
        sample = fit_idx[torch.randperm(len(fit_idx))[:65536]].sort().values
        normalizer.fit(torch.cat([members(val.batch(sample[s:s + 2048], "cuda"))[2]
                                  for s in range(0, len(sample), 2048)]))
    head = build_head(args.arm, normalizer, lambda_, lm_topk=args.lm_topk).cuda()

    params = [p for p in list(head.parameters()) + list(members.parameters()) if p.requires_grad]
    print ("arm=%s: %d trainable parameters, lr=%g, batch size %d" % (
        args.arm, sum(p.numel() for p in params), lr, batch_size))
    eval_bs = batch_size if args.arm == "nonlinear" else 2048
    history = []
    best = {"es": evaluate(members, head, val, es_idx, eval_bs), "step": 0,
            "state": copy.deepcopy({"head": head.state_dict(), "members": members.state_dict()})}
    print ("step 0: early-stop ppl %.4f" % best["es"]["ppl"])

    if args.arm != "fixed":
        opt = torch.optim.Adam(params, lr=lr) if args.arm == "pct" else \
              torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        step, bad, train_nll = 0, 0, []
        while step < args.max_steps and bad < args.patience:
            perm = fit_idx[torch.randperm(len(fit_idx))]
            for start in range(0, len(perm) - batch_size + 1, batch_size):
                batch = val.batch(perm[start:start + batch_size].sort().values, "cuda")
                lm_logp, knn_p, feats = members(batch)
                loss = -head_forward(head, lm_logp, knn_p, feats, batch)[0].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                train_nll.append(loss.item())
                step += 1
                if step % args.eval_interval == 0:
                    es = evaluate(members, head, val, es_idx, eval_bs)
                    history.append({"step": step, "train_nll": float(np.mean(train_nll)), **{"es_" + key: v for key, v in es.items()}})
                    train_nll = []
                    improved = es["nll"] < best["es"]["nll"] - 1e-5
                    if improved:
                        best = {"es": es, "step": step,
                                "state": copy.deepcopy({"head": head.state_dict(), "members": members.state_dict()})}
                        bad = 0
                    else:
                        bad += 1
                    print ("step %d: train nll %.4f, early-stop ppl %.4f%s (%ds)" % (
                        step, history[-1]["train_nll"], es["ppl"], " *" if improved else "", time.time() - t0), flush=True)
                if step >= args.max_steps or bad >= args.patience:
                    break

    head.load_state_dict(best["state"]["head"])
    members.load_state_dict(best["state"]["members"])
    torch.save(best["state"], os.path.join(out_dir, "best.pt"))
    if args.arm == "pct":
        torch.save({"config": pct_config, "pct_lm": members.pct_lm.state_dict(),
                    "pct_knn": members.pct_knn.state_dict(),
                    "lambda": float(torch.softmax(head.w.detach(), -1)[1])}, os.path.join(out_dir, "pct.pt"))

    fit = evaluate(members, head, val, fit_idx, eval_bs)
    del val
    test = Split(paths, "test", k)
    test_metrics, test_nll = evaluate(members, head, test, torch.arange(len(test)), eval_bs, return_nll=True)
    np.save(os.path.join(out_dir, "test_nll.npy"), test_nll)
    metrics = {"subset": args.subset, "arm": args.arm, "pct": args.pct, "name": name, "k": k, "tau": tau,
               "lambda_init": lambda_, "lr": lr, "batch_size": batch_size, "seed": args.seed,
               "best_step": best["step"], "fit": fit, "early_stop": best["es"], "test": test_metrics,
               "test_lm_only_ppl": float(test.lm_loss.double().mean().exp()), "pct_config": pct_config,
               "history": history, "seconds": time.time() - t0}
    overlap_path = paths.overlap("test")
    if overlap_path is not None:
        # test perplexity by how much of each document is duplicated in train (as scripts/report_knn.py)
        token_overlap = np.load(overlap_path)[np.load(paths.cached("test", "doc_ids"))]
        lm_nll = test.lm_loss.numpy().astype(np.float64)
        groups = {"<50% duplicated": token_overlap < 0.5, ">=50% duplicated": token_overlap >= 0.5,
                  "<13 words": np.isnan(token_overlap)}
        metrics["test_by_overlap"] = {
            g: {"n_tokens": int(m.sum()), "ppl": float(np.exp(test_nll[m].astype(np.float64).mean())),
                "lm_only_ppl": float(np.exp(lm_nll[m].mean()))} for g, m in groups.items() if m.any()}
    if args.arm in ["fixed", "pct"]:
        metrics["lambda"] = float(torch.softmax(head.w.detach(), -1)[1])
    if args.arm.startswith("dawin"):
        metrics["dawin"] = {key: v.detach().cpu().tolist() for key, v in head.named_parameters()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    open(os.path.join(out_dir, ".done"), "w").close()
    print ("%s: best step %d; fit ppl %.4f, early-stop ppl %.4f, test ppl %.4f (LM only %.4f)" % (
        name, best["step"], fit["ppl"], best["es"]["ppl"], test_metrics["ppl"], metrics["test_lm_only_ppl"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--last_idx", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--arm", required=True, choices=list(DEFAULTS))
    parser.add_argument("--pct", default=None, help="pct.pt from a pct run; applied to the members, frozen")
    parser.add_argument("--name", default=None, help="output directory name (default: arm[+pct])")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=96)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lm_pieces", type=int, default=20)
    parser.add_argument("--knn_pieces", type=int, default=10)
    parser.add_argument("--lm_topk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
