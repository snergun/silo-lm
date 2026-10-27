"""Ways of combining Pythia's next-token distribution with the kNN-LM's.

Ported from pos_lm_v2 (pos_lm/models/pw_ensemble.py; the learnable DAWIN from the pile-diverse-ensemble
worktree), specialised to one LM plus kNN:

- fixed: p = (1-lambda) p_LM + lambda p_kNN, the kNN-LM itself
- dawin: per-token mixture weights softmax_m(b_m - H_m / tau_m [+ c_m . covariates]), H_m = member entropy
- adaptive: per-token mixture weights from an MLP over the token features
- nonlinear: an MLP scores every token in (LM top-n union kNN support) from both members' log-probs and
  the token features; the mass left outside the union is spread in proportion to p_LM

Members (Members below) can each be recalibrated with a piecewise temperature (PCT, ensemble/temperature.py):
on the LM's logits, and on the kNN-LM's neighbor scores / tau before their softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ensemble.temperature import MLP, TemperatureScaler

N_FEATURES = 5   # LM entropy, kNN entropy, top-3 neighbor scores / tau

def log1mexp(x):
    # log(1 - exp(x)) for x <= 0 (from pos_lm_v2's pos_lm/utils.py)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))

def make_pct(pieces, lo, hi):
    """Piecewise temperature with `pieces` slopes, knots evenly spaced and frozen inside [lo, hi] (the
    best-transferring configuration in pos_lm_v2's results/pct_consistency_report.md); starts as the identity."""
    return TemperatureScaler(mode="absolute", pieces=pieces, init_temp="ones", init_thresholds="linspace",
                             threshold_min=lo, threshold_max=hi, fixed_thresholds=True)

def apply_pct(pct, x):
    # plotting=True only skips caching a copy of every batch for plots; no option used here depends on it
    return x if pct is None else pct(x, plotting=True)

class Members(nn.Module):
    """Full-vocabulary log-probs of the LM and the kNN-LM for a batch of evaluated positions."""

    def __init__(self, embed_out, tau, pct_lm=None, pct_knn=None):
        super().__init__()
        # (V, d), frozen; kept out of state_dict so checkpoints hold only the temperatures
        self.register_buffer("embed_out", embed_out.float(), persistent=False)
        self.tau = tau
        self.pct_lm, self.pct_knn = pct_lm, pct_knn

    @property
    def vocab_size(self):
        return self.embed_out.size(0)

    def forward(self, batch):
        logits = apply_pct(self.pct_lm, batch["h"].float() @ self.embed_out.t())
        lm_logp = torch.log_softmax(logits, dim=-1)                                  # (B, V)

        vals, scores = batch["knn_vals"].long(), batch["knn_scores"]                 # (B, k)
        s = apply_pct(self.pct_knn, scores / self.tau).masked_fill(vals < 0, float("-inf"))
        knn_w = torch.softmax(s, dim=-1)
        knn_p = lm_logp.new_zeros(lm_logp.shape).scatter_add(1, vals.clamp_min(0), knn_w)   # (B, V), sums to 1

        h_lm = -(lm_logp.exp() * lm_logp).sum(-1)
        h_knn = -(knn_p * knn_p.clamp_min(1e-30).log()).sum(-1)
        feats = torch.cat([h_lm.unsqueeze(1), h_knn.unsqueeze(1), scores[:, :3] / self.tau], dim=1)  # (B, 5)
        return lm_logp, knn_p, feats

def target_logps(lm_logp, knn_p, targets):
    lm_t = lm_logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    # clamped rather than -inf when no neighbor has the target, so gradients stay finite
    knn_t = knn_p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-30).log()
    return lm_t, knn_t

def mix(log_weights, lm_t, knn_t):
    """log sum_m w_m p_m(target), with log_weights (B, 2) or (2,) over (LM, kNN)."""
    return torch.logsumexp(torch.log_softmax(log_weights, dim=-1) + torch.stack([lm_t, knn_t], dim=1), dim=1)

class Normalizer(nn.Module):
    """z-scores the token features with statistics of the fit slice (neighbor scores are in the hundreds)."""

    def __init__(self, n=N_FEATURES):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))

    def fit(self, feats):
        self.mean.copy_(feats.mean(0))
        self.std.copy_(feats.std(0).clamp_min(1e-6))

    def forward(self, feats):
        return (feats - self.mean) / self.std

class Fixed(nn.Module):

    def __init__(self, lambda_):
        super().__init__()
        lambda_ = min(max(lambda_, 1e-4), 1 - 1e-4)
        self.w = nn.Parameter(torch.tensor([0.0, float(torch.logit(torch.tensor(lambda_)))]))

    def forward(self, lm_logp, knn_p, feats, targets):
        return mix(self.w, *target_logps(lm_logp, knn_p, targets)), torch.softmax(self.w, -1)[1].expand(len(targets))

class Dawin(nn.Module):
    """softmax_m(b_m - H_m / tau_m [+ c_m . z(top-1 score)]); b_0 and c_0 are pinned to 0 (softmax shift
    invariance). At initialisation (tau = 1, b = c = 0) this is plain DAWIN, softmax(-H)."""

    def __init__(self, normalizer, covariate=False):
        super().__init__()
        self.normalizer = normalizer
        self.log_tau = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(1))
        self.cov = nn.Parameter(torch.zeros(1)) if covariate else None

    def forward(self, lm_logp, knn_p, feats, targets):
        w = -feats[:, :2] / self.log_tau.exp()
        w = w + torch.cat([self.bias.new_zeros(1), self.bias])
        if self.cov is not None:
            top1 = self.normalizer(feats)[:, 2:3]
            w = w + top1 * torch.cat([self.cov.new_zeros(1), self.cov]).unsqueeze(0)
        return mix(w, *target_logps(lm_logp, knn_p, targets)), torch.softmax(w, -1)[:, 1]

class Adaptive(nn.Module):
    """Mixture weights from an MLP over the z-scored token features. The output layer starts at zero
    weights with the fixed lambda as bias, so training starts from the kNN-LM."""

    def __init__(self, normalizer, lambda_, d_inner=256, n_layer=3):
        super().__init__()
        self.normalizer = normalizer
        self.mlp = MLP(N_FEATURES, 2, d_inner, n_layer, dropout=0.0)
        self.mlp.out_proj.weight.data.zero_()
        lambda_ = min(max(lambda_, 1e-4), 1 - 1e-4)
        self.mlp.out_proj.bias.data.copy_(torch.tensor([0.0, float(torch.logit(torch.tensor(lambda_)))]))

    def forward(self, lm_logp, knn_p, feats, targets):
        w = self.mlp(self.normalizer(feats))
        return mix(w, *target_logps(lm_logp, knn_p, targets)), torch.softmax(w, -1)[:, 1]

class NonLinear(nn.Module):
    """Scores each token in the union of the LM's top-`lm_topk` tokens and the kNN support with an MLP,
    one MLP for kNN-supported tokens (input: LM log-prob, kNN log-prob, token features) and another for
    LM-only tokens (LM log-prob, token features) -- pos_lm_v2's `separate_groups`. A learnt score for the
    rest of the vocabulary, log p_LM(outside union) + bias, is normalised together with them, and that
    mass is spread over the outside tokens in proportion to p_LM, so the output is a full distribution."""

    def __init__(self, normalizer, lm_topk=1024, d_inner=256, n_layer=3):
        super().__init__()
        self.normalizer = normalizer
        self.lm_topk = lm_topk
        self.mlp_knn = MLP(2 + N_FEATURES, 1, d_inner, n_layer, dropout=0.0)
        self.mlp_lm = MLP(1 + N_FEATURES, 1, d_inner, n_layer, dropout=0.0)
        self.out_bias = nn.Parameter(torch.zeros(1))

    def candidates(self, lm_logp, knn_vals):
        """(B, C) candidate token ids, sorted, with `valid` marking the first copy of each real id."""
        cand = torch.cat([knn_vals.long(), lm_logp.topk(self.lm_topk, dim=-1).indices], dim=1)
        cand, _ = cand.sort(dim=1)
        valid = cand >= 0
        valid[:, 1:] &= cand[:, 1:] != cand[:, :-1]
        return cand, valid

    def forward(self, lm_logp, knn_p, feats, targets, knn_vals, return_full=False):
        cand, valid = self.candidates(lm_logp, knn_vals)
        safe = cand.clamp_min(0)
        lm_c = lm_logp.gather(1, safe)                                     # (B, C)
        kp = knn_p.gather(1, safe)
        in_knn = (kp > 0) & valid
        knn_c = torch.where(in_knn, kp.clamp_min(1e-30).log(), torch.full_like(kp, -100.0))

        tok = self.normalizer(feats).unsqueeze(1).expand(-1, cand.size(1), -1)
        s_knn = self.mlp_knn(torch.cat([lm_c.unsqueeze(-1), knn_c.unsqueeze(-1), tok], dim=-1)).squeeze(-1)
        s_lm = self.mlp_lm(torch.cat([lm_c.unsqueeze(-1), tok], dim=-1)).squeeze(-1)
        s = torch.where(in_knn, s_knn, s_lm).masked_fill(~valid, float("-inf"))

        lm_p_in = torch.logsumexp(lm_c.masked_fill(~valid, float("-inf")), dim=1).clamp_max(-1e-7)
        lm_p_out = log1mexp(lm_p_in)                                       # log p_LM(outside union)
        logp = torch.log_softmax(torch.cat([s, (lm_p_out + self.out_bias).unsqueeze(1)], dim=1), dim=1)
        out_logp = logp[:, -1]

        hit = (cand == targets.unsqueeze(1)) & valid
        in_union = hit.any(1)
        t_in = logp.gather(1, hit.float().argmax(1, keepdim=True)).squeeze(1)
        lm_t = lm_logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        t_logp = torch.where(in_union, t_in, lm_t - lm_p_out + out_logp)
        if not return_full:
            return t_logp, None
        full = lm_logp - lm_p_out.unsqueeze(1) + out_logp.unsqueeze(1)
        rows, cols = valid.nonzero(as_tuple=True)   # valid ids are unique within a row
        full[rows, cand[rows, cols]] = logp[rows, cols]
        return t_logp, full

def build_head(arm, normalizer, lambda_, lm_topk=1024):
    if arm in ["fixed", "pct"]:
        return Fixed(lambda_)
    if arm == "dawin":
        return Dawin(normalizer)
    if arm == "dawin_cov":
        return Dawin(normalizer, covariate=True)
    if arm == "adaptive":
        return Adaptive(normalizer, lambda_)
    if arm == "nonlinear":
        return NonLinear(normalizer, lm_topk=lm_topk)
    raise ValueError(arm)

def head_forward(head, lm_logp, knn_p, feats, batch):
    """(target log-prob, per-token kNN weight or None) for any head."""
    if isinstance(head, NonLinear):
        return head(lm_logp, knn_p, feats, batch["targets"], batch["knn_vals"])
    return head(lm_logp, knn_p, feats, batch["targets"])
