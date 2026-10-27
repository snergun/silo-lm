"""Synthetic checks for ensemble/combiners.py: every head's output is a normalised distribution over the
vocabulary, gradients are finite, and the degenerate cases reduce to the members.

    PYTHONPATH=$REPO python3 -m ensemble.test_combiners
"""
import torch

from ensemble.combiners import Members, Normalizer, build_head, head_forward, make_pct

V, D, B, K = 60, 8, 5, 6

def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = torch.randint(0, V, (B, K), generator=g, dtype=torch.int32)
    vals[0, :3] = 7          # repeated neighbor values
    vals[1, -1] = -1         # a missing neighbor
    scores = -torch.rand(B, K, generator=g).sort(dim=1, descending=True).values * 200
    return {"h": torch.randn(B, D, generator=g).half(), "knn_vals": vals, "knn_scores": scores,
            "targets": torch.randint(0, V, (B,), generator=g)}

def all_targets(batch):
    """The batch repeated once per vocabulary entry as target: row v * B + b scores target v at position b."""
    return {key: (torch.arange(V).repeat_interleave(B) if key == "targets" else x.repeat(V, *[1] * (x.dim() - 1)))
            for key, x in batch.items()}

def check(members, arm, lambda_=0.3):
    batch = make_batch()
    normalizer = Normalizer()
    normalizer.fit(members(make_batch(1))[2])
    head = build_head(arm, normalizer, lambda_, lm_topk=10)
    for p in head.parameters():   # move off the initialisation so every term is exercised
        p.data.add_(0.1 * torch.randn_like(p))

    rep = all_targets(batch)
    t_logp, _ = head_forward(head, *members(rep), rep)
    total = t_logp.view(V, B).exp().sum(0)
    assert torch.allclose(total, torch.ones(B), atol=1e-4), (arm, total)

    if arm == "nonlinear":
        lm_logp, knn_p, feats = members(batch)
        t, full = head(lm_logp, knn_p, feats, batch["targets"], batch["knn_vals"], return_full=True)
        assert torch.allclose(full.exp().sum(1), torch.ones(B), atol=1e-4)
        assert torch.allclose(full.gather(1, batch["targets"].unsqueeze(1)).squeeze(1), t, atol=1e-5)

    loss = -head_forward(head, *members(batch), batch)[0].mean()
    loss.backward()
    for name, p in list(head.named_parameters()) + list(members.named_parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (arm, name)
    print ("ok: %s%s" % (arm, " +pct" if members.pct_lm is not None else ""))

def main():
    torch.manual_seed(0)
    embed_out = torch.randn(V, D)
    members = Members(embed_out, tau=40.0)

    batch = make_batch()
    lm_logp, knn_p, _ = members(batch)
    assert torch.allclose(knn_p.sum(1), torch.ones(B), atol=1e-5)
    assert (knn_p[1].nonzero().squeeze(1) == batch["knn_vals"][1, :-1].long().unique()).all()

    # the fixed head is (1 - lambda) p_LM + lambda p_kNN
    fixed = build_head("fixed", Normalizer(), 0.25)
    t, _ = head_forward(fixed, lm_logp, knn_p, None, batch)
    tgt = batch["targets"].unsqueeze(1)
    expected = (0.75 * lm_logp.gather(1, tgt).exp() + 0.25 * knn_p.gather(1, tgt)).log().squeeze(1)
    assert torch.allclose(t, expected, atol=1e-5)

    for arm in ["fixed", "dawin", "dawin_cov", "adaptive", "nonlinear"]:
        check(members, arm)

    # PCT starts as the identity, and stays a valid distribution once its slopes move
    pct_members = Members(embed_out, tau=40.0, pct_lm=make_pct(20, -5.0, 5.0), pct_knn=make_pct(10, -5.0, 0.0))
    a, b = members(batch), pct_members(batch)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b))
    for p in pct_members.parameters():
        if p.requires_grad:   # the slopes; the knots are frozen
            p.data.mul_(1 + 0.3 * torch.rand_like(p))
    for arm in ["pct", "adaptive", "nonlinear"]:
        check(pct_members, arm)
    print ("all checks passed")

if __name__ == "__main__":
    main()
