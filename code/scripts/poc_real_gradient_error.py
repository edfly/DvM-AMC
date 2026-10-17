"""
POC-2 (decisive): measure the REAL truncated-vs-full BPTT gradient error on the
actual SpikeRPGNN model.

Why this is not circular
------------------------
programfiles/stmp.py L212-215 implements truncation the honest way:

    if self.use_dvm and self.dvm_mode == "truncated" and self.training:
        if (self.T - 1 - t) >= self.dvm_window:
            p_hyp = p_hyp.detach(); p_euc = p_euc.detach()

States older than the window leave the autograd graph; NO theoretical factor
(rho^k, alpha^k) is ever multiplied onto the gradient. The metric correction is
wired in-graph via PerStepMetricCorrection (L195-197), so it reaches the weights
through the genuine chain rule. Therefore the truncated gradient is whatever it
turns out to be -- it is *not* algebraically forced to reproduce Theorem 1.

Comparing it against a full-BPTT reference on the same weights and the same batch
(dvm_mode="full_bptt_mc") yields a data-dependent empirical error. If that error
tracks the shape predicted by 1-rho^(T-M) it is genuine corroboration; if it does
not, Theorem 1 is simply not supported by this measurement and must be softened.
Either outcome is informative -- which is exactly what a *non-circular* test means.
"""
import json
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "programfiles"))

from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler, batch_episode_data)
from msg_baseline.utils import set_seed  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}")


def named_grads(model):
    """Dict name -> grad for every parameter that received one."""
    return {n: p.grad.reshape(-1) for n, p in model.named_parameters()
            if p.grad is not None}


def zero_fill_grads(model):
    for p in model.parameters():
        p.grad = torch.zeros_like(p)


def pair_up(ga, gb):
    """Concatenate gradients over the INTERSECTION of parameter names.

    Returns (g_a, g_b, n_shared, dropped_a, dropped_b) so any structural
    mismatch between the two modes is reported rather than silently swallowed.
    """
    common = [n for n in ga if n in gb and ga[n].numel() == gb[n].numel()]
    drop_a = [n for n in ga if n not in common]
    drop_b = [n for n in gb if n not in common]
    va = torch.cat([ga[n] for n in common])
    vb = torch.cat([gb[n] for n in common])
    return va, vb, len(common), drop_a, drop_b


def rel_err(g_a, g_b):
    return ((g_a - g_b).norm() / g_b.norm().clamp(min=1e-12)).item()


def cos_sim(g_a, g_b):
    return F.cosine_similarity(g_a.reshape(1, -1), g_b.reshape(1, -1)).item()


def run(dataset="MUTAG", T=16, seed=42, windows=(2, 4, 6, 8, 12, 16)):
    set_seed(seed)
    ds = TUGraphDataset(name=dataset, tu_name=dataset, train_ratio=0.8, seed=seed)
    in_dim = max(ds.num_features, 1)
    sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=seed)
    ep = sampler.sample_episode(mode="train")
    s_b, s_l, q_b, q_l = batch_episode_data(ep, DEV)

    def make_cfg(window=None):
        kw = dict(hidden_dim=128, num_layers=2, T=T, tau=20.0,
                  ablation="full", use_metric_correction=True,
                  lr=1e-3, epochs=1)
        if window is None:
            # same construction, but the full graph is retained: exact gradient
            kw.update(use_dvm=True, dvm_mode="full_bptt_mc")
        else:
            kw.update(use_dvm=True, dvm_mode="truncated", dv_m_window=window)
        return SpikeRPGNNConfig(**kw)

    def loss_of(model, window=None):
        set_seed(seed)
        cfg = make_cfg(window)
        m = create_spikergnn(cfg, in_dim=in_dim).to(DEV)
        m.train()
        return m, cfg

    # ---- reference: full graph retained (exact gradient, same construction) ----
    ref_model, _ = loss_of(None)
    logits = ref_model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
    loss = F.cross_entropy(logits, q_l)
    loss.backward()
    g_ref_raw = named_grads(ref_model)
    g_ref_flat = torch.cat([v for v in g_ref_raw.values()])
    ref_state = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    print(f"  reference params with grad = {len(g_ref_raw)}  "
          f"grad norm = {g_ref_flat.norm().item():.6e}")

    out = {}
    meta = {}
    for M in windows:
        set_seed(seed)
        m = create_spikergnn(make_cfg(M), in_dim=in_dim).to(DEV)
        m.load_state_dict(ref_state, strict=False)   # identical weights
        m.train()
        logits = m(s_b.x, s_b.edge_index, s_b.batch,
                   q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        loss = F.cross_entropy(logits, q_l)
        loss.backward()
        g_t_raw = named_grads(m)
        g_a, g_b, n_shared, drop_a, drop_b = pair_up(g_t_raw, g_ref_raw)
        if M == windows[0]:
            meta["shared_params"] = n_shared
            meta["dropped_in_truncated"] = drop_b
            meta["dropped_in_reference"] = drop_a
            if drop_b:
                print(f"  [note] compared over {n_shared} shared params; "
                      f"{len(drop_b)} name(s) excluded: {drop_b[:3]}")
        out[M] = {"rel_err": rel_err(g_a, g_b),
                  "cos": cos_sim(g_a, g_b),
                  "gnorm_ratio": (g_a.norm() / g_b.norm().clamp(min=1e-12)).item()}
        print(f"  M={M:2d}  rel_err={out[M]['rel_err']:.6f}  "
              f"cos={out[M]['cos']:.4f}  |g_t|/|g_ref|={out[M]['gnorm_ratio']:.4f}")
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()

    out["_meta"] = meta
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "MUTAG"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    print(f"\n=== dataset={ds}, T=16, seed={seed} ===")
    res = run(ds, seed=seed)
    Ms = [k for k in res if isinstance(k, int)]
    closed = {M: 1 - 0.9 ** (16 - M) for M in Ms}
    print(f"\n{'M':>3} {'measured rel_err':>16} {'closed 1-rho^(T-M)':>18} {'delta':>10}")
    for M in Ms:
        print(f"{M:>3} {res[M]['rel_err']:>16.6f} {closed[M]:>18.6f} "
              f"{abs(res[M]['rel_err'] - closed[M]):>10.6f}")
    json.dump({"windows": {str(k): v for k, v in res.items() if isinstance(k, int)},
               "meta": res.get("_meta", {})},
              open(f"poc_real_gradient_error_{ds}.json", "w"), indent=2)
    print(f"\nsaved -> poc_real_gradient_error_{ds}.json")
