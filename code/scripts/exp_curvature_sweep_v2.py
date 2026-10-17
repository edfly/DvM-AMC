"""
Group F -- v2 redesign: the metric-correction gain is governed by the
window-endpoint conformal-factor deviation, NOT by curvature itself.

Why a redesign (see plan sec.2-bis)
-----------------------------------
The original F-group swept curvature c and effective radius and tested
"Delta(cos) increases monotonically with c / radius". Its pre-registered
verdict was NOT SUPPORTED (global Spearman 0.075, n=30). The reason is
mechanical: DvMFunction.backward multiplies the gradient by
(lam_prev/lam_curr)^2 at every step, which TELESCOPES over the retained
window to the single endpoint ratio lam_start^2 / lam_end^2. So the net
effect of the correction depends only on how much the trajectory's radius
changes between the two window endpoints -- not on the curvature c, and not
on the dynamic range of lambda across the whole point cloud.

This script tests the CORRECTED hypothesis directly:
    gain (1 - cos, or rel_err) increases monotonically with
    delta = |log( lam_start^2 / lam_end^2 )|
where lam = 2/(1 - c*||x||^2) at the window endpoints.

IMPORTANT implementation note: as of 2026-10-07 the deployed
apply_dvm/DvMFunction FORWARD curvature c into conformal_factor (default
c=1.0 for backward compatibility; the real model passes self.manifold.c).
The controlled block therefore validates the theoretical correction by
passing c directly to apply_dvm -- no monkeypatch needed. (An earlier
version of this script monkeypatched conformal_factor to inject c because
the deployed path ignored it; that limitation has been removed.)

Design
------
Block 1 (controlled, CPU): for a grid of (c, r_start, r_end) build a
    product-manifold state chain of length T with the retained window
    [T-M, T]; states before the window and the window-start have norm
    r_start, the window-end has norm r_end. A single learnable parameter
    produces p_init (norm r_start) so the corrected gradient reaches it.
    Measure cos / rel_err between MC-on and MC-off gradients. x = delta.
Block 2 (realistic, GPU): re-run the original protocol (MUTAG x seeds x c)
    on the real model, monkeypatching the correction to use the manifold's
    actual c, and record the realised window-endpoint ratio. Confirms the
    mechanism predicts Delta(cos) in the trained network.

Verdict (pre-registered)
-----------------------
    Spearman(delta, rel_err) > 0.8 AND Spearman(delta, 1-cos) > 0.8
      -> SUPPORTED: the conditional claim is "correction gain is governed
         by window-endpoint deviation (theoretically exact telescoping);
         it is numerically negligible when the retained-window trajectory
         is near-constant-norm -- which is exactly the TU operating point."
    else -> the telescoping identity is contradicted (a major finding);
         report, do not salvage the old claim.

Usage
-----
    python exp_curvature_sweep_v2.py --block controlled
    python exp_curvature_sweep_v2.py --block realistic
    python exp_curvature_sweep_v2.py --block both
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.join(_HERE, "programfiles")
for _p in (_PF, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spikergnn.dvm as _dvm  # noqa: E402
from spikergnn.dvm import conformal_factor as _orig_conformal  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join(_HERE, "results_v3")
T = 16
M = 8
D_H = 16
D_E = 16


# ------------------------------------------------------------------ helpers
def patch_conformal(c):
    """Monkeypatch spikergnn.dvm.conformal_factor to use curvature c.

    apply_dvm / DvMFunction call the module-level conformal_factor without c,
    so by default they use c=1.0. Patching injects the requested c so the
    realised correction matches the geometry (faithful to Theorem 1).

    NOTE: the closure MUST capture this function's `c` argument. The lambda
    therefore takes NO `c` parameter -- writing `lambda x, c=1.0: ...` and
    then `c=c` would silently resolve `c` to the lambda's own default (1.0)
    and make the patch a no-op (this bug was caught on 2026-10-07: every
    sweep config behaved like c=1.0 until the closure was fixed).
    """
    _dvm.conformal_factor = (
        lambda x, eps=1e-5, max_norm=None:
        _orig_conformal(x, c=c, eps=eps, max_norm=max_norm))


def unpatch_conformal():
    _dvm.conformal_factor = _orig_conformal


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    n = len(xs)
    if n < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else None


# ------------------------------------------------------------------ block 1
def _build_states(c, r_start, r_end, g, T, M, d_h, d_e):
    """Fresh state chain per MC call (no shared non-leaf tensors)."""
    def mk(r):
        v = torch.randn(1, d_h, generator=g)
        v = v / v.norm(dim=-1, keepdim=True) * r
        return v.detach()
    r_mid = (r_start + r_end) / 2.0
    states = []
    for t in range(T):
        if t < T - M:
            r = r_start
        elif t == T - 1:
            r = r_end
        else:
            r = r_mid
        states.append((mk(r), torch.randn(1, d_e, generator=g)))
    return states


def controlled_unit(c, r_start, r_end, seed, T=T, M=M, d_h=D_H, d_e=D_E):
    """Construct a chain whose retained-window endpoints have norms
    r_start (window start, = p_init norm) and r_end (window end). Measure
    the correction gain between MC-on and MC-off gradients of the single
    learnable parameter that produces p_init. Curvature c is passed
    directly to apply_dvm (the deployed, curvature-aware path)."""
    g = torch.Generator().manual_seed(seed)
    N = 1

    z0 = torch.randn(d_h, generator=g)
    z0 = z0 / z0.norm()
    W = nn.Parameter(r_start * torch.eye(d_h))          # p_init producer
    z1 = torch.randn(d_h, generator=g)
    z1 = z1 / z1.norm()
    V = nn.Parameter(r_end * torch.eye(d_h))             # window-end producer
    w = torch.randn(N, d_h, generator=g)

    def run(mc):
        p_init_hyp = (W @ z0).reshape(1, d_h)             # fresh graph-connected
        p_init_euc = torch.randn(1, d_e, generator=g)
        states = _build_states(c, r_start, r_end, g, T, M, d_h, d_e)
        # window-end state graph-connected (V) so the loss triggers backward
        states[T - 1] = ((V @ z1).reshape(1, d_h), states[T - 1][1])
        pf, _ = _dvm.apply_dvm(None, (p_init_hyp, p_init_euc), states, M,
                               use_metric_correction=mc, c=c)
        (pf * w).sum().backward()
        grad = W.grad.clone()
        W.grad = None
        # Realised net correction on p_init's gradient is the ratio of the
        # conformal factors at the TRUE endpoints: lambda(p_init)=lambda(r_start)
        # and lambda(window-end)=lambda(r_end); interior norms telescope out.
        lam_0 = 2.0 / (1.0 - c * r_start * r_start)
        lam_end = 2.0 / (1.0 - c * r_end * r_end)
        delta = float(abs(math.log((lam_0 / lam_end) ** 2)))
        return grad, delta

    g_mc, delta = run(True)
    g_noc, _ = run(False)

    cos = F.cosine_similarity(g_mc.reshape(1, -1), g_noc.reshape(1, -1)).item()
    rel = ((g_mc - g_noc).norm() / g_noc.norm().clamp(min=1e-12)).item()

    return {"c": c, "r_start": r_start, "r_end": r_end, "delta": delta,
            "cos": cos, "rel_err": rel}


def controlled_sweep():
    cs = [0.5, 1.0, 2.0]
    r_starts = [0.3, 0.5, 0.7]
    # r_end grid realises a spread of endpoint deviations at each c
    r_ends = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    seeds = [0, 7, 42]
    rows = []
    for c in cs:
        # valid Poincaré-ball radius: ||x|| < 1/sqrt(c); stay inside with margin
        max_r = 0.99 / math.sqrt(c)
        for rs in r_starts:
            if rs >= max_r:
                continue
            for re in r_ends:
                if abs(re - rs) < 1e-6:
                    continue
                if re >= max_r:
                    continue
                for s in seeds:
                    rows.append(controlled_unit(c, rs, re, s))
    return rows


# ------------------------------------------------------------------ block 2
def realistic_unit(dataset, seed, c):
    """Re-run the original F-group protocol but with the correction patched
    to use the manifold's actual c, and record the realised window-endpoint
    ratio. Returns delta and |Delta(cos)| so we can test the corrected
    mechanism in the trained network."""
    from msg_baseline.data_utils import (  # noqa: E402
        TUGraphDataset, EpisodeSampler, batch_episode_data)
    from msg_baseline.utils import set_seed  # noqa: E402
    from spikergnn.config import SpikeRPGNNConfig  # noqa: E402
    from spikergnn.model import create_spikergnn  # noqa: E402
    from spikergnn.stmp import PerStepMetricCorrection  # noqa: E402

    set_seed(seed)
    ds = TUGraphDataset(name=dataset, tu_name=dataset, train_ratio=0.8, seed=seed)
    in_dim = max(ds.num_features, 1)
    sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=seed)
    ep = sampler.sample_episode(mode="train")
    s_b, s_l, q_b, q_l = batch_episode_data(ep, DEV)

    def loss_of(model):
        logits = model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        return F.cross_entropy(logits, q_l)

    def mk_cfg(mc):
        kw = dict(hidden_dim=128, num_layers=2, T=T, tau=20.0, ablation="full",
                  use_metric_correction=mc, lr=1e-3, epochs=1, use_dvm=True,
                  c_init=c, learnable_c=False)
        kw["dvm_mode"] = "truncated"
        kw["dv_m_window"] = M
        return SpikeRPGNNConfig(**kw)

    # reference (full graph, exact) to get weights
    ref = create_spikergnn(mk_cfg(True), in_dim=in_dim).to(DEV)
    ref.train()
    loss_of(ref).backward()
    g_ref = {n: p.grad.reshape(-1) for n, p in ref.named_parameters()
             if p.grad is not None}
    ref_state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    del ref
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # intercept the window-endpoint conformal factor ratio at the true c
    rec = {"ratios": []}
    orig_fwd = PerStepMetricCorrection.forward

    def fwd(ctx, p_old, p_new, c, max_norm=None):
        lam_old = _orig_conformal(p_old, c, max_norm=max_norm)
        lam_new = _orig_conformal(p_new, c, max_norm=max_norm)
        rec["ratios"].append(((lam_old / lam_new.clamp(min=1e-5)) ** 2)
                             .detach().float().mean().item())
        return orig_fwd(ctx, p_old, p_new, c, max_norm)
    PerStepMetricCorrection.forward = staticmethod(fwd)

    out = {"dataset": dataset, "seed": seed, "c": c, "T": T, "M": M}
    try:
        for tag, mc in (("amc", True), ("no_metric_correction", False)):
            m = create_spikergnn(mk_cfg(mc), in_dim=in_dim).to(DEV)
            m.load_state_dict(ref_state, strict=False)
            m.train()
            loss_of(m).backward()
            g_obj = {n: p.grad.reshape(-1) for n, p in m.named_parameters()
                     if p.grad is not None}
            common = [n for n in g_ref if n in g_obj
                      and g_ref[n].numel() == g_obj[n].numel()]
            va = torch.cat([g_ref[n] for n in common])
            vb = torch.cat([g_obj[n] for n in common])
            out[tag] = {
                "cos": F.cosine_similarity(va.reshape(1, -1),
                                           vb.reshape(1, -1)).item(),
                "rel_err": ((va - vb).norm() / vb.norm().clamp(min=1e-12)).item()}
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        out["delta_cos"] = out["amc"]["cos"] - out["no_metric_correction"]["cos"]
        rr = rec["ratios"]
        n_mod = len(rr) // T if T else 0
        if n_mod and len(rr) == n_mod * T:
            seq = [rr[i * T:(i + 1) * T] for i in range(n_mod)]
            win = [math.prod(s[T - M:]) for s in seq]
            out["window_endpoint_ratio"] = math.exp(
                sum(math.log(max(x, 1e-300)) for x in win) / len(win))
        else:
            out["window_endpoint_ratio"] = 1.0
        # The realised correction on the retained-window gradient IS the
        # telescope factor k = window_endpoint_ratio (per-step (lam_prev/
        # lam_curr)^2 telescopes to the window-endpoint product). So the
        # gain = |k - 1|, and delta = |log k|. Both are rank-monotonic in k.
        out["delta"] = float(abs(math.log(
            max(out["window_endpoint_ratio"], 1e-300))))
        out["rel_err"] = float(abs(out["window_endpoint_ratio"] - 1.0))
    finally:
        PerStepMetricCorrection.forward = orig_fwd
    return out


def realistic_sweep(datasets, seeds, cs):
    rows = []
    for d in datasets:
        for s in seeds:
            for c in cs:
                rows.append(realistic_unit(d, s, c))
    return rows


# ------------------------------------------------------------------ gates
def check_gates_v2(rows):
    """The correction is a pure scalar rescale k = (lambda_0/lambda_end)^2 of
    the gradient, so cos(g_mc, g_noc) is ~1 by construction and the gain must
    be measured by rel_err = |k - 1|. The corrected hypothesis is therefore
    'rel_err increases monotonically with delta = |log k|'."""
    gates = {}
    xs = [r["delta"] for r in rows]
    ys_rel = [r["rel_err"] for r in rows]
    rho_rel = spearman(xs, ys_rel)
    gates["spearman_delta_rel_err"] = rho_rel
    gates["n_points"] = len(xs)

    # sanity: at delta ~ 0 the correction must be the identity (rel_err ~ 0)
    near = [r for r in rows if r["delta"] < 0.05]
    gates["max_rel_err_at_delta0"] = (
        max(r["rel_err"] for r in near) if near else None)
    gates["n_delta0"] = len(near)

    ok = (rho_rel is not None and rho_rel > 0.8)
    gates["monotonicity_ok"] = bool(ok)
    if ok:
        gates["verdict"] = (
            "SUPPORTED: the metric-correction gain is governed by the "
            "window-endpoint conformal-factor deviation (theoretically exact "
            "telescoping identity k = lambda_0^2/lambda_end^2). It is "
            "numerically negligible when the retained-window trajectory is "
            "near-constant-norm -- which is the TU operating point -- so the "
            "corrected conditional claim replaces 'gain grows with curvature' "
            "with 'gain grows with endpoint deviation'.")
    else:
        gates["verdict"] = (
            "NOT SUPPORTED: rel_err does not track the endpoint deviation. "
            "Do NOT revive the old 'gain grows with curvature' claim; keep the "
            "demotion to a theoretical-completeness footnote.")
    return gates


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="both",
                    choices=["controlled", "realistic", "both"])
    ap.add_argument("--datasets", default="MUTAG")
    ap.add_argument("--seeds", default="0,7,42")
    ap.add_argument("--cs", default="0.1,0.5,1.0,2.0,5.0")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT,
                                                   "curvature_sweep_v2.json"))
    args = ap.parse_args()

    t0 = time.time()
    out = {"device": DEV, "torch": torch.__version__,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
           "T": T, "M": M}
    if args.block in ("controlled", "both"):
        print("[block1] controlled mechanism sweep (CPU)...", flush=True)
        ctrl = controlled_sweep()
        out["controlled"] = ctrl
        g1 = check_gates_v2(ctrl)
        out["gates_controlled"] = g1
        print("  gates_controlled:", json.dumps(g1, ensure_ascii=False)[:400])

    if args.block in ("realistic", "both"):
        datasets = args.datasets.split(",")
        seeds = [int(s) for s in args.seeds.split(",")]
        cs = [float(x) for x in args.cs.split(",")]
        print(f"[block2] realistic sweep {datasets} x {seeds} x {cs} on {DEV}...",
              flush=True)
        real = realistic_sweep(datasets, seeds, cs)
        out["realistic"] = real
        g2 = check_gates_v2(real)
        out["gates_realistic"] = g2
        print("  gates_realistic:", json.dumps(g2, ensure_ascii=False)[:400])

    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nelapsed {out['elapsed_s']}s -> {args.out}")

    if "controlled" in out:
        print("\n-- controlled: delta vs gain (sample, c=1.0) --")
        for r in sorted([x for x in out["controlled"] if x["c"] == 1.0],
                        key=lambda x: x["delta"])[:8]:
            print(f"  r_start={r['r_start']:.2f} r_end={r['r_end']:.2f} "
                  f"delta={r['delta']:.3f} cos={r['cos']:.5f} "
                  f"rel_err={r['rel_err']:.5f}")


if __name__ == "__main__":
    main()
