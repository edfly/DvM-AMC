"""
Group F -- curvature / radius sweep: delimiting where the metric correction matters.

Motivation (PaperSpine audit v2, section 2 of the experiment plan)
------------------------------------------------------------------
Group B measured that removing the metric correction changes the gradient cosine
by only ~2e-4 at the working point of the TU benchmarks, while the paper claims a
46-point gap. The cause is geometric, not a bug: the hyperbolic embeddings sit at
||p|| ~ 0.34 inside a Poincare ball of radius 1, so lambda = 2/(1 - c||x||^2)
varies over a dynamic range of only ~1.02 and the correction is numerically
indistinguishable from the identity.

lambda^2 grows without bound as sqrt(c)*||x|| -> 1, so the correction should
matter increasingly as the effective radius r = sqrt(c)*||x|| grows. This script
tests exactly that. If the measured Delta(cos) does NOT increase monotonically
with r, the honest conclusion is that the metric correction contributes little at
any curvature, and the claim must be dropped -- the gate below enforces this
rather than letting us cherry-pick a favourable sub-range.

Design
------
Axis 1 (curvature, measured): c in {0.1, 0.5, 1.0, 2.0, 5.0}. projx clips the
    ball to radius 1/sqrt(c), so c is the control knob that moves the working
    point relative to the boundary.
Axis 2 (radius, analytic): lambda^2 dynamic range as a function of r, independent
    of any model. This is the mechanism table, not a result.

Protocol per (dataset, seed, c): identical weights, identical episode batch.
    reference  : dvm_mode="full_bptt_mc"        (exact gradient)
    amc        : dvm_mode="truncated", M=8, mc=True
    no_correction: dvm_mode="truncated", M=8, mc=False
Report cos / rel_err for both variants and Delta = amc - no_correction.

Usage
-----
    python exp_curvature_sweep.py --dry-run
    python exp_curvature_sweep.py                       # full run
    python exp_curvature_sweep.py --datasets MUTAG --seeds 42
"""
import argparse
import math
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.join(_HERE, "programfiles")
for _p in (_PF, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler, batch_episode_data)
from msg_baseline.utils import set_seed  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

# stmp.py imports `from spikergnn.dvm import PerStepMetricCorrection`; patching
# the top-level `dvm` module would silently target a different class object.
from spikergnn.dvm import PerStepMetricCorrection, conformal_factor  # noqa: E402
import spikergnn.stmp as _stmp  # noqa: E402
assert _stmp.PerStepMetricCorrection is PerStepMetricCorrection, \
    "patching the wrong PerStepMetricCorrection class object"

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join(_HERE, "results_v3")

DEFAULT_DATASETS = ["MUTAG", "DD"]
DEFAULT_SEEDS = [0, 7, 42]
DEFAULT_CS = [0.1, 0.5, 1.0, 2.0, 5.0]
T = 16
M = 8


# ------------------------------------------------------------------ helpers
def named_grads(model):
    return {n: p.grad.reshape(-1) for n, p in model.named_parameters()
            if p.grad is not None}


def pair_up(ga, gb):
    common = [n for n in ga if n in gb and ga[n].numel() == gb[n].numel()]
    va = torch.cat([ga[n] for n in common])
    vb = torch.cat([gb[n] for n in common])
    return va, vb, len(common)


def metrics(g_obj, g_ref):
    return {
        "rel_err": ((g_obj - g_ref).norm() / g_ref.norm().clamp(min=1e-12)).item(),
        "cos": F.cosine_similarity(g_obj.reshape(1, -1),
                                   g_ref.reshape(1, -1)).item(),
    }


def make_cfg(c, window=None, mc=True, T=T):
    kw = dict(hidden_dim=128, num_layers=2, T=T, tau=20.0,
              ablation="full", use_metric_correction=mc,
              lr=1e-3, epochs=1, use_dvm=True,
              c_init=c, learnable_c=False)
    kw["dvm_mode"] = "full_bptt_mc" if window is None else "truncated"
    if window is not None:
        kw["dv_m_window"] = window
    return SpikeRPGNNConfig(**kw)


def instrument_ratio(model, T):
    """Record lambda_old / lambda_new at every correction call.

    PerStepMetricCorrection multiplies the incoming gradient by
    (lambda_old/lambda_new)^2, so along a chain the factors telescope and the
    NET effect of the correction over a window is the endpoint ratio
    lambda_start^2 / lambda_end^2 -- NOT the dynamic range across the point
    cloud. Measuring this ratio directly is what distinguishes "the correction
    is a no-op" from "the correction is not wired up".
    """
    rec = {"ratios": []}
    orig_fwd = PerStepMetricCorrection.forward

    def fwd(ctx, p_old, p_new, c):
        lam_old = conformal_factor(p_old, c)
        lam_new = conformal_factor(p_new, c)
        rec["ratios"].append(((lam_old / lam_new.clamp(min=1e-5)) ** 2)
                             .detach().float().mean().item())
        return orig_fwd(ctx, p_old, p_new, c)

    PerStepMetricCorrection.forward = staticmethod(fwd)
    return rec, orig_fwd


def instrument(model):
    """Wrap hyperbolic projx to record state norms and the live curvature.

    We deliberately do not touch the model's math: the wrapper only observes the
    tensor passing through, so the recorded norms are the ones the network
    actually used.
    """
    rec = {"norms": [], "c": None, "n_wrapped": 0}
    for mod in model.modules():
        hyp = getattr(getattr(mod, "manifold", None), "hyp", None)
        if hyp is None or not hasattr(hyp, "projx"):
            continue
        orig = hyp.projx
        if getattr(orig, "_is_recorder", False):
            continue

        def wrapped(x, _orig=orig, _rec=rec):
            out = _orig(x)
            # projx is reached with tensors of different rank (per-node states
            # and per-step unsqueezed features), so flatten before recording.
            _rec["norms"].append(out.detach().float().norm(dim=-1).reshape(-1))
            return out
        wrapped._is_recorder = True
        hyp.projx = wrapped              # instance attribute shadows the method
        rec["n_wrapped"] += 1
    return rec


def read_curve(model):
    for mod in model.modules():
        man = getattr(mod, "manifold", None)
        if man is not None and hasattr(man, "c"):
            c = man.c
            return float(c.item()) if torch.is_tensor(c) else float(c)
    return None


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ one unit
def run_unit(dataset, seed, c):
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

    out = {"dataset": dataset, "seed": seed, "c": c, "T": T, "M": M}

    # ---- reference: full graph, exact gradient ----
    ref = create_spikergnn(make_cfg(c, None), in_dim=in_dim).to(DEV)
    ref.train()
    rec_ref = instrument(ref)
    rec_ratio, orig_fwd = instrument_ratio(ref, T)
    try:
        loss_of(ref).backward()
    finally:
        PerStepMetricCorrection.forward = orig_fwd   # never leave it patched
    g_ref = named_grads(ref)
    ref_state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    out["c_effective"] = read_curve(ref)
    del ref
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # ---- geometry actually realised (recorded during the reference pass) ----
    if rec_ref["norms"]:
        all_n = torch.cat(rec_ref["norms"])
        cv = out["c_effective"] or c
        r_eff = (cv ** 0.5) * all_n
        lam = 2.0 / (1.0 - (cv * all_n ** 2).clamp(max=0.999999))
        lam2 = lam ** 2
        out["geometry"] = {
            "n_proj_calls": len(rec_ref["norms"]),
            "n_points": int(all_n.numel()),
            "hyp_norm_mean": all_n.mean().item(),
            "hyp_norm_max": all_n.max().item(),
            "r_eff_mean": r_eff.mean().item(),
            "r_eff_max": r_eff.max().item(),
            "lambda_mean": lam.mean().item(),
            "lambda_min": lam.min().item(),
            "lambda_max": lam.max().item(),
            "lambda2_dynamic_range": (lam2.max() / lam2.min().clamp(min=1e-12)).item(),
            "numerically_stable": bool(torch.isfinite(lam).all()),
        }
    else:
        out["geometry"] = None

    # ---- the two truncated variants, identical weights ----
    for tag, mc in (("amc", True), ("no_metric_correction", False)):
        m = create_spikergnn(make_cfg(c, M, mc=mc), in_dim=in_dim).to(DEV)
        m.load_state_dict(ref_state, strict=False)
        m.train()
        loss_of(m).backward()
        g_obj, g_ref2, n_shared = pair_up(named_grads(m), g_ref)
        out[tag] = metrics(g_obj, g_ref2)
        out[tag]["n_shared_params"] = n_shared
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()

    out["delta_cos"] = out["amc"]["cos"] - out["no_metric_correction"]["cos"]
    out["delta_rel_err"] = (out["no_metric_correction"]["rel_err"]
                            - out["amc"]["rel_err"])

    # ---- telescoped endpoint ratio: the quantity that actually governs the
    # net effect of the correction over the retained window ----
    rr = rec_ratio["ratios"]
    n_mod = len(rr) // T if T else 0
    if n_mod and len(rr) == n_mod * T:
        seq = [rr[i * T:(i + 1) * T] for i in range(n_mod)]
        full = [math.prod(s) for s in seq]
        win = [math.prod(s[T - M:]) for s in seq]
        out["endpoint_ratio"] = {
            "n_correction_calls": len(rr),
            "n_stmp_modules": n_mod,
            "full_chain_per_module": full,
            "window_only_per_module": win,
            "full_chain_geomean": math.exp(sum(math.log(max(x, 1e-300))
                                               for x in full) / len(full)),
            "window_geomean": math.exp(sum(math.log(max(x, 1e-300))
                                           for x in win) / len(win)),
        }
    else:
        out["endpoint_ratio"] = {"n_correction_calls": len(rr),
                                 "note": "unexpected call count; not reshapeable"}
    return out


# ------------------------------------------------------- analytic mechanism
def analytic_axis(r_grid=(0.2, 0.34, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """lambda^2 dynamic range vs effective radius. Model-free, CPU only."""
    rows = []
    for r in r_grid:
        lo = 2.0 / (1.0 - (r * 0.9) ** 2)     # +/-10% spread around r
        hi = 2.0 / (1.0 - r ** 2)
        rows.append({"r_eff": r, "lambda_lo": lo, "lambda_hi": hi,
                     "lambda2_dynamic_range": (hi / lo) ** 2})
    return rows


# ------------------------------------------------------------------ gates
def spearman(xs, ys):
    """Rank correlation with average ranks for ties."""
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


def check_gates(rows):
    """Decide, from the data alone, whether the conditional claim survives."""
    gates = {}
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[(r["dataset"], r["seed"])].append(r)

    # (a) global rank correlation of Delta(cos) against effective radius
    xs = [r["geometry"]["r_eff_mean"] for r in rows if r.get("geometry")]
    ys = [r["delta_cos"] for r in rows if r.get("geometry")]
    rho_global = spearman(xs, ys)
    gates["spearman_rho_global"] = rho_global
    gates["n_points"] = len(xs)

    # (b) per-unit correlation: does the trend hold for each (dataset, seed)?
    per_unit = {}
    for k, rs in by_unit.items():
        rs = [r for r in rs if r.get("geometry")]
        if len(rs) >= 3:
            per_unit[f"{k[0]}|s{k[1]}"] = spearman(
                [r["geometry"]["r_eff_mean"] for r in rs],
                [r["delta_cos"] for r in rs])
    gates["spearman_rho_per_unit"] = per_unit
    worst = min(per_unit.values()) if per_unit else None
    gates["spearman_rho_worst_unit"] = worst

    # (c) sanity: the correction must be a no-op at the smallest radius
    near = [r for r in rows if r.get("geometry")
            and r["geometry"]["r_eff_mean"] < 0.4]
    gates["max_abs_delta_cos_at_small_radius"] = (
        max(abs(r["delta_cos"]) for r in near) if near else None)

    # (d) numerical stability across the sweep
    gates["all_finite"] = all(
        r["geometry"]["numerically_stable"] for r in rows if r.get("geometry"))

    rho_ok = rho_global is not None and rho_global > 0.8
    unit_ok = worst is None or worst > 0.8
    gates["monotonicity_ok"] = bool(rho_ok and unit_ok)
    gates["all_ok"] = bool(gates["monotonicity_ok"] and gates["all_finite"])

    # What the paper may claim, decided by the data -- not by us.
    if gates["monotonicity_ok"]:
        gates["verdict"] = ("SUPPORTED: Delta(cos) increases monotonically with "
                            "effective radius; the conditional claim "
                            "'correction gain grows with curvature' may be made.")
    else:
        gates["verdict"] = ("NOT SUPPORTED: no monotone increase. Per the plan, "
                            "the claim 'metric correction contributes 46 points' "
                            "MUST be deleted and the correction demoted to a "
                            "theoretical-completeness footnote. Do not "
                            "cherry-pick a favourable sub-range.")
    return gates


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--cs", default=",".join(map(str, DEFAULT_CS)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT,
                                                  "curvature_sweep.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    cs = [float(x) for x in args.cs.split(",")]
    units = [(d, s, c) for d in datasets for s in seeds for c in cs]

    if args.dry_run:
        print(f"device={DEV}  units={len(units)}  T={T} M={M}")
        for u in units[:8]:
            print("  ", u)
        print("  analytic mechanism axis (model-free):")
        for row in analytic_axis():
            print(f"    r_eff={row['r_eff']:<5} lambda^2 range="
                  f"{row['lambda2_dynamic_range']:.3f}")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_dir = os.path.join(OUT_ROOT, "_done_F")
    os.makedirs(done_dir, exist_ok=True)

    t0 = time.time()
    rows = []
    for i, (d, s, c) in enumerate(units, 1):
        marker = os.path.join(done_dir, f"{d}__s{s}__c{c}.done")
        if os.path.exists(marker):
            rows.append(json.load(open(marker, encoding="utf-8")))
            print(f"[{i}/{len(units)}] SKIP {d}/s{s}/c{c}")
            continue
        print(f"[{i}/{len(units)}] RUN  {d}/s{s}/c{c}", flush=True)
        try:
            row = run_unit(d, s, c)
        except RuntimeError as e:
            print(f"    !! failed: {str(e)[:160]}")
            row = {"dataset": d, "seed": s, "c": c, "T": T, "M": M,
                   "error": str(e)[:300]}
        json.dump(row, open(marker, "w", encoding="utf-8"))
        rows.append(row)
        json.dump({"partial": True, "rows": rows},
                  open(args.out, "w", encoding="utf-8"), indent=2)

    good = [r for r in rows if "error" not in r]
    gates = check_gates(good)
    out = {
        "config": {"datasets": datasets, "seeds": seeds, "curvatures": cs,
                   "T": T, "M": M, "hidden_dim": 128, "num_layers": 2,
                   "reference": "dvm_mode=full_bptt_mc",
                   "variants": ["amc", "no_metric_correction"]},
        "device": DEV,
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "elapsed_s": round(time.time() - t0, 1),
        "analytic_mechanism_axis": analytic_axis(),
        "gates": gates,
        "rows": rows,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print(f"\nelapsed {out['elapsed_s']}s -> {args.out}")
    print("\n-- measured Delta(cos) vs effective radius --")
    print(f"{'ds':>6} {'seed':>5} {'c':>5} {'r_eff':>7} {'lam2 rng':>9} "
          f"{'AMC cos':>9} {'NoCorr cos':>11} {'Delta':>9}")
    for r in good:
        g = r.get("geometry") or {}
        print(f"{r['dataset']:>6} {r['seed']:>5} {r['c']:>5} "
              f"{g.get('r_eff_mean', float('nan')):>7.3f} "
              f"{g.get('lambda2_dynamic_range', float('nan')):>9.3f} "
              f"{r['amc']['cos']:>9.5f} {r['no_metric_correction']['cos']:>11.5f} "
              f"{r['delta_cos']:>+9.5f}")

    print("\nGATES:")
    for k, v in gates.items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
