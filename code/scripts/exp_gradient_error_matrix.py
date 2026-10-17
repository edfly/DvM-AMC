"""
Group A / Group B experiment matrix for the AMC paper revision.

Group A -- REAL truncation error of the analytic metric correction.
    Reference : dvm_mode="full_bptt_mc"    (full autograd graph, exact gradient)
    Object    : dvm_mode="truncated"       (window M, standard detach-based TBPTT)
    Both share identical weights and the same episode batch; the comparison is
    over the intersection of parameter gradients.

    Why this is not circular (cf. PaperSpine audit v2, P0-A): stmp.py L212-215
    implements truncation as a plain detach outside the window -- NO theoretical
    factor is multiplied onto the gradient. The measured error therefore depends
    on the data and on the seed, and the M == T row acts as a self-consistency
    sentinel that must come out at exactly zero.

Group B -- metric-correction ablation (rebuilds Table 6).
    Adds use_metric_correction=False under the same protocol.

Usage
-----
    python exp_gradient_error_matrix.py --group A            # recommended first
    python exp_gradient_error_matrix.py --group AB
    python exp_gradient_error_matrix.py --group A --datasets MUTAG DD --dry-run

Outputs results_v3/gradient_error_matrix.json with per-unit .done markers so a
crashed run resumes instead of restarting.
"""
import argparse
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join(_HERE, "results_v3")

DEFAULT_DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
DEFAULT_SEEDS = [0, 7, 17, 23, 42]
DEFAULT_TS = [8, 16, 24]


# ----------------------------------------------------------------- helpers
def named_grads(model):
    return {n: p.grad.reshape(-1) for n, p in model.named_parameters()
            if p.grad is not None}


def pair_up(ga, gb):
    """Concatenate over the intersection of parameter names.

    Any structural mismatch is reported rather than silently swallowed, because
    "which parameters stop receiving gradient" is itself a reportable finding.
    """
    common = [n for n in ga if n in gb and ga[n].numel() == gb[n].numel()]
    drop_ref = [n for n in gb if n not in common]
    drop_obj = [n for n in ga if n not in common]
    va = torch.cat([ga[n] for n in common])
    vb = torch.cat([gb[n] for n in common])
    return va, vb, len(common), drop_ref, drop_obj


def metrics(g_obj, g_ref):
    return {
        "rel_err": ((g_obj - g_ref).norm() / g_ref.norm().clamp(min=1e-12)).item(),
        "cos": F.cosine_similarity(g_obj.reshape(1, -1),
                                   g_ref.reshape(1, -1)).item(),
        "gnorm_ratio": (g_obj.norm() / g_ref.norm().clamp(min=1e-12)).item(),
    }


def make_cfg(T, window, mc=True):
    kw = dict(hidden_dim=128, num_layers=2, T=T, tau=20.0,
              ablation="full", use_metric_correction=mc,
              lr=1e-3, epochs=1, use_dvm=True)
    kw["dvm_mode"] = "full_bptt_mc" if window is None else "truncated"
    if window is not None:
        kw["dv_m_window"] = window
    return SpikeRPGNNConfig(**kw)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------- one unit
def run_unit(dataset, seed, T, windows, group):
    """One (dataset, seed, T) unit: reference + every window + optional ablation."""
    set_seed(seed)
    ds = TUGraphDataset(name=dataset, tu_name=dataset,
                        train_ratio=0.8, seed=seed)
    in_dim = max(ds.num_features, 1)
    sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=seed)
    ep = sampler.sample_episode(mode="train")
    s_b, s_l, q_b, q_l = batch_episode_data(ep, DEV)

    def loss_of(model):
        logits = model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        return F.cross_entropy(logits, q_l)

    rows, notes = [], {}

    # ---- reference: full graph (exact gradient) ----
    ref = create_spikergnn(make_cfg(T, None), in_dim=in_dim).to(DEV)
    ref.train()
    loss_of(ref).backward()
    g_ref_raw = named_grads(ref)
    ref_state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    g_ref_flat = torch.cat([v for v in g_ref_raw.values()])
    ref_norm = g_ref_flat.norm().item()
    del ref
    if DEV == "cuda":
        torch.cuda.empty_cache()

    variants = [("amc_truncated", True)]
    if group in ("AB", "B"):
        variants.append(("no_metric_correction", False))

    for tag, mc in variants:
        for M in windows:
            m = create_spikergnn(make_cfg(T, M, mc=mc), in_dim=in_dim).to(DEV)
            m.load_state_dict(ref_state, strict=False)
            m.train()
            loss_of(m).backward()
            g_raw = named_grads(m)
            g_obj, g_ref2, n_shared, drop_ref, drop_obj = pair_up(g_raw, g_ref_raw)
            row = {"dataset": dataset, "seed": seed, "T": T, "M": M,
                   "variant": tag, "n_shared_params": n_shared,
                   "ref_grad_norm": ref_norm}
            row.update(metrics(g_obj, g_ref2))
            rows.append(row)
            if (tag, M) == (variants[0][0], windows[0]):
                notes["dropped_in_reference"] = drop_ref
                notes["dropped_in_object"] = drop_obj
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

    return rows, notes


# ----------------------------------------------------------------- gates
def check_gates(rows):
    """Fail loudly rather than hand over a broken matrix to the paper."""
    gates = {}
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["dataset"], r["seed"], r["T"], r["variant"])].append(r)

    sentinel_rows, nos_rows = [], []
    for k, rs in by_key.items():
        _, _, T, variant = k
        for r in rs:
            # Only the corrected variant must collapse to the exact gradient at
            # M == T. The uncorrected variant differs from the reference even at
            # M == T -- that gap IS the measured contribution of the correction.
            is_sentinel = (r["M"] == T and variant == "amc_truncated")
            (sentinel_rows if is_sentinel else nos_rows).append(r)

    gates["sentinel_max_rel_err"] = (
        max(r["rel_err"] for r in sentinel_rows) if sentinel_rows else None)
    gates["sentinel_ok"] = gates["sentinel_max_rel_err"] is not None and \
        gates["sentinel_max_rel_err"] < 1e-4

    # rows must differ: identical values across units => degenerate construction
    sigs = {round(r["rel_err"], 6) for r in nos_rows}
    gates["n_distinct_values"] = len(sigs)
    gates["n_non_sentinel_rows"] = len(nos_rows)
    gates["distinct_ok"] = len(sigs) > len(nos_rows) * 0.5

    # monotone non-increasing in M within each unit
    viol = []
    for k, rs in by_key.items():
        rs = sorted(rs, key=lambda r: r["M"])
        for a, b in zip(rs, rs[1:]):
            if b["rel_err"] > a["rel_err"] + 0.02:
                viol.append((k, a["M"], b["M"], a["rel_err"], b["rel_err"]))
    gates["monotone_violations"] = viol[:10]
    gates["monotone_ok"] = len(viol) == 0
    gates["all_ok"] = all([gates["sentinel_ok"], gates["distinct_ok"],
                           gates["monotone_ok"]])
    return gates


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A", choices=["A", "B", "AB"])
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--Ts", default=",".join(map(str, DEFAULT_TS)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT,
                                                  "gradient_error_matrix.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    Ts = [int(t) for t in args.Ts.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_dir = os.path.join(OUT_ROOT, "_done")
    os.makedirs(done_dir, exist_ok=True)

    def windows_for(T):
        ws = [w for w in (2, 4, 6, 8, 12) if w < T] + [T]   # last = sentinel
        return ws

    units = [(d, s, T) for d in datasets for s in seeds for T in Ts]
    if args.dry_run:
        print(f"group={args.group}  units={len(units)}  device={DEV}")
        for d, s, T in units[:6]:
            print(f"  {d:10s} seed={s:<3} T={T:<3} windows={windows_for(T)}")
        print(f"  ... {len(units)} total")
        return

    t0 = time.time()
    all_rows, all_notes = [], {}
    for i, (d, s, T) in enumerate(units, 1):
        marker = os.path.join(done_dir, f"{d}__s{s}__T{T}__{args.group}.done")
        if os.path.exists(marker):
            rows, notes = json.load(open(marker, encoding="utf-8"))
            print(f"[{i}/{len(units)}] SKIP {d}/s{s}/T{T}")
        else:
            print(f"[{i}/{len(units)}] RUN  {d}/s{s}/T{T}", flush=True)
            rows, notes = run_unit(d, s, T, windows_for(T), args.group)
            json.dump([rows, notes], open(marker, "w", encoding="utf-8"))
        all_rows.extend(rows)
        all_notes[f"{d}|{s}|{T}"] = notes
        # incremental save so a crash never loses work
        json.dump({"partial": True, "rows": all_rows},
                  open(args.out, "w", encoding="utf-8"), indent=2)

    gates = check_gates(all_rows)
    out = {
        "config": {"group": args.group, "datasets": datasets, "seeds": seeds,
                   "Ts": Ts, "hidden_dim": 128, "num_layers": 2,
                   "reference": "dvm_mode=full_bptt_mc",
                   "object": "dvm_mode=truncated"},
        "device": DEV,
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "notes_params_without_gradient": all_notes,
        "gates": gates,
        "rows": all_rows,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print(f"\nelapsed {out['elapsed_s']}s -> {args.out}")
    print("GATES:")
    for k, v in gates.items():
        print(f"  {k:24s} {v}")
    if not gates["all_ok"]:
        print("\n[FAIL] at least one gate failed -- do NOT feed these numbers "
              "into the paper before understanding why.")


if __name__ == "__main__":
    main()
