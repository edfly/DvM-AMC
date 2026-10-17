"""
run_T_sweep.py — supplementary T=32/64 experiment sweep for the AMC paper.

Purpose
-------
Address the Neurocomputing R2 objection: at the reported T=8 / M=8 the
truncation window equals the sequence length (start_idx = max(0, T-M) = 0),
so the O(M) memory claim is vacuous. This sweep demonstrates the method at
longer horizons where truncation is REAL:

  For each dataset in {MUTAG, PROTEINS, NCI1, DD}:
    For each T in TS (default 32, 64):
      regime "truncated"  -> dvm_mode="truncated",      dv_m_window=M_FIXED (8)
      regime "full"       -> dvm_mode="full_bptt_mc",   dv_m_window=T   (O(T) oracle)

For every (dataset, T, regime, run) we record:
  - final accuracy  (mean over test episodes, identical protocol to the OM arm)
  - peak GPU memory (torch.cuda.max_memory_allocated)  -> demonstrates O(M) vs O(T)

The seed formula (seed_base + run_id*100 + shot*10) is IDENTICAL to
spikergnn.run_experiment, so the new T=32/64 runs are a controlled comparison
against the existing T=8 anchor (71.7%, rerun_outputs/MUTAG_OM/...). At T=8
both regimes coincide (start_idx=0), so 71.7% is the shared anchor.

Resume-safe: results are written per-run to JSON; already-completed
(dataset, T, regime, run) cells are skipped on restart.

Usage
-----
  PYTHONPATH="$PWD/programfiles" CUDA_VISIBLE_DEVICES=0 \
    C:/Python314/python.exe run_T_sweep.py \
      --datasets MUTAG PROTEINS NCI1 DD --Ts 32 64 --n-runs 9 --epochs 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.train import run_single_experiment

# ── NCI1 dataset-config injection (P0-1 fix) ──────────────────────────────
# run_single_experiment looks NCI1 up in msg_baseline.config.DATASET_CONFIGS,
# but that dict (as shipped) only lists MUTAG/PROTEINS/DD/ENZYMES/REDDIT-M5K.
# The GNN-baseline path that produced the main-matrix NCI1 numbers uses a
# separate loader, so NCI1 was never registered here. Inject it at runtime so
# this SNN sweep can cover NCI1 too — without mutating the shared config module.
from msg_baseline.config import DATASET_CONFIGS as _DSC
if "NCI1" not in _DSC:
    _DSC["NCI1"] = {
        "n_way": 2,
        "shots": [1, 5],
        "n_graphs": 4110,
        "n_classes": 2,
        "tu_name": "NCI1",
    }

M_FIXED = 8  # truncation window for the "truncated" regime (matches paper's M=8)


def build_config(T, regime, epochs):
    """Build the SpikeRPGNNConfig for one (T, regime) cell."""
    if regime == "truncated":
        dv_m_window = M_FIXED
        dvm_mode = "truncated"
    elif regime == "full":
        dv_m_window = T
        dvm_mode = "full_bptt_mc"
    else:
        raise ValueError(f"unknown regime {regime}")
    return SpikeRPGNNConfig(
        ablation="full",
        use_dvm=True,
        use_metric_correction=True,
        T=T,
        dvm_mode=dvm_mode,
        dv_m_window=dv_m_window,
        epochs=epochs,
    )


def run_one(dataset, T, regime, run_id, epochs, device):
    """Run one (dataset, T, regime, run). Returns a result dict or an error dict.

    On any exception (e.g. CUDA OOM for the O(T) full regime at long T on big
    graphs) we return an error marker instead of raising, so a single failed
    cell cannot abort the whole sweep. An OOM for the full regime is itself a
    legitimate finding (it demonstrates the O(T) memory cost that O(M) avoids).
    """
    cfg = build_config(T, regime, epochs)
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        res = run_single_experiment(
            dataset_name=dataset,
            shot=1,
            run_id=run_id,
            config=cfg,
            device=device,
        )
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        return {
            "acc": float(res["final_acc"]),
            "best_acc": float(res["best_acc"]),
            "peak_mem_mb": float(peak_mb),
            "T": T,
            "regime": regime,
            "dv_m_window": cfg.dv_m_window,
            "dvm_mode": cfg.dvm_mode,
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001 — capture and continue
        # Record the peak reached BEFORE the crash. For the full-BPTT regime a
        # NaN-induced crash is itself a legitimate finding (numerical instability
        # of O(T) backprop at long T); the pre-crash peak still demonstrates the
        # O(T) memory footprint we contrast against O(M).
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        return {
            "acc": None,
            "best_acc": None,
            "peak_mem_mb": float(peak_mb),
            "T": T,
            "regime": regime,
            "dv_m_window": cfg.dv_m_window,
            "dvm_mode": cfg.dvm_mode,
            "ok": False,
            "crashed_peak": True,
            "error": repr(e)[:300],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["MUTAG", "PROTEINS", "NCI1", "DD"])
    ap.add_argument("--Ts", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--regimes", nargs="+",
                    default=["truncated", "full"])
    ap.add_argument("--n-runs", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="T_sweep_results.json")
    ap.add_argument("--only-dataset", default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cuda_available={torch.cuda.is_available()}", flush=True)

    out_path = os.path.join(_PROJECT_ROOT, args.out)
    # Load existing (resume-safe)
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"[resume] loaded {out_path}", flush=True)
    else:
        results = {
            "meta": {
                "started": datetime.now().isoformat(),
                "device": device,
                "epochs": args.epochs,
                "n_runs": args.n_runs,
                "shot": 1,
                "M_fixed": M_FIXED,
                "seed_formula": "seed_base + run_id*100 + shot*10 (identical to run_experiment)",
                "regimes": args.regimes,
                "Ts": args.Ts,
            },
            "results": {},
        }

    datasets = [d for d in args.datasets if (args.only_dataset is None or d == args.only_dataset)]
    total_cells = len(datasets) * len(args.Ts) * len(args.regimes) * args.n_runs
    done = 0

    for ds in datasets:
        results["results"].setdefault(ds, {})
        for T in args.Ts:
            key_T = f"T{T}"
            results["results"][ds].setdefault(key_T, {})
            for regime in args.regimes:
                cell = results["results"][ds][key_T].setdefault(regime, {
                    "accs": [], "best_accs": [], "peak_mem_mb": [],
                    "T": T, "regime": regime,
                })
                # resume: count how many runs already done
                n_done = len(cell["accs"])
                if n_done >= args.n_runs:
                    print(f"[skip] {ds} {key_T} {regime} ({n_done}/{args.n_runs})", flush=True)
                    done += args.n_runs
                    continue
                print(f"\n=== {ds} {key_T} {regime} (M={M_FIXED if regime=='truncated' else T}) ===",
                      flush=True)
                for run_id in range(n_done, args.n_runs):
                    t0 = time.time()
                    r = run_one(ds, T, regime, run_id, args.epochs, device)
                    if not r["ok"]:
                        # record the failure; if a pre-crash peak was captured
                        # (NaN crash on the O(T) full regime) keep it as evidence.
                        if r.get("peak_mem_mb") is not None:
                            cell["peak_mem_mb"].append(r["peak_mem_mb"])
                        cell.setdefault("errors", []).append(r)
                        dt = time.time() - t0
                        pk = r.get("peak_mem_mb")
                        pk_s = f"  crash_peak={pk:.1f}MB" if pk is not None else ""
                        print(f"  run{run_id}: ERROR {r['error']}{pk_s}  {dt:.0f}s  "
                              f"({done}/{total_cells})", flush=True)
                        continue
                    cell["accs"].append(r["acc"])
                    cell["best_accs"].append(r["best_acc"])
                    cell["peak_mem_mb"].append(r["peak_mem_mb"])
                    done += 1
                    dt = time.time() - t0
                    print(f"  run{run_id}: acc={r['acc']*100:.2f}%  "
                          f"peak={r['peak_mem_mb']:.1f}MB  {dt:.0f}s  "
                          f"({done}/{total_cells})", flush=True)
                    # incremental save
                    _finalize_cell(cell)
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2, default=str)

    # finalize all cells
    for ds in results["results"]:
        for key_T in results["results"][ds]:
            for regime in results["results"][ds][key_T]:
                _finalize_cell(results["results"][ds][key_T][regime])
    results["meta"]["finished"] = datetime.now().isoformat()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nALL DONE -> {out_path}", flush=True)


def _safe_fmean(xs):
    return sum(xs) / len(xs)


def _safe_pstdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _finalize_cell(cell):
    # NOTE: do NOT use statistics.fmean — it is unavailable in the Python
    # build on this machine. Use plain sum/len so a single finalize can never
    # block persisting already-measured raw runs.
    if cell["accs"]:
        cell["mean_acc"] = _safe_fmean(cell["accs"])
        cell["std_acc"] = _safe_pstdev(cell["accs"])
    if cell["peak_mem_mb"]:
        cell["mean_peak_mb"] = _safe_fmean(cell["peak_mem_mb"])
        cell["max_peak_mb"] = max(cell["peak_mem_mb"])


if __name__ == "__main__":
    main()
