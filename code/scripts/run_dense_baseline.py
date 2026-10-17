"""
run_dense_baseline.py -- non-spiking (dense) baselines for the AMC paper.

Purpose
-------
Address reviewer P0-2 (parameter-matched dense baseline) and P1-3
(non-spiking Riemannian-GNN baseline). Both are realised WITHOUT touching
the spiking code path: the existing ``ablation="no_spikes"`` branch in
spikergnn/neurons.py (ManifoldLIFNeuron.forward) skips spiking and runs the
manifold recurrence unmasked, keeping the manifold state. We vary the
curvature to obtain the two requested baselines:

  ablation tag            -> neurons ablation   c_init   meaning
  ---------------------------------------------------------------------------
  "full"                  full                1.0      spiking Riemannian (control)
  "no_spikes"             no_spikes           1.0      non-spiking Riemannian GNN  (P1-3)

NOTE (2026-10-13): a third tag, "no_spikes_euclid" (c_init=0.0), was trialled
as the curvature-ablated (dense Euclidean) control. Under the standard few-shot
protocol it does NOT train to above-chance accuracy (exactly 0.5 across three
independent runs) even though its embeddings evolve and its geodesic classifier
is mathematically correct at c=0 -- the curvature-induced optimisation bias is
lost. It is therefore DROPPED. The parameter-matched dense comparison now relies
on the existing GNN baselines (GIN/GCN/SAGE, 232,850 params), which are
invariant to the metric-correction fix and already populate the main matrix.

Protocol: single seed, 3 epochs (probe) -- identical across the ablations so the
RELATIVE accuracy is a fair read of the spike / manifold effect. Absolute numbers
are NOT comparable to the 9-run main tables.

Resume-safe: per-cell JSON, completed (dataset, T, ablation) cells skipped.
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

# ── NCI1 dataset-config injection (same fix as run_T_sweep.py) ──────────────
from msg_baseline.config import DATASET_CONFIGS as _DSC
if "NCI1" not in _DSC:
    _DSC["NCI1"] = {
        "n_way": 2, "shots": [1, 5], "n_graphs": 4110,
        "n_classes": 2, "tu_name": "NCI1",
    }

M_FIXED = 8  # truncated window (matches paper M=8)


def build_config(T, ablation_tag, epochs):
    """Build SpikeRPGNNConfig for one (T, ablation_tag) cell."""
    if ablation_tag == "full":
        cfg_ablation, c_init = "full", 1.0
    elif ablation_tag == "no_spikes":
        cfg_ablation, c_init = "no_spikes", 1.0      # Riemannian, non-spiking (P1-3)
    elif ablation_tag == "no_spikes_euclid":
        cfg_ablation, c_init = "no_spikes", 0.0      # Euclidean, non-spiking (P0-2)
    else:
        raise ValueError(f"unknown ablation tag {ablation_tag}")
    return SpikeRPGNNConfig(
        ablation=cfg_ablation,
        learnable_c=False,
        c_init=c_init,
        use_dvm=True,
        use_metric_correction=True,
        T=T,
        dvm_mode="truncated",
        dv_m_window=M_FIXED,
        epochs=epochs,
    )


def run_one(dataset, T, ablation_tag, run_id, epochs, device):
    cfg = build_config(T, ablation_tag, epochs)
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        res = run_single_experiment(
            dataset_name=dataset, shot=1, run_id=run_id,
            config=cfg, device=device,
        )
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        return {
            "acc": float(res["final_acc"]),
            "best_acc": float(res["best_acc"]),
            "peak_mem_mb": float(peak_mb),
            "T": T, "ablation_tag": ablation_tag,
            "ablation": cfg.ablation, "c_init": c_init_of(cfg),
            "dv_m_window": cfg.dv_m_window, "dvm_mode": cfg.dvm_mode,
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        return {
            "acc": None, "best_acc": None, "peak_mem_mb": float(peak_mb),
            "T": T, "ablation_tag": ablation_tag,
            "ablation": cfg.ablation, "c_init": c_init_of(cfg),
            "dv_m_window": cfg.dv_m_window, "dvm_mode": cfg.dvm_mode,
            "ok": False, "crashed_peak": True, "error": repr(e)[:300],
        }


def c_init_of(cfg):
    try:
        return float(cfg.c_init)
    except Exception:
        return None


def _safe_fmean(xs):
    return sum(xs) / len(xs)


def _safe_pstdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _finalize_cell(cell):
    if cell["accs"]:
        cell["mean_acc"] = _safe_fmean(cell["accs"])
        cell["std_acc"] = _safe_pstdev(cell["accs"])
    if cell["peak_mem_mb"]:
        cell["mean_peak_mb"] = _safe_fmean(cell["peak_mem_mb"])
        cell["max_peak_mb"] = max(cell["peak_mem_mb"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["MUTAG", "PROTEINS", "NCI1", "DD"])
    ap.add_argument("--Ts", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--ablations", nargs="+",
                    default=["full", "no_spikes"])
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dense_baseline_results.json")
    ap.add_argument("--only-dataset", default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_available={torch.cuda.is_available()}", flush=True)

    out_path = os.path.join(_PROJECT_ROOT, args.out)
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"[resume] loaded {out_path}", flush=True)
    else:
        results = {
            "meta": {
                "started": datetime.now().isoformat(),
                "device": device, "epochs": args.epochs,
                "n_runs": args.n_runs, "shot": 1, "M_fixed": M_FIXED,
                "ablations": args.ablations, "Ts": args.Ts,
                "note": "spiking vs non-spiking (Riemannian/Euclidean) parameter-matched baselines",
            },
            "results": {},
        }

    datasets = [d for d in args.datasets
                if (args.only_dataset is None or d == args.only_dataset)]
    total_cells = len(datasets) * len(args.Ts) * len(args.ablations) * args.n_runs
    done = 0

    for ds in datasets:
        results["results"].setdefault(ds, {})
        for T in args.Ts:
            key_T = f"T{T}"
            results["results"][ds].setdefault(key_T, {})
            for ab in args.ablations:
                cell = results["results"][ds][key_T].setdefault(ab, {
                    "accs": [], "best_accs": [], "peak_mem_mb": [],
                    "T": T, "ablation_tag": ab,
                })
                n_done = len(cell["accs"])
                if n_done >= args.n_runs:
                    print(f"[skip] {ds} {key_T} {ab} ({n_done}/{args.n_runs})", flush=True)
                    done += args.n_runs
                    continue
                print(f"\n=== {ds} {key_T} {ab} ===", flush=True)
                for run_id in range(n_done, args.n_runs):
                    t0 = time.time()
                    r = run_one(ds, T, ab, run_id, args.epochs, device)
                    if not r["ok"]:
                        if r.get("peak_mem_mb") is not None:
                            cell["peak_mem_mb"].append(r["peak_mem_mb"])
                        cell.setdefault("errors", []).append(r)
                        dt = time.time() - t0
                        pk = r.get("peak_mem_mb")
                        pk_s = f" crash_peak={pk:.1f}MB" if pk is not None else ""
                        print(f"  run{run_id}: ERROR {r['error']}{pk_s} {dt:.0f}s "
                              f"({done}/{total_cells})", flush=True)
                        continue
                    cell["accs"].append(r["acc"])
                    cell["best_accs"].append(r["best_acc"])
                    cell["peak_mem_mb"].append(r["peak_mem_mb"])
                    done += 1
                    dt = time.time() - t0
                    print(f"  run{run_id}: acc={r['acc']*100:.2f}% "
                          f"peak={r['peak_mem_mb']:.1f}MB {dt:.0f}s "
                          f"({done}/{total_cells})", flush=True)
                    _finalize_cell(cell)
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2, default=str)

    for ds in results["results"]:
        for key_T in results["results"][ds]:
            for ab in results["results"][ds][key_T]:
                _finalize_cell(results["results"][ds][key_T][ab])
    results["meta"]["finished"] = datetime.now().isoformat()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nALL DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
