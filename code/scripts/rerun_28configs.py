#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rerun_28configs.py — turnkey rerun of the 28-config experiment matrix
======================================================================

Purpose
-------
After the P0-② fix (metric correction now reaches the recurrent weights in
truncated-BPTT mode, see programfiles/stmp.py), the experiment matrix is
re-run so the manuscript tables reflect the CORRECTED gradient.

IMPORTANT — what is in the matrix (verified 2026-09-23):
  The metric-correction fix ONLY affects SpikeRPGNN (OM / No-Spikes /
  No-Correction / Full-BPTT / Full-BPTT+MC). The GNN baselines
  SGC / GIN / GCN / SAGE do NOT use DvM at all, so their numbers are
  INVARIANT to the fix — but they MUST be reproducible from source in their
  own right, so each now has a real runnable entry point:
  programfiles/train_gnn_baselines.py (Euclidean GNN encoder + Euclidean
  ProtoNet, reusing msg_baseline's data/sampling/eval protocol). They are
  RUN (not SKIPPED) so the entire 28-config matrix is fresh and self-consistent;
  their existing paper numbers are expected to be reproduced, not blindly
  overwritten.

Per dataset (MUTAG, PROTEINS, NCI1, DD) the RUN jobs are:
  1. OM            -> python -m spikergnn.run_experiment --ablation full
  2. No-Spikes     -> python -m spikergnn.run_experiment --ablation no_spikes
  3. MSG*          -> python -m msg_baseline.run_flexible --datasets <ds>
  4. MC A/B set    -> python programfiles/run_final_matrix.py --dataset <ds>
                     (covers No Correction M=8, Full BPTT, Full BPTT+MC, Proto-GNN)
  5. SGC/GIN/GCN/SAGE -> python programfiles/train_gnn_baselines.py --method <m> --datasets <ds>
                     (Euclidean GNN baseline; invariant to fix but re-run for full reproducibility)

Environment requirement (HARD)
------------------------------
This matrix is GPU-bound. Run it on a machine with:
  - CUDA GPU + torch (cuda=True)
  - `spikergnn` + `msg_baseline` importable. There is NO setup.py, so do NOT
    use pip install. Instead set PYTHONPATH to the programfiles dir:
        export PYTHONPATH="$PWD/programfiles"
    (programfiles/ is the spikergnn package; programfiles/msg_baseline is the
     msg_baseline package — both resolve from that one path.)
  - torch_geometric + TU datasets / ogbn-arxiv downloaded

NOTE — CUDA_VISIBLE_DEVICES:
  The preflight below auto-remaps a mis-set CUDA_VISIBLE_DEVICES to "0" when
  it points past the real device count (observed on this box: env had "1"
  while only GPU 0 exists, which made torch.cuda.is_available()==False even
  though the RTX 2060 driver was fine). If you launch manually, set
  CUDA_VISIBLE_DEVICES=0 (or unset it) so torch sees the card.

Usage
-----
  python rerun_28configs.py --dry-run      # print the plan, no run
  python rerun_28configs.py               # preflight + run all (resume-safe)
  python rerun_28configs.py --only MUTAG   # restrict to one dataset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PY = sys.executable

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
# GNN baselines do not use DvM -> invariant to the fix -> re-run for full matrix reproducibility.
_GNN_BASELINES = ["SGC", "GIN", "GCN", "SAGE"]

# Filesystem-safe variant of a method label. Some method names (e.g. the
# MC-A/B set) contain '/', '(', ')' which are illegal/ambiguous in paths and
# were interpreted as directory separators -> FileNotFoundError at log open.
# Only used for .done markers and .log filenames; the subprocess command and
# human-readable labels keep the original name.
_SAFE_TRANS = str.maketrans({"/": "_", "(": "_", ")": ""})


def _safe_name(m: str) -> str:
    return m.translate(_SAFE_TRANS)


def preflight(allow_cpu: bool = False) -> bool:
    print("=" * 64)
    print("PREFLIGHT")
    print("=" * 64)
    ok = True

    # 1. torch + CUDA
    try:
        import torch
        # Auto-remap a mis-set CUDA_VISIBLE_DEVICES that points past the real
        # device count (e.g. env had "1" but only GPU 0 exists) — this makes
        # torch see the card without the user having to fix the env manually.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            try:
                idxs = [int(x) for x in cvd.split(",") if x.strip() != ""]
                n_dev = torch.cuda.device_count()
                if any(i >= n_dev for i in idxs):
                    print(f"  [FIX] CUDA_VISIBLE_DEVICES={cvd} but only {n_dev} "
                          f"device(s); remapping to '0'")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            except ValueError:
                pass
        print(f"  torch {torch.__version__} | cuda available = {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("  [BLOCK] CUDA not available. The 28-config matrix needs a GPU.")
            ok = False
    except Exception as e:  # noqa
        print(f"  [BLOCK] torch import failed: {e}")
        ok = False

    # 2. spikergnn package importable
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    try:
        import spikergnn  # noqa
        print(f"  spikergnn importable: {spikergnn.__file__}")
    except Exception as e:  # noqa
        print(f"  [BLOCK] cannot import spikergnn: {e}")
        print("           fix: export PYTHONPATH=$PROJECT_ROOT/programfiles  (no setup.py exists)")
        ok = False

    # 3. torch_geometric
    try:
        import torch_geometric  # noqa
        print(f"  torch_geometric {torch_geometric.__version__}")
    except Exception as e:  # noqa
        print(f"  [BLOCK] torch_geometric missing: {e}")
        ok = False

    # 4. data presence (soft warning, not a hard block — runners can download)
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    print(f"  data dir: {data_dir} {'EXISTS' if os.path.isdir(data_dir) else 'MISSING'}")

    if not ok and not allow_cpu:
        print("-" * 64)
        print("PREFLIGHT FAILED — refusing to run on this environment.")
        print("Run on a CUDA GPU machine with spikergnn + torch_geometric installed.")
        print("=" * 64)
        return False
    print("PREFLIGHT OK.")
    print("=" * 64)
    return True


def build_matrix(only_dataset: str = None, only_method: str = None) -> list:
    """Return list of (dataset, method, cmd-or-None, kind) tuples.

    kind is "run" (cmd is a real command) or "skip" (cmd is None, with a
    note in the method label explaining why).
    """
    jobs = []
    for ds in DATASETS:
        if only_dataset and ds != only_dataset:
            continue
        save = os.path.join(_PROJECT_ROOT, "rerun_outputs")

        # 1. OM — SpikeRPGNN full (THE fixed metric-correction path)
        jobs.append((ds, "OM", [
            _PY, "-m", "spikergnn.run_experiment",
            "--dataset", ds, "--shot", "1", "--ablation", "full",
            "--n-runs", "9", "--device", "cuda",
            "--save-dir", os.path.join(save, f"{ds}_OM"),
        ], "run"))

        # 2. No-Spikes — SpikeRPGNN no_spikes ablation
        #    N6 (2026-10-14): the original rerun only covered --shot 1, but the
        #    manuscript table tab:nospike_scope reports BOTH 1-shot and 5-shot
        #    No-Spikes numbers. The N1 neuron fix (unmasked recurrence + continuous
        #    rate proxy) changes the ablation's behaviour at every shot, so the
        #    5-shot column is also stale post-fix and must be re-measured. We therefore
        #    emit two jobs: 1-shot (label "No-Spikes") and 5-shot (label
        #    "No-Spikes-5shot", distinct .done marker / save file). Both write into
        #    the same save-dir as <DATASET>_{1shot,5shot}_no_spikes.json.
        jobs.append((ds, "No-Spikes", [
            _PY, "-m", "spikergnn.run_experiment",
            "--dataset", ds, "--shot", "1", "--ablation", "no_spikes",
            "--n-runs", "9", "--device", "cuda",
            "--save-dir", os.path.join(save, f"{ds}_NoSpikes"),
        ], "run"))
        jobs.append((ds, "No-Spikes-5shot", [
            _PY, "-m", "spikergnn.run_experiment",
            "--dataset", ds, "--shot", "5", "--ablation", "no_spikes",
            "--n-runs", "9", "--device", "cuda",
            "--save-dir", os.path.join(save, f"{ds}_NoSpikes"),
        ], "run"))

        # 3. MSG* baseline
        jobs.append((ds, "MSG", [
            _PY, "-m", "msg_baseline.run_flexible",
            "--datasets", ds, "--shots", "1", "5", "--n-runs", "9",
        ], "run"))

        # 4. Metric-correction A/B set (No Correction M=8 / Full BPTT /
        #    Full BPTT+MC / Proto-GNN) — validates the fix directly.
        jobs.append((ds, "MC-A/B(NoCorr/FullBPTT/FullBPTT+MC/ProtoGNN)", [
            _PY, os.path.join(_PROJECT_ROOT, "programfiles", "run_final_matrix.py"),
            "--dataset", ds, "--device", "cuda", "--n-runs", "9",
        ], "run"))

        # 5. GNN baselines — Euclidean GNN encoder + Euclidean ProtoNet.
        #    Invariant to the metric-correction fix, but re-run so the 28-config
        #    matrix is fully reproducible from source (not just "carried over").
        for m in _GNN_BASELINES:
            jobs.append((ds, m, [
                _PY, os.path.join(_PROJECT_ROOT, "programfiles", "train_gnn_baselines.py"),
                "--method", m.lower(), "--datasets", ds,
                "--shots", "1", "5", "--n-runs", "9", "--device", "cuda",
            ], "run"))
    # method filter (added 2026-10-13: lets us rerun only the arms whose
    # behaviour changed after a code fix, e.g. No-Spikes after the
    # no_spikes-neuron state-evolution fix).
    if only_method:
        wanted = [m.strip() for m in only_method.split(",") if m.strip()]
        jobs = [j for j in jobs if j[1] in wanted]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    ap.add_argument("--method", default=None,
                    help="restrict to one or more comma-separated methods "
                         "(e.g. 'No-Spikes' or 'OM,No-Spikes')")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="allow running on CPU (NOT recommended; very slow)")
    args = ap.parse_args()

    jobs = build_matrix(args.only, args.method)

    n_run = sum(1 for j in jobs if j[3] == "run")
    n_skip = sum(1 for j in jobs if j[3] == "skip")
    print(f"\nExperiment plan: {len(jobs)} entries across 4 datasets")
    print(f"  RUN  : {n_run}  (all 7 methods incl. GNN baselines + MC A/B set)")
    print(f"  SKIP : {n_skip}  (none — full 28-config matrix is re-runnable)\n")

    if args.dry_run:
        for i, (ds, m, cmd, kind) in enumerate(jobs, 1):
            tag = "RUN " if kind == "run" else "SKIP"
            detail = (" ".join(cmd)) if cmd else "(invariant to fix; no entry point)"
            print(f"  [{i:2d}] {tag} {ds:9s} {m:9s}  {detail}")
        print("\n(dry-run: no execution)")
        return 0

    if not preflight(allow_cpu=args.allow_cpu):
        return 2
    if not args.allow_cpu:
        # double guard: never start the full matrix on CPU
        import torch
        if not torch.cuda.is_available():
            print("Refusing to run matrix on CPU. Use --allow-cpu only"
                  " for a tiny smoke test on a single config.")
            return 2

    out_root = os.path.join(_PROJECT_ROOT, "rerun_outputs")
    log_root = os.path.join(_PROJECT_ROOT, "rerun_logs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    manifest = []

    t0 = time.time()
    for i, (ds, m, cmd, kind) in enumerate(jobs, 1):
        marker = os.path.join(out_root, f"{ds}__{_safe_name(m)}.done")
        if kind == "skip":
            print(f"[{i}/{len(jobs)}] SKIP {ds}/{m} — invariant to fix, no entry point")
            manifest.append({"dataset": ds, "method": m, "status": "skipped"})
            continue
        if os.path.exists(marker):
            print(f"[{i}/{len(jobs)}] SKIP {ds}/{m} (already done)")
            manifest.append({"dataset": ds, "method": m, "status": "skipped"})
            continue
        log_path = os.path.join(log_root, f"{ds}__{_safe_name(m)}.log")
        print(f"[{i}/{len(jobs)}] RUN  {ds}/{m}  -> logging {log_path}")
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, cwd=_PROJECT_ROOT, stdout=lf,
                                stderr=subprocess.STDOUT)
        status = "done" if rc.returncode == 0 else f"FAILED(rc={rc.returncode})"
        if rc.returncode == 0:
            open(marker, "w").write(datetime.now().isoformat())
        manifest.append({"dataset": ds, "method": m, "status": status,
                         "returncode": rc.returncode})
        print(f"         => {status}")

    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump({"finished_at": datetime.now().isoformat(),
                   "elapsed_sec": time.time() - t0,
                   "jobs": manifest}, f, indent=2)
    print(f"\nDone. {sum(1 for j in manifest if j['status']=='done')} run / "
          f"{sum(1 for j in manifest if j['status']=='skipped')} skipped "
          f"of {len(jobs)} entries. Manifest -> rerun_outputs/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
