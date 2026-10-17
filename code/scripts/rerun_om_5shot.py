#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rerun_om_5shot.py — phase-2 supplement
======================================

Fill in the **5-shot** column for the two SpikeRPGNN cores (OM / No-Spikes)
that the main 28-config matrix (rerun_28configs.py) runs ONLY at 1-shot.

Why this exists
---------------
rerun_28configs.py dispatches OM and No-Spikes with `--shot 1` only, so after
the full matrix finishes these two paper-critical columns LACK a 5-shot number.
Because the P0-② metric-correction fix touches exactly these SpikeRPGNN paths,
the 5-shot numbers must ALSO be regenerated from the CORRECTED code — the old
paper 5-shot values are stale (they predate the fix). This script regenerates
only the 5-shot column for OM and No-Spikes on all four paper datasets, WITHOUT
touching the already-computed 1-shot results:

  - main matrix writes:  rerun_outputs/<DS>_OM/<DS>_1shot_full.json
  - this script writes:  rerun_outputs/<DS>_OM/<DS>_5shot_full.json   (new file)
  Different filename => the 1-shot work is never overwritten or re-run.

Run AFTER rerun_28configs.py finishes, so we (a) don't contend for the single
GPU and (b) don't waste the 1-shot work already done. Resume-safe: if a
5-shot output already exists it is skipped.

Usage
-----
  python rerun_om_5shot.py --dry-run      # print plan, no execution
  python rerun_om_5shot.py               # preflight + run OM/No-Spikes 5-shot
  python rerun_om_5shot.py --only MUTAG   # restrict to one dataset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Allow an explicit interpreter (e.g. the torch-enabled one) to be injected;
# default to the interpreter that launched this script.
_PY = os.environ.get("PYEXE") or sys.executable

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
# (method_label, --ablation value, save_subdir under rerun_outputs)
_TARGETS = [
    ("OM", "full", "OM"),
    ("No-Spikes", "no_spikes", "NoSpikes"),
]


def preflight() -> bool:
    """Mirror rerun_28configs.py: CUDA + import checks, auto-remap CUDA_VISIBLE_DEVICES."""
    print("=" * 64)
    print("PREFLIGHT (rerun_om_5shot)")
    print("=" * 64)
    ok = True

    try:
        import torch
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
            print("  [BLOCK] CUDA not available. This supplement needs a GPU.")
            ok = False
    except Exception as e:  # noqa
        print(f"  [BLOCK] torch import failed: {e}")
        ok = False

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    try:
        import spikergnn  # noqa
        print(f"  spikergnn importable: {spikergnn.__file__}")
    except Exception as e:  # noqa
        print(f"  [BLOCK] cannot import spikergnn: {e}")
        print("           fix: launch with PYTHONPATH=$PROJECT_ROOT/programfiles (no setup.py)")
        ok = False

    try:
        import torch_geometric  # noqa
        print(f"  torch_geometric {torch_geometric.__version__}")
    except Exception as e:  # noqa
        print(f"  [BLOCK] torch_geometric missing: {e}")
        ok = False

    if not ok:
        print("-" * 64)
        print("PREFLIGHT FAILED — refusing to run.")
        print("=" * 64)
        return False
    print("PREFLIGHT OK.")
    print("=" * 64)
    return True


def build_jobs(only_dataset: str = None) -> list:
    """Return list of (dataset, method_label, out_json_path, cmd) tuples."""
    jobs = []
    for ds in DATASETS:
        if only_dataset and ds != only_dataset:
            continue
        for label, abl, sub in _TARGETS:
            save = os.path.join(_PROJECT_ROOT, "rerun_outputs", f"{ds}_{sub}")
            out_json = os.path.join(save, f"{ds}_5shot_{abl}.json")
            cmd = [
                _PY, "-m", "spikergnn.run_experiment",
                "--dataset", ds, "--shot", "5", "--ablation", abl,
                "--n-runs", "9", "--device", "cuda",
                "--save-dir", save,
            ]
            jobs.append((ds, label, out_json, cmd))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2: OM/No-Spikes 5-shot supplement")
    ap.add_argument("--dry-run", action="store_true", help="print plan, no run")
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    args = ap.parse_args()

    jobs = build_jobs(args.only)
    print(f"\nPhase-2 plan: {len(jobs)} jobs "
          f"(OM + No-Spikes, 5-shot, 4 datasets)\n")

    if args.dry_run:
        for i, (ds, label, out_json, cmd) in enumerate(jobs, 1):
            print(f"  [{i:2d}] 5-shot {ds}/{label}")
            print(f"        -> {out_json}")
            print(f"        {' '.join(cmd)}")
        print("\n(dry-run: no execution)")
        return 0

    if not preflight():
        return 2

    out_root = os.path.join(_PROJECT_ROOT, "rerun_outputs")
    log_root = os.path.join(_PROJECT_ROOT, "rerun_logs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    manifest = []

    t0 = time.time()
    for i, (ds, label, out_json, cmd) in enumerate(jobs, 1):
        if os.path.exists(out_json):
            print(f"[{i}/{len(jobs)}] SKIP {ds}/{label} (5-shot already at {out_json})")
            manifest.append({"dataset": ds, "method": label, "status": "skipped"})
            continue
        log_path = os.path.join(log_root, f"{ds}__{label}__5shot.log")
        print(f"[{i}/{len(jobs)}] RUN  {ds}/{label} (5-shot) -> {log_path}")
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, cwd=_PROJECT_ROOT, stdout=lf,
                                stderr=subprocess.STDOUT)
        status = "done" if rc.returncode == 0 else f"FAILED(rc={rc.returncode})"
        manifest.append({"dataset": ds, "method": label, "status": status,
                         "returncode": rc.returncode, "out_json": out_json})
        print(f"         => {status}")

    with open(os.path.join(out_root, "manifest_5shot.json"), "w") as f:
        json.dump({"finished_at": datetime.now().isoformat(),
                   "elapsed_sec": time.time() - t0,
                   "jobs": manifest}, f, indent=2)
    n_done = sum(1 for j in manifest if j["status"] == "done")
    n_skip = sum(1 for j in manifest if j["status"] == "skipped")
    n_fail = sum(1 for j in manifest if j["status"].startswith("FAILED"))
    print(f"\nPhase-2 done. {n_done} run / {n_skip} skipped / {n_fail} failed "
          f"of {len(jobs)} jobs. Manifest -> rerun_outputs/manifest_5shot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
