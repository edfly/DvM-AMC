#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rerun_fixed_supplement.py — corrected-code supplement (supersedes rerun_om_5shot.py)
====================================================================================

Why this exists
---------------
Two things are missing/broken after the main 28-config matrix (rerun_28configs.py):

1. **No-Spikes implementation was numerically broken.** The `no_spikes` ablation
   short-circuited the LIF recurrence, diverging to NaN and collapsing to a
   constant class-0 prediction -> an exact 50.00% on balanced binary tasks.
   This was fixed on 2026-10-01 in programfiles/neurons.py: the ablation now
   RETAINS the full T-step recurrence and manifold state evolution and only
   replaces the all-or-none spike with a continuous activation
   sigma(u_mag - u_th). Evidence of the old failure:
   programfiles/diagnostics/diagnose_no_spikes.py -> diag_no_spikes.log.

   => the No-Spikes 1-shot numbers in rerun_outputs/<DS>_NoSpikes/ (50.00%) are
      INVALID and must be regenerated with the corrected code. The old JSONs are
      preserved as evidence; corrected numbers are written to a NEW directory
      <DS>_NoSpikesFixed/ so nothing is overwritten.

2. **OM / No-Spikes had no 5-shot column** (the matrix runs them at 1-shot only).
   OM 5-shot is unaffected by the No-Spikes bug (ablation="full"); its 1/4 jobs
   already exist and are skipped here. No-Spikes 5-shot is regenerated with the
   corrected code. Both are covered by this script, so rerun_om_5shot.py is now
   redundant.

This script therefore runs, per dataset (MUTAG, PROTEINS, NCI1, DD):
  - OM        5-shot  -> rerun_outputs/<DS>_OM/<DS>_5shot_full.json        (skip if present)
  - No-Spikes 1-shot  -> rerun_outputs/<DS>_NoSpikesFixed/<DS>_1shot_no_spikes.json
  - No-Spikes 5-shot  -> rerun_outputs/<DS>_NoSpikesFixed/<DS>_5shot_no_spikes.json
Resume-safe: any output that already exists is skipped.

Usage
-----
  python rerun_fixed_supplement.py --dry-run        # print plan, no execution
  python rerun_fixed_supplement.py                  # preflight + run (resume-safe)
  python rerun_fixed_supplement.py --only MUTAG     # restrict to one dataset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_PY = os.environ.get("PYEXE") or sys.executable

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]

# (method_label, --ablation, shot, save_subdir, output_stem)
_TARGETS = [
    ("OM",         "full",      "5", "OM",             "5shot_full"),
    ("No-Spikes",  "no_spikes", "1", "NoSpikesFixed",  "1shot_no_spikes"),
    ("No-Spikes",  "no_spikes", "5", "NoSpikesFixed",  "5shot_no_spikes"),
]


def preflight() -> bool:
    """Mirror rerun_28configs.py: CUDA + import checks, auto-remap CUDA_VISIBLE_DEVICES."""
    print("=" * 64)
    print("PREFLIGHT (rerun_fixed_supplement)")
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

    # Sanity: the no_spikes fix must be present (guard against launching on stale code).
    try:
        import inspect
        from spikergnn.neurons import ManifoldLIFNeuron
        src = inspect.getsource(ManifoldLIFNeuron.forward)
        if "continuous = (self.ablation == \"no_spikes\")" in src:
            print("  no_spikes FIX present (continuous-activation branch found)")
        else:
            print("  [BLOCK] no_spikes fix NOT found in neurons.py — refusing to run "
                  "(would regenerate the broken 50.00% numbers).")
            ok = False
    except Exception as e:  # noqa
        print(f"  [BLOCK] could not verify the no_spikes fix: {e}")
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
    """Return list of (dataset, label, shot, out_json, cmd) tuples."""
    jobs = []
    for ds in DATASETS:
        if only_dataset and ds != only_dataset:
            continue
        for label, abl, shot, sub, stem in _TARGETS:
            save = os.path.join(_PROJECT_ROOT, "rerun_outputs", f"{ds}_{sub}")
            out_json = os.path.join(save, f"{ds}_{stem}.json")
            cmd = [
                _PY, "-m", "spikergnn.run_experiment",
                "--dataset", ds, "--shot", shot, "--ablation", abl,
                "--n-runs", "9", "--device", "cuda",
                "--save-dir", save,
            ]
            jobs.append((ds, label, shot, out_json, cmd))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Corrected-code supplement: No-Spikes (1+5 shot) + OM 5-shot")
    ap.add_argument("--dry-run", action="store_true", help="print plan, no run")
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    args = ap.parse_args()

    jobs = build_jobs(args.only)
    print(f"\nSupplement plan: {len(jobs)} jobs "
          f"(OM 5-shot + No-Spikes 1-shot + No-Spikes 5-shot, 4 datasets)\n")

    if args.dry_run:
        for i, (ds, label, shot, out_json, cmd) in enumerate(jobs, 1):
            print(f"  [{i:2d}] {label:10s} {shot}-shot  {ds}")
            print(f"        -> {out_json}")
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
    for i, (ds, label, shot, out_json, cmd) in enumerate(jobs, 1):
        if os.path.exists(out_json):
            print(f"[{i}/{len(jobs)}] SKIP {ds}/{label} {shot}-shot (exists: {out_json})")
            manifest.append({"dataset": ds, "method": label, "shot": shot,
                             "status": "skipped"})
            continue
        log_path = os.path.join(log_root, f"{ds}__{label}__{shot}shot.log")
        print(f"[{i}/{len(jobs)}] RUN  {ds}/{label} {shot}-shot -> {log_path}")
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, cwd=_PROJECT_ROOT, stdout=lf,
                                stderr=subprocess.STDOUT)
        status = "done" if rc.returncode == 0 else f"FAILED(rc={rc.returncode})"
        manifest.append({"dataset": ds, "method": label, "shot": shot,
                         "status": status, "returncode": rc.returncode,
                         "out_json": out_json})
        print(f"         => {status}")

    with open(os.path.join(out_root, "manifest_fixed_supplement.json"), "w") as f:
        json.dump({"finished_at": datetime.now().isoformat(),
                   "elapsed_sec": time.time() - t0,
                   "jobs": manifest}, f, indent=2)
    n_done = sum(1 for j in manifest if j["status"] == "done")
    n_skip = sum(1 for j in manifest if j["status"] == "skipped")
    n_fail = sum(1 for j in manifest if j["status"].startswith("FAILED"))
    print(f"\nSupplement done. {n_done} run / {n_skip} skipped / {n_fail} failed "
          f"of {len(jobs)} jobs. Manifest -> rerun_outputs/manifest_fixed_supplement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
