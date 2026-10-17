#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backfill_nospike.py — recompute tab:nospike_scope from fresh run JSONs.
========================================================================

The manuscript table tab:nospike_scope reports, for each of the four TU
datasets and for BOTH 1-shot and 5-shot, the spiking OnManifold (OM) accuracy
and the non-spiking continuous (No-Spk) ablation accuracy, plus the gap Delta
(percentage points) and the paired Wilcoxon signed-rank p-value.

Data sources (the authoritative, post-fix runs):
  OM      : rerun_outputs/<DS>_OM/<DS>_{shot}shot_full.json
  No-Spk  : rerun_outputs/<DS>_NoSpikes/<DS>_{shot}shot_no_spikes.json

Freshness rule: the N1 neuron fix (programfiles/neurons.py) changed the
no_spikes ablation's behaviour for EVERY shot. Any No-Spk JSON whose mtime is
older than neurons.py is PRE-fix and MUST NOT be used. A file is "fresh" only
if mtime > neurons.py mtime.

OM numbers are invariant to the fix (c=1.0 -> numerically identical), so they
are accepted regardless of age, but we still report their mtime for audit.

Std convention: the paper table uses the SAMPLE standard deviation (ddof=1),
matching the OM column. The run-file `std_acc` is the population std (ddof=0);
we recompute ddof=1 from the raw accs before emitting table values.

Seeds are aligned by run index (seed = seed_base + run_id*100 + shot*10, with
no method field), so OM accs[i] and No-Spk accs[i] are a valid paired sample.

Outputs:
  - a human-readable report (current state; flags MISSING / STALE / PENDING)
  - verify_suite/nospike_scope_expected.json : the single source of truth that
    verify_numbers.py compares the tex table against (present=false marks a
    cell that must still be re-run, which forces verify to stay RED).
  - when every cell is fresh, a LaTeX fragment for tab:nospike_scope and the
    prose Delta/p values used at L1296 / L1352 / L1375.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PROG = os.path.join(ROOT, "programfiles")
DATA_ROOT = os.environ.get("DVM_DATA_ROOT", ROOT)
EXPECTED = os.path.join(_HERE, "nospike_scope_expected.json")

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
SHOTS = [1, 5]

NEURONS = os.path.join(ROOT, "spikergnn", "neurons.py")
NEURONS_MTIME = os.path.getmtime(NEURONS) if os.path.exists(NEURONS) else 0.0


def _load(path):
    with open(path) as f:
        return json.load(f)


def _fresh(path):
    """A No-Spk file is fresh iff its mtime is newer than neurons.py (N1 fix)."""
    if not os.path.exists(path):
        return False, "MISSING"
    mt = os.path.getmtime(path)
    if mt <= NEURONS_MTIME:
        return False, "STALE(pre-fix)"
    return True, "fresh"


def _wilcoxon_p(om_accs, ns_accs):
    from scipy.stats import wilcoxon
    diffs = [n - o for o, n in zip(om_accs, ns_accs)]
    try:
        return wilcoxon(diffs).pvalue
    except ValueError:
        # zero variance in diffs -> Wilcoxon undefined; treat as 1.0 (no signal)
        return 1.0


def main():
    print(f"neurons.py mtime = {time.strftime('%Y-%m-%d_%H:%M', time.localtime(NEURONS_MTIME))}")
    print(f"{'dataset':9s} {'shot':4s} {'OM_mean':>8s} {'OM_std':>7s} "
          f"{'NS_mean':>8s} {'NS_std':>7s} {'fresh?':>14s} {'Delta':>7s} {'p':>8s} {'+/9':>4s}")
    print("-" * 92)

    rows = {}
    all_fresh = True
    expected = {}
    for ds in DATASETS:
        expected[ds] = {}
        for shot in SHOTS:
            om_path = os.path.join(DATA_ROOT, "rerun_outputs", f"{ds}_OM",
                                   f"{ds}_{shot}shot_full.json")
            ns_path = os.path.join(DATA_ROOT, "rerun_outputs", f"{ds}_NoSpikes",
                                   f"{ds}_{shot}shot_no_spikes.json")
            om = _load(om_path) if os.path.exists(om_path) else None
            ns = _load(ns_path) if os.path.exists(ns_path) else None
            entry = dict(present=False)
            if om is None or ns is None:
                all_fresh = False
                tag = "MISSING"
                print(f"{ds:9s} {shot:<4d} {'-':>8s} {'-':>7s} {'-':>8s} {'-':>7s} "
                      f"{tag:>14s} {'-':>7s} {'-':>8s} {'-':>4s}")
            else:
                fresh, tag = _fresh(ns_path)
                if not fresh:
                    all_fresh = False
                om_accs = np.array(om["accs"], dtype=float)
                ns_accs = np.array(ns["accs"], dtype=float)
                om_mean = om_accs.mean()
                om_std = om_accs.std(ddof=1)          # sample std, matches OM column
                ns_mean = ns_accs.mean()
                ns_std = ns_accs.std(ddof=1)
                delta = (ns_mean - om_mean) * 100.0
                p = _wilcoxon_p(om_accs.tolist(), ns_accs.tolist())
                n_pos = int((ns_accs > om_accs).sum())
                rows[(ds, shot)] = dict(om_mean=om_mean, om_std=om_std,
                                        ns_mean=ns_mean, ns_std=ns_std,
                                        delta=delta, p=p, n_pos=n_pos)
                entry = dict(present=True,
                             om_mean=round(om_mean * 100, 4),
                             om_std=round(om_std * 100, 4),
                             ns_mean=round(ns_mean * 100, 4),
                             ns_std=round(ns_std * 100, 4),
                             delta=round(delta, 4),
                             p=round(float(p), 6),
                             n_pos=n_pos)
                print(f"{ds:9s} {shot:<4d} {om_mean*100:8.2f} {om_std*100:7.2f} "
                      f"{ns_mean*100:8.2f} {ns_std*100:7.2f} {tag:>14s} "
                      f"{delta:+7.2f} {p:8.4f} {n_pos:>2d}/9")
            expected[ds][str(shot)] = entry

    # Always (re)write the consolidated expected-values file so verify_numbers
    # can guard the table even while cells are still pending.
    with open(EXPECTED, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"\n[written] {EXPECTED}")

    print("-" * 92)
    if not all_fresh:
        print("STATUS: NOT ALL CELLS FRESH. The following must be re-run "
              "(post-N1-fix) before backfilling the paper:")
        for ds in DATASETS:
            for shot in SHOTS:
                ns_path = os.path.join(DATA_ROOT, "rerun_outputs", f"{ds}_NoSpikes",
                                       f"{ds}_{shot}shot_no_spikes.json")
                fresh, tag = _fresh(ns_path)
                if not os.path.exists(ns_path):
                    print(f"   - {ds} {shot}-shot No-Spikes: MISSING "
                          f"(rerun_28configs.py --method No-Spikes,No-Spikes-5shot)")
                elif not fresh:
                    print(f"   - {ds} {shot}-shot No-Spikes: STALE(pre-fix) "
                          f"(rerun_28configs.py --method No-Spikes,No-Spikes-5shot)")
        return 1

    # All fresh -> emit LaTeX backfill.
    print("\nSTATUS: ALL CELLS FRESH. LaTeX backfill for tab:nospike_scope:\n")
    line_parts = []
    for ds in DATASETS:
        r1 = rows[(ds, 1)]
        r5 = rows[(ds, 5)]

        def fmt(r):
            return (f"{r['om_mean']*100:.2f}$\\pm${r['om_std']*100:.2f} "
                    f"& {r['ns_mean']*100:.2f}$\\pm${r['ns_std']*100:.2f} "
                    f"& {r['delta']:+.2f} ({r['p']:.3f})")
        line_parts.append(f"    {ds} & {fmt(r1)} & {fmt(r5)} \\\\")
    print("\\begin{tabular}{@{}lcccccc@{}}")
    print("\\toprule")
    print("Dataset & \\multicolumn{3}{c}{1-shot} & \\multicolumn{3}{c}{5-shot} \\\\")
    print("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    print(" & OM & No-Spk & $\\Delta$ (p) & OM & No-Spk & $\\Delta$ (p) \\\\")
    print("\\midrule")
    for lp in line_parts:
        print(lp)
    print("\\bottomrule")
    print("\\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
