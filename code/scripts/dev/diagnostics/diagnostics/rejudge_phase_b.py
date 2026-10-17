"""rejudge_phase_b.py -- re-apply the Phase B verdict with a corrected rule.

WHY THIS FILE EXISTS (2026-10-02)
--------------------------------
`phase_b_radius_scan.py` shipped the pre-registered rule

    acc_recovered = max(accs) > chance + 5

where `accs` pools EVERY scale arm.  That is a maximum-over-arms test, i.e. a
cherry-pick: an arm that does NOT restore firing (e.g. s0.5) can satisfy it
alone and drag the whole dataset into the "H2-causal" bucket.  On the real
NCI1 run the rule did exactly that -- s0.5 (tail3 fire = 0.0%) reached 56.0%
and the dataset was reported as "firing AND accuracy both recover", while
the arms that actually restored firing (s4/s8) showed 54.8%/54.2%, i.e. no
accuracy movement at all.

The raw measurements are NOT touched.  This script re-reads the JSON and
emits a corrected verdict, keeping the original string next to it so the
change is auditable rather than silent.

Corrected rule (all three conditions must use the SAME arm, the one whose
firing actually recovered):

    fire_recovered  = tail3_fire(s8.0) - tail3_fire(s1.0) > 5 pp
    acc_recovered   = mean_eval_acc(s8.0) - mean_eval_acc(s1.0)
                      > max(5 pp, 2 * pooled_noise)
    pooled_noise    = max(std of last-3 eval_acc within any arm, 0.5 pp)

Exit codes / verdicts
    H2-causal                                   fire + accuracy recover
    drive-restored-but-representation-collapsed fire recovers, accuracy does not
    H2-incomplete                               firing does not recover
    no-collapse-at-base                         sentinel: base already saturated
    insufficient-arms                          s-family incomplete

Usage
    PYTHONPATH=$PWD/programfiles python rejudge_phase_b.py \
        --inputs rerun_outputs/diagnostics/phase_b_NCI1.json \
                rerun_outputs/diagnostics/phase_b_MUTAG.json \
        --out rerun_outputs/diagnostics/phase_b_rejudged.json
"""

import argparse
import json
import math
import os

# NOTE: `import statistics` is deliberately avoided -- with
# PYTHONPATH=$PWD/programfiles the project's own `programfiles/statistics.py`
# (BH-FDR / significance tests) shadows the stdlib module and
# `statistics.stdev` resolves to AttributeError.  Renaming that file is the
# proper fix but is deferred: the running supplement job imports it.

TAIL = 3          # epochs averaged at the end of a schedule
FIRE_EPS = 5.0    # pp of firing recovery required
ACC_EPS = 5.0     # pp of accuracy recovery required


def _stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _tail(curve, key, k=TAIL):
    vals = [e[key] for e in curve[-k:]] if curve else []
    return sum(vals) / len(vals) if vals else float("nan")


def _arm(rows, pred):
    for r in rows:
        if pred(r):
            return r
    return None


def rejudge(doc):
    ds = doc["dataset"]
    rows = doc["rows"]
    chance = 50.0

    # within-arm noise of the accuracy estimate (last TAIL epochs)
    stds = []
    for r in rows:
        accs = [e["eval_acc"] for e in r["fire_curve"][-TAIL:]]
        if len(accs) > 1:
            stds.append(_stdev(accs))
    noise = max(max(stds) if stds else 0.0, 0.5)

    s1 = _arm(rows, lambda r: r["scale"] is not None
              and abs(r["scale"] - 1.0) < 1e-9)
    s8 = _arm(rows, lambda r: r["scale"] is not None
              and abs(r["scale"] - 8.0) < 1e-9)
    scale_arms = [r for r in rows if r["scale"] is not None]

    detail = {
        "pooled_noise_pp": round(noise, 2),
        "acc_threshold_pp": round(max(ACC_EPS, 2 * noise), 2),
        "n_scale_arms": len(scale_arms),
    }

    if not scale_arms:
        return "insufficient-arms", detail
    if s1 is None or s8 is None:
        return "insufficient-arms", detail

    fire1 = s1["stmp_fire_pct_tail3"]
    fire8 = s8["stmp_fire_pct_tail3"]
    acc1 = _tail(s1["fire_curve"], "eval_acc")
    acc8 = _tail(s8["fire_curve"], "eval_acc")
    d_fire, d_acc = fire8 - fire1, acc8 - acc1
    detail.update({
        "s1.0": {"tail3_fire": fire1, "tail3_acc": acc1},
        "s8.0": {"tail3_fire": fire8, "tail3_acc": acc8},
        "delta_fire_pp": round(d_fire, 2),
        "delta_acc_pp": round(d_acc, 2),
        "monotone_in_s": all(
            b["stmp_fire_pct_tail3"] >= a["stmp_fire_pct_tail3"] - 1e-9
            for a, b in zip(scale_arms, scale_arms[1:])),
    })

    if fire1 >= 40.0:
        return "no-collapse-at-base", detail
    fire_rec = d_fire > FIRE_EPS
    acc_rec = d_acc > max(ACC_EPS, 2 * noise)
    if fire_rec and acc_rec:
        return "H2-causal", detail
    if fire_rec:
        return "drive-restored-but-representation-collapsed", detail
    return "H2-incomplete", detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="rerun_outputs/diagnostics/"
                                     "phase_b_rejudged.json")
    args = ap.parse_args()

    out = {"note": "verdict recomputed with the same-arm rule; original "
                   "verdict preserved for audit", "datasets": {}}
    for p in args.inputs:
        with open(p) as f:
            doc = json.load(f)
        verdict, detail = rejudge(doc)
        ds = doc["dataset"]
        entry = out["datasets"].setdefault(ds, {})
        entry["original_verdict"] = doc.get("verdict")
        entry["corrected_verdict"] = verdict
        entry["detail"] = detail
        print(f"[{ds}] original : {doc.get('verdict')}")
        print(f"[{ds}] corrected: {verdict}")
        for k, v in detail.items():
            print(f"          {k}: {v}")
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
