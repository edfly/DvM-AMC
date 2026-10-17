"""Four-fold verification (2/4): NUMBERS — every table value traced to source.

Parses result tables in the manuscript and checks each reported value against
the JSON artifact it cites. Adapted from papersh verify_numbers_template.

Two parse modes:
  'pct' : cells are "79.5\\% & 5.8\\%" -> percentage matches, col_idx 0=mean,1=std
  'col' : cells hold raw numbers (peak MB / rel_err); col_idx selects the
          numeric token in row order.

Exit 0 = all numbers match their sources.
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PROG = os.path.join(ROOT, "programfiles")
DATA_ROOT = os.environ.get("DVM_DATA_ROOT", ROOT)
TEX = os.path.join(os.path.dirname(ROOT), "paper", "DvM_Paper_Anonymous.tex")

# (label, json_path, mode, [(row_substr, accessor, col_idx, scale, tol), ...])
# accessor walked into JSON; 'peak' triggers peak_mem_mb[0] handling.
TABLES = [
    ("tab:gnn_results", os.path.join(DATA_ROOT, "final_results_MUTAG.json"), "pct", [
        ("Full BPTT (baseline)", ("Full BPTT", "mean"), 0, 100, 0.2),
        ("AMC $M{=}8$", ("DvM M=8 truncated", "mean"), 0, 100, 0.2),
        ("AMC truncated $M{=}4$", ("DvM M=4 truncated", "mean"), 0, 100, 0.2),
        ("No Correction $M{=}8$", ("No Correction M=8", "mean"), 0, 100, 0.2),
        ("Full BPTT $+$ Metric Corr.", ("Full BPTT + MC", "mean"), 0, 100, 0.2),
        ("Full BPTT (baseline)", ("Full BPTT", "std"), 1, 100, 0.2),
        ("AMC $M{=}8$", ("DvM M=8 truncated", "std"), 1, 100, 0.2),
        ("AMC truncated $M{=}4$", ("DvM M=4 truncated", "std"), 1, 100, 0.2),
        ("No Correction $M{=}8$", ("No Correction M=8", "std"), 1, 100, 0.2),
        ("Full BPTT $+$ Metric Corr.", ("Full BPTT + MC", "std"), 1, 100, 0.2),
    ]),
    ("tab:dd_results", os.path.join(DATA_ROOT, "final_results_DD.json"), "pct", [
        ("Full BPTT (baseline)", ("Full BPTT", "mean"), 0, 100, 0.2),
        ("AMC $M{=}8$", ("DvM M=8 truncated", "mean"), 0, 100, 0.2),
        ("AMC truncated $M{=}4$", ("DvM M=4 truncated", "mean"), 0, 100, 0.2),
        ("No Correction $M{=}8$", ("No Correction M=8", "mean"), 0, 100, 0.2),
        ("Full BPTT $+$ MC", ("Full BPTT + MC", "mean"), 0, 100, 0.2),
        ("Full BPTT (baseline)", ("Full BPTT", "std"), 1, 100, 0.2),
        ("AMC $M{=}8$", ("DvM M=8 truncated", "std"), 1, 100, 0.2),
        ("AMC truncated $M{=}4$", ("DvM M=4 truncated", "std"), 1, 100, 0.2),
        ("No Correction $M{=}8$", ("No Correction M=8", "std"), 1, 100, 0.2),
        ("Full BPTT $+$ MC", ("Full BPTT + MC", "std"), 1, 100, 0.2),
    ]),
    ("tab:measured_om", os.path.join(DATA_ROOT, "T_sweep", "T_sweep_MUTAG.json"), "col", [
        ("32 & AMC truncated", ("results", "MUTAG", "T32", "truncated", "peak"), 2, 1.0, 1.5),
        ("32 & Full BPTT (oracle)", ("results", "MUTAG", "T32", "full", "peak"), 2, 1.0, 1.5),
        ("64 & AMC truncated", ("results", "MUTAG", "T64", "truncated", "peak"), 2, 1.0, 1.5),
        ("64 & Full BPTT (oracle)", ("results", "MUTAG", "T64", "full", "peak"), 2, 1.0, 1.5),
    ]),
    # P0-1 long-horizon trade-off (single seed, 3 epochs probe; NCI1 cells pending sweep)
    ("tab:longT_tradeoff", os.path.join(DATA_ROOT, "ablations", "longT_accuracy_results.json"), "col", [
        # MUTAG
        ("93.8 & 68.0", ("results", "MUTAG", "T16", "truncated", "peak"), 2, 1.0, 2.0),
        ("93.8 & 68.0", ("results", "MUTAG", "T16", "truncated", "mean_acc"), 3, 100, 1.0),
        ("154.0 & 67.8", ("results", "MUTAG", "T16", "full", "peak"), 2, 1.0, 2.0),
        ("154.0 & 67.8", ("results", "MUTAG", "T16", "full", "mean_acc"), 3, 100, 1.0),
        ("105.7 & 68.6", ("results", "MUTAG", "T32", "truncated", "peak"), 2, 1.0, 2.0),
        ("105.7 & 68.6", ("results", "MUTAG", "T32", "truncated", "mean_acc"), 3, 100, 1.0),
        ("284.8 & ---", ("results", "MUTAG", "T32", "full", "peak"), 2, 1.0, 2.0),
        ("160.1 & 60.0", ("results", "MUTAG", "T64", "truncated", "peak"), 2, 1.0, 2.0),
        ("160.1 & 60.0", ("results", "MUTAG", "T64", "truncated", "mean_acc"), 3, 100, 1.0),
        ("524.5 & ---", ("results", "MUTAG", "T64", "full", "peak"), 2, 1.0, 2.0),
        # PROTEINS
        ("313.0 & 51.0", ("results", "PROTEINS", "T16", "truncated", "peak"), 2, 1.0, 2.0),
        ("313.0 & 51.0", ("results", "PROTEINS", "T16", "truncated", "mean_acc"), 3, 100, 1.0),
        ("407.7 & ---", ("results", "PROTEINS", "T16", "full", "peak"), 2, 1.0, 2.0),
        ("367.7 & 47.8", ("results", "PROTEINS", "T32", "truncated", "peak"), 2, 1.0, 2.0),
        ("367.7 & 47.8", ("results", "PROTEINS", "T32", "truncated", "mean_acc"), 3, 100, 1.0),
        ("714.5 & ---", ("results", "PROTEINS", "T32", "full", "peak"), 2, 1.0, 2.0),
        ("588.1 & 48.0", ("results", "PROTEINS", "T64", "truncated", "peak"), 2, 1.0, 2.0),
        ("588.1 & 48.0", ("results", "PROTEINS", "T64", "truncated", "mean_acc"), 3, 100, 1.0),
        ("1406.9 & ---", ("results", "PROTEINS", "T64", "full", "peak"), 2, 1.0, 2.0),
        # DD
        ("1334.1 & 49.8", ("results", "DD", "T16", "truncated", "peak"), 2, 1.0, 2.0),
        ("1334.1 & 49.8", ("results", "DD", "T16", "truncated", "mean_acc"), 3, 100, 1.0),
        ("2428.0 & 49.8", ("results", "DD", "T16", "full", "peak"), 2, 1.0, 2.0),
        ("2428.0 & 49.8", ("results", "DD", "T16", "full", "mean_acc"), 3, 100, 1.0),
        ("1573.3 & 50.4", ("results", "DD", "T32", "truncated", "peak"), 2, 1.0, 2.0),
        ("1573.3 & 50.4", ("results", "DD", "T32", "truncated", "mean_acc"), 3, 100, 1.0),
        ("4065.0 & ---", ("results", "DD", "T32", "full", "peak"), 2, 1.0, 2.0),
        ("2554.7 & 49.8", ("results", "DD", "T64", "truncated", "peak"), 2, 1.0, 2.0),
        ("2554.7 & 49.8", ("results", "DD", "T64", "truncated", "mean_acc"), 3, 100, 1.0),
        ("8412.9 & ---", ("results", "DD", "T64", "full", "peak"), 2, 1.0, 2.0),
        # NCI1 (full BPTT stable at all T; uses max_peak_mb because resume appended the failed-run peak to peak_mem_mb;
        #       NOTE: "NCI1" contains a digit so row_numbers shifts columns by +1: peak->col3, acc->col4)
        ("180.1 & 55.4", ("results", "NCI1", "T16", "truncated", "max_peak_mb"), 3, 1.0, 2.0),
        ("180.1 & 55.4", ("results", "NCI1", "T16", "truncated", "mean_acc"), 4, 100, 1.0),
        ("309.8 & 54.2", ("results", "NCI1", "T16", "full", "max_peak_mb"), 3, 1.0, 2.0),
        ("309.8 & 54.2", ("results", "NCI1", "T16", "full", "mean_acc"), 4, 100, 1.0),
        ("210.9 & 50.2", ("results", "NCI1", "T32", "truncated", "max_peak_mb"), 3, 1.0, 2.0),
        ("210.9 & 50.2", ("results", "NCI1", "T32", "truncated", "mean_acc"), 4, 100, 1.0),
        ("598.6 & 55.4", ("results", "NCI1", "T32", "full", "max_peak_mb"), 3, 1.0, 2.0),
        ("598.6 & 55.4", ("results", "NCI1", "T32", "full", "mean_acc"), 4, 100, 1.0),
        ("333.9 & 49.0", ("results", "NCI1", "T64", "truncated", "max_peak_mb"), 3, 1.0, 2.0),
        ("333.9 & 49.0", ("results", "NCI1", "T64", "truncated", "mean_acc"), 4, 100, 1.0),
        ("1174.6 & 53.8", ("results", "NCI1", "T64", "full", "max_peak_mb"), 3, 1.0, 2.0),
        ("1174.6 & 53.8", ("results", "NCI1", "T64", "full", "mean_acc"), 4, 100, 1.0),
    ]),
    ("tab:positive_interval", os.path.join(DATA_ROOT, "arxiv_validation", "positive_interval.json"), "col", [
        ("0.2 &", ("cells", "rT_0.2", "unc_rel_err"), 2, 1.0, 0.02),
        ("1.0 &", ("cells", "rT_1.0", "unc_rel_err"), 2, 1.0, 0.02),
        ("2.8 &", ("cells", "rT_2.8", "unc_rel_err"), 2, 1.0, 0.02),
        ("0.2 &", ("cells", "rT_0.2", "amc_rel_err"), 3, 1.0, 0.02),
        ("1.0 &", ("cells", "rT_1.0", "amc_rel_err"), 3, 1.0, 0.02),
        ("2.8 &", ("cells", "rT_2.8", "amc_rel_err"), 3, 1.0, 0.02),
    ]),
]


def block(text, label):
    m = re.search(r"\\label\{%s\}.*?\\bottomrule" % re.escape(label), text, re.S)
    return m.group(0) if m else None


def json_get(d, keys):
    cur = d
    for k in keys:
        if k == "peak":
            return cur["peak_mem_mb"][0]
        cur = cur[k]
    return cur


def row_numbers(row, mode):
    if mode == "pct":
        return [float(x) for x in re.findall(r"([\d.]+)\\%", row)]
    return [float(x) for x in re.findall(r"[-+]?[\d.]+", row)]


def check_nospike(text, oks, fails):
    """Guard tab:nospike_scope against the consolidated expected-values JSON.

    The expected file (nospike_scope_expected.json, emitted by
    backfill_nospike.py) marks every cell present=false until the corresponding
    post-N1-fix run exists. A pending cell forces this check to stay RED so the
    paper cannot be declared consistent while the 5-shot rerun is outstanding.
    """
    label = "tab:nospike_scope"
    exp_path = os.path.join(_HERE, "nospike_scope_expected.json")
    if not os.path.exists(exp_path):
        print(f"\n=== {label}: expected JSON missing (run backfill_nospike.py) ===")
        fails.append(f"{label}: expected JSON missing")
        return
    expected = json.load(open(exp_path))
    blk = block(text, label)
    if blk is None:
        print(f"\n=== {label}: block not found in tex ===")
        fails.append(f"{label}: block not found")
        return
    print(f"\n=== {label} ===")

    # column layout per dataset row (col mode, percentage scale = 1.0):
    #  0 OMm 1 OMs 2 NSm 3 NSs 4 D1 5 p1 | 6 OM5m 7 OM5s 8 NS5m 9 NS5s 10 D5 11 p5
    layout = [
        ("1", [("om_mean", 0, 0.05), ("om_std", 1, 0.05),
               ("ns_mean", 2, 0.05), ("ns_std", 3, 0.05),
               ("delta", 4, 0.05), ("p", 5, 0.01)]),
        ("5", [("om_mean", 6, 0.05), ("om_std", 7, 0.05),
               ("ns_mean", 8, 0.05), ("ns_std", 9, 0.05),
               ("delta", 10, 0.05), ("p", 11, 0.01)]),
    ]
    for ds in ["MUTAG", "PROTEINS", "NCI1", "DD"]:
        row = next((l for l in blk.split("\n") if l.strip().startswith(ds + " &")), None)
        if row is None:
            fails.append(f"{label}/{ds}: row not found")
            continue
        # Drop the leading dataset-name token (e.g. "NCI1" ends in a digit that
        # the number regex would otherwise pick up as a spurious leading cell).
        cells = row.split("&", 1)[1] if "&" in row else row
        nums = [float(x) for x in re.findall(r"[-+]?[\d.]+", cells)]
        for shot, fields in layout:
            ent = expected.get(ds, {}).get(shot, {})
            if not ent.get("present", False):
                fails.append(f"{label}/{ds}/{shot}-shot: PENDING re-run "
                             f"(post-N1-fix No-Spikes missing)")
                print(f"  [!!] {ds:9s} {shot}-shot: PENDING re-run "
                      f"(post-N1-fix No-Spikes missing)")
                continue
            for field, col, tol in fields:
                if col >= len(nums):
                    fails.append(f"{label}/{ds}/{shot}/{field}: col {col} oob")
                    continue
                paper_val = nums[col]
                src = ent[field]
                if abs(paper_val - src) <= tol:
                    oks += 1
                    print(f"  [OK] {ds:9s} {shot}-shot {field:8s} "
                          f"paper={paper_val:8.3f}  exp={src:8.3f}")
                else:
                    fails.append(f"{label}/{ds}/{shot}/{field}: paper {paper_val} "
                                 f"vs exp {src}")
                    print(f"  [!!] {ds:9s} {shot}-shot {field:8s} "
                          f"paper={paper_val:8.3f}  exp={src:8.3f}")


def main():
    if not os.path.exists(TEX):
        print(f"MISSING: {TEX}")
        return 1
    text = open(TEX, encoding="utf-8").read()
    oks, fails = 0, []
    print("=" * 70)
    print("NUMBER TRACEABILITY CHECK (tables -> JSON artifacts)")
    print("=" * 70)

    for label, jp, mode, maps in TABLES:
        if not os.path.exists(jp):
            print(f"\n=== {label}: SOURCE MISSING {jp} ===")
            fails.append(f"{label}: source missing")
            continue
        blk = block(text, label)
        if blk is None:
            print(f"\n=== {label}: block not found in tex (add the table) ===")
            fails.append(f"{label}: block not found")
            continue
        d = json.load(open(jp))
        print(f"\n=== {label} ===")
        for substr, accessor, col, scale, tol in maps:
            row = next((l for l in blk.split("\n") if substr in re.sub(r"\s+", " ", l)), None)
            if row is None:
                fails.append(f"{label}: row '{substr}' not found")
                continue
            nums = row_numbers(row, mode)
            if col >= len(nums):
                fails.append(f"{label}/{substr}: col {col} oob (got {nums})")
                continue
            paper_val = nums[col]
            try:
                src = json_get(d, accessor) * scale
            except Exception as e:
                fails.append(f"{label}/{substr}: json access fail {e}")
                continue
            if abs(paper_val - src) <= tol:
                oks += 1
                print(f"  [OK] {substr[:34]:34s} paper={paper_val:8.3f}  json={src:8.3f}")
            else:
                fails.append(f"{label}/{substr}: paper {paper_val} vs json {src}")
                print(f"  [!!] {substr[:34]:34s} paper={paper_val:8.3f}  json={src:8.3f}")

    check_nospike(text, oks, fails)

    print("-" * 70)
    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"ALL {oks} NUMBERS MATCH THEIR SOURCE JSONS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
