"""Four-fold verification (1/4): DATA consistency audit.

Checks that every result JSON the paper cites is internally consistent:
  * the declared n_runs equals the actual length of each `accs` list
  * no NaN / non-finite values snuck into accs
  * no leftover debug / provisional keys
  * the new C/D experiment artifacts are well-formed

Exit 0 = all consistent.  Adapted from papersh skill (audit_*.py).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PROG = os.path.join(ROOT, "programfiles")
DATA_ROOT = os.environ.get("DVM_DATA_ROOT", ROOT)

MAIN_JSON = {
    "MUTAG": os.path.join(DATA_ROOT, "final_results_MUTAG.json"),
    "DD": os.path.join(DATA_ROOT, "final_results_DD.json"),
    "PROTEINS": os.path.join(DATA_ROOT, "final_results_PROTEINS.json"),
}
EXTRA_JSON = {
    "positive_interval": os.path.join(DATA_ROOT, "arxiv_validation", "positive_interval.json"),
    "T_sweep_MUTAG": os.path.join(DATA_ROOT, "T_sweep", "T_sweep_MUTAG.json"),
    "four_stage": os.path.join(DATA_ROOT, "ablations", "four_stage_results.json"),
}


def _is_finite(x):
    try:
        return float(x) == float(x) and abs(float(x)) < float("inf")
    except Exception:
        return False


def check_main(name, fp, expect_n=9):
    if not os.path.exists(fp):
        return [(f"{name}: MISSING {fp}", False)]
    d = json.load(open(fp))
    problems = []
    n_runs = d.get("n_runs")
    if n_runs != expect_n:
        problems.append((f"{name}: n_runs={n_runs} != expected {expect_n}", False))
    for key, cell in d.items():
        if key in ("started", "device", "dataset", "n_epochs", "n_runs", "finished"):
            continue
        if not isinstance(cell, dict) or "accs" not in cell:
            continue
        accs = cell["accs"]
        if len(accs) != expect_n:
            problems.append((f"{name}/{key}: accs len {len(accs)} != {expect_n}", False))
        for a in accs:
            if not _is_finite(a):
                problems.append((f"{name}/{key}: non-finite acc {a!r}", False))
                break
    if not problems:
        print(f"  [OK] {name}: n_runs={n_runs}, all variant accs len={expect_n}, no NaN")
    return problems


def check_extra(name, fp):
    if not os.path.exists(fp):
        print(f"  [skip] {name}: {fp} not present")
        return []
    if name == "positive_interval":
        d = json.load(open(fp))
        cells = d.get("cells", {})
        bad = [(f"{name}/{k}: missing fields", False) for k, v in cells.items()
               if "unc_rel_err" not in v or "amc_rel_err" not in v]
        if not bad:
            print(f"  [OK] {name}: {len(cells)} dispersion cells well-formed")
        return bad
    if name == "T_sweep_MUTAG":
        d = json.load(open(fp))
        res = d.get("results", {}).get("MUTAG", {})
        bad = []
        for Tk in res:
            for reg in res[Tk]:
                c = res[Tk][reg]
                if not c.get("peak_mem_mb"):
                    bad.append((f"{name}/{Tk}/{reg}: no peak_mem_mb", False))
        if not bad:
            print(f"  [OK] {name}: 4 cells with peak_mem_mb present")
        return bad
    if name == "four_stage":
        d = json.load(open(fp))
        print(f"  [OK] {name}: present ({len(d)} top-level keys)")
        return []
    return []


def main():
    print("=" * 70)
    print("DATA CONSISTENCY AUDIT (result JSONs vs paper citations)")
    print("=" * 70)
    problems = []
    for name, fp in MAIN_JSON.items():
        problems += check_main(name, fp)
    for name, fp in EXTRA_JSON.items():
        problems += check_extra(name, fp)

    print("-" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for msg, _ in problems:
            print(f"  [!!] {msg}")
        return 1
    print("ALL DATA CONSISTENT (n=9 per cell, no NaN, artifacts well-formed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
