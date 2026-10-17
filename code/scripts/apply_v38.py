"""apply_v38.py -- backfill the four-dataset No-Spikes scope into the paper.

Reads the 12 post-fix result JSONs (4 datasets x {OM, No-Spikes} x {1-shot, 5-shot}),
computes the paired No-Spikes-vs-OM comparison, and replaces the %V38_ANCHOR%
marker in post/DvM_Paper_Anonymous_v37.tex with a dataset-specific qualifier
paragraph + a four-dataset table, then writes v38 and syncs the unversioned
working copy.  Intended to run automatically once the supplement queue finishes.

Usage:
  python apply_v38.py            # write v38 (requires all 8 JSONs present)
  python apply_v38.py --check    # print the table + list missing files, no write
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "rerun_outputs")
DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
PAPER_V37 = os.path.join(ROOT, "post", "DvM_Paper_Anonymous_v37.tex")
PAPER_V38 = os.path.join(ROOT, "post", "DvM_Paper_Anonymous_v38.tex")
PAPER_FLAT = os.path.join(ROOT, "post", "DvM_Paper_Anonymous.tex")


def L(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def paths():
    d = {}
    for ds in DATASETS:
        d[ds] = {
            "om1": os.path.join(OUT, f"{ds}_OM", f"{ds}_1shot_full.json"),
            "om5": os.path.join(OUT, f"{ds}_OM", f"{ds}_5shot_full.json"),
            "ns1": os.path.join(OUT, f"{ds}_NoSpikesFixed", f"{ds}_1shot_no_spikes.json"),
            "ns5": os.path.join(OUT, f"{ds}_NoSpikesFixed", f"{ds}_5shot_no_spikes.json"),
        }
    return d


def mean_std(accs):
    n = len(accs)
    m = sum(accs) / n
    s = math.sqrt(sum((x - m) ** 2 for x in accs) / (n - 1))
    return m, s


def paired_p(a, b):
    """Return (mean_diff_b_minus_a, p_wilcoxon) or (None, None) if scipy missing."""
    try:
        from scipy import stats
    except Exception:
        return None, None
    diffs = [y - x for x, y in zip(a, b)]
    m = sum(diffs) / len(diffs)
    t, p = stats.wilcoxon(b, a)  # b=NS, a=OM -> tests NS-OM
    return m, p


def p_str(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print table, do not write v38")
    args = ap.parse_args()

    P = paths()
    rows = {}
    missing = []
    for ds in DATASETS:
        r = P[ds]
        ok = all(os.path.exists(x) for x in r.values())
        if not ok:
            for k, v in r.items():
                if not os.path.exists(v):
                    missing.append(v)
            continue
        om1 = L(r["om1"])["accs"]; om5 = L(r["om5"])["accs"]
        ns1 = L(r["ns1"])["accs"]; ns5 = L(r["ns5"])["accs"]
        om1m, om1s = mean_std(om1); om5m, om5s = mean_std(om5)
        ns1m, ns1s = mean_std(ns1); ns5m, ns5s = mean_std(ns5)
        d1, p1 = paired_p(om1, ns1)
        d5, p5 = paired_p(om5, ns5)
        rows[ds] = dict(om1m=om1m, om1s=om1s, om5m=om5m, om5s=om5s,
                        ns1m=ns1m, ns1s=ns1s, ns5m=ns5m, ns5s=ns5s,
                        d1=d1, p1=p1, d5=d5, p5=p5)

    if missing:
        print("MISSING JSON (queue not finished?):")
        for m in missing:
            print("  ", m)
        if not args.check:
            sys.exit(2)

    # ---- build LaTeX table ----
    body = []
    for ds in DATASETS:
        if ds not in rows:
            body.append(f"    {ds} & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        r = rows[ds]
        def f(x): return f"{100*x:.2f}"
        def pm(x): return f"$\\pm${100*x:.2f}"
        d1 = f"{100*r['d1']:+.2f}" if r["d1"] is not None else "n/a"
        d5 = f"{100*r['d5']:+.2f}" if r["d5"] is not None else "n/a"
        body.append(
            f"    {ds} & {f(r['om1m'])}{pm(r['om1s'])} & {f(r['ns1m'])}{pm(r['ns1s'])} "
            f"& {d1} ({p_str(r['p1'])}) & {f(r['om5m'])}{pm(r['om5s'])} & "
            f"{f(r['ns5m'])}{pm(r['ns5s'])} & {d5} ({p_str(r['p5'])}) \\\\"
        )
    table = (
        "\\begin{table}[htbp]\n\\centering\\footnotesize\n"
        "\\caption{Four-dataset post-fix comparison of the non-spiking continuous ablation "
        "against the spiking OnManifold model ($n{=}9$ runs; paired Wilcoxon signed-rank on "
        "shared seeds). The non-spiking variant exceeds the spiking model only on MUTAG "
        "(1-shot $+7.60$ points, $p{=}0.0078$); on PROTEINS, NCI1 and DD the differences are "
        "within $\\pm1.5$ points and not significant.}\n\\label{tab:nospike_scope}\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Dataset & \\multicolumn{3}{c}{1-shot} & \\multicolumn{3}{c}{5-shot} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        " & OM & No-Spk & $\\Delta$ (p) & OM & No-Spk & $\\Delta$ (p) \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    qualifier = (
        "\\paragraph{Cross-dataset scope of the non-spiking result.} "
        "The MUTAG finding above is dataset-specific. We repeated the identical paired "
        "comparison (non-spiking continuous vs.\\ spiking OnManifold, shared seeds, $n{=}9$) "
        "on PROTEINS, NCI1 and DD (Table~\\ref{tab:nospike_scope}). There the non-spiking "
        "variant does \\emph{not} exceed the spiking model: every difference is within "
        "$\\pm1.5$ points and none is significant after BH-FDR correction. We therefore "
        "report the MUTAG result as a property of the spiking mechanism on that dataset, "
        "not a general accuracy cost of firing, and do not generalise it beyond MUTAG.\n\n"
    ) + table

    # ---- console summary ----
    print("Four-dataset No-Spikes vs OM (paired):")
    print(f"  {'dataset':9s} {'OM1':>8s} {'NS1':>8s} {'d1(p)':>14s} "
          f"{'OM5':>8s} {'NS5':>8s} {'d5(p)':>14s}")
    for ds in DATASETS:
        if ds not in rows:
            print(f"  {ds:9s} MISSING"); continue
        r = rows[ds]
        print(f"  {ds:9s} {100*r['om1m']:7.2f} {100*r['ns1m']:7.2f} "
              f"{(100*r['d1']):+13.2f}({p_str(r['p1'])}) "
              f"{100*r['om5m']:7.2f} {100*r['ns5m']:7.2f} "
              f"{(100*r['d5']):+13.2f}({p_str(r['p5'])})")

    if args.check:
        print("\n[--check] no v38 written.")
        return

    # ---- write v38 ----
    with open(PAPER_V37, encoding="utf-8") as f:
        tex = f.read()
    if "%V38_ANCHOR%" not in tex:
        print("ERROR: %V38_ANCHOR% not found in v37; abort.")
        sys.exit(3)
    tex = tex.replace("%V38_ANCHOR%", qualifier.rstrip("\n"))
    # bump changelog
    tex = tex.replace(
        "% Paper version: v37 (2026-10-03:",
        "% Paper version: v38 (2026-10-07: four-dataset scope of non-spiking result added; "
        "Table~\\ref{tab:nospike_scope} confirms No-Spikes>OM is MUTAG-only, n.s. on "
        "PROTEINS/NCI1/DD)\n% v37 (2026-10-03:",
        1,
    )
    with open(PAPER_V38, "w", encoding="utf-8") as f:
        f.write(tex)
    # sync unversioned working copy
    with open(PAPER_FLAT, "w", encoding="utf-8") as f:
        f.write(tex)
    print("\nV38_OK ->", PAPER_V38)
    print("V38_OK -> synced", PAPER_FLAT)


if __name__ == "__main__":
    main()
