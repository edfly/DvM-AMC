"""
bootstrap_stats.py -- P1-1 statistical-power honesty check (OFFLINE, no GPU).

Reads the existing per-(seed,fold) accuracy arrays (n=9 = 3 seeds x 3 folds)
that already back significance.json, and adds non-parametric bootstrap 95% CIs:
  * per-method mean accuracy CI on every dataset/method,
  * paired bootstrap CI of the mean difference for the flagged MUTAG contrasts.
This complements the BH-FDR Wilcoxon p-values already reported, and makes the
limited statistical power of n=9 explicit and quantitative (the reviewer's P1-1).

Exit 0 = produced bootstrap_stats.json.
"""
import os
import sys
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DVM_DATA_ROOT", _ROOT)

SIG = json.load(open(os.path.join(DATA_ROOT, "stats", "significance.json")))
PER_DS = SIG["per_dataset"]
HC = SIG.get("headline_comparisons", [])

# alias: significance.json method name -> key inside final_results_<DS>.json dict
_ALIAS = {
    "AMC M=8": "DvM M=8 truncated",
    "AMC M=4": "DvM M=4 truncated",
    "Full BPTT": "Full BPTT",
    "No Correction M=8": "No Correction M=8",
    "Full BPTT + MC": "Full BPTT + MC",
    "OM (SpikeRPGNN full)": "OM (SpikeRPGNN full)",
}


def get_accs(dataset, method_sig):
    src = PER_DS[dataset]["methods"][method_sig]["source"]
    p = os.path.join(DATA_ROOT, src) if not os.path.isabs(src) else src
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if isinstance(d, dict) and "accs" in d:          # rerun_outputs file
        return np.asarray(d["accs"], dtype=float)
    if isinstance(d, dict):                            # final_results_<DS>.json dict
        key = _ALIAS.get(method_sig, method_sig)
        if key in d and isinstance(d[key], dict) and "accs" in d[key]:
            return np.asarray(d[key]["accs"], dtype=float)
        # normalized fallback
        norm = method_sig.lower().replace(" ", "")
        for k, v in d.items():
            if isinstance(v, dict) and "accs" in v and k.lower().replace(" ", "") == norm:
                return np.asarray(v["accs"], dtype=float)
    return None


def boot_ci(x, n_boot=10000, seed=12345):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    out = {"per_method_ci": {}, "mutag_contrast_ci": []}
    for ds, block in PER_DS.items():
        out["per_method_ci"][ds] = {}
        for method, info in block["methods"].items():
            accs = get_accs(ds, method)
            if accs is None or len(accs) == 0:
                out["per_method_ci"][ds][method] = {"n": info.get("n"), "mean": info.get("mean"),
                                                    "ci95": None, "note": "accs not found"}
                continue
            lo, hi = boot_ci(accs)
            out["per_method_ci"][ds][method] = {
                "n": len(accs),
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs, ddof=1)),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
            print(f"  {ds}/{method}: mean={np.mean(accs):.4f} CI95=[{lo:.4f},{hi:.4f}] n={len(accs)}", flush=True)

    for hc in HC:
        if hc.get("dataset") != "MUTAG":
            continue
        A, B = hc["A"], hc["B"]
        a = get_accs("MUTAG", A)
        b = get_accs("MUTAG", B)
        if a is None or b is None or len(a) != len(b):
            out["mutag_contrast_ci"].append({**hc, "boot_diff_ci95": None, "note": "unpaired/missing"})
            continue
        diff = a - b
        lo, hi = boot_ci(diff)
        out["mutag_contrast_ci"].append({
            "A": A, "B": B,
            "n": int(hc["n"]),
            "diff_pp": hc.get("diff_pp"),
            "p_wilcoxon": hc.get("p_wilcoxon"),
            "p_bh_fdr": hc.get("p_bh_fdr"),
            "hedges_g": hc.get("hedges_g"),
            "boot_diff_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "verdict": hc.get("verdict"),
        })
        print(f"  contrast {A} vs {B}: diff={hc.get('diff_pp')}pp bootCI=[{lo*100:.2f},{hi*100:.2f}]pp p={hc.get('p_bh_fdr'):.4f}", flush=True)

    with open(os.path.join(DATA_ROOT, "bootstrap_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE bootstrap_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
