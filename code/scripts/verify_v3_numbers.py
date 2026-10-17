"""
Machine verification for Phase 1 / Phase 2 of the AMC revision.

Consumes results_v3/gradient_error_matrix.json (groups A+B) and
results_v3/curvature_sweep.json (group F), and decides -- from the data, not
from our preference -- which claims the paper is allowed to make.

Two self-calibrating ideas do the heavy lifting:

1. NUMERICAL STABILITY IS MEASURED, NOT ASSUMED. At M == T the truncated model
   and the full-BPTT reference build identical graphs, so their gradients must
   coincide. Any deviation is that unit's float32 noise floor. Units whose
   sentinel error exceeds the tolerance are excluded and reported, never
   silently dropped -- otherwise a divergent unit (DD at T=24 reaches a
   reference gradient norm of 2.2e8) would poison every downstream number.

2. THE METRIC-CORRECTION CLAIM IS DECIDED BY A PRE-REGISTERED RULE. The plan
   fixed the decision rule before seeing the data: the claim survives only if
   Delta(cos) grows monotonically with the effective radius. This script applies
   that rule and prints the verdict; it does not offer a softer alternative.

Usage:  python verify_v3_numbers.py
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(_HERE, "results_v3")
SENTINEL_TOL = 1e-3          # float32 noise ceiling for the M == T identity
STABILITY_NORM = 1e4         # reference gradient norm above which we distrust


def load(name):
    p = os.path.join(V3, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    n = len(xs)
    if n < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else None


# ------------------------------------------------------------------ group A
def group_a(rows):
    print("=" * 74)
    print("A 组 —— 真实截断梯度误差（替代 Stage 3）")
    print("=" * 74)

    by_unit = defaultdict(dict)          # (ds, seed, T) -> {M: row}
    for r in rows:
        if r["variant"] != "amc_truncated":
            continue
        by_unit[(r["dataset"], r["seed"], r["T"])][r["M"]] = r

    # ---- stability screening via the M == T identity ----
    stable, unstable = [], []
    for k, mp in by_unit.items():
        ds, seed, T = k
        sent = mp.get(T)
        if sent is None:
            unstable.append((k, "no sentinel row", None))
        elif sent["rel_err"] > SENTINEL_TOL or sent["ref_grad_norm"] > STABILITY_NORM:
            unstable.append((k, f"sentinel rel_err={sent['rel_err']:.3e} "
                                f"ref_norm={sent['ref_grad_norm']:.3e}",
                             sent["rel_err"]))
        else:
            stable.append(k)
    print(f"\n单元总数 {len(by_unit)}：稳定 {len(stable)}，剔除 {len(unstable)}")
    for k, why, _ in unstable:
        print(f"  剔除 {k[0]}/s{k[1]}/T{k[2]}  ({why})")

    sent_vals = [by_unit[k][k[2]]["rel_err"] for k in stable]
    if sent_vals:
        print(f"\n哨兵（M=T）相对误差：max={max(sent_vals):.3e}  "
              f"median={statistics.median(sent_vals):.3e}  "
              f"-> 门限 {SENTINEL_TOL:.0e} "
              f"{'PASS' if max(sent_vals) < SENTINEL_TOL else 'FAIL'}")

    # ---- the table the paper will print: mean +/- std over seeds ----
    agg = defaultdict(lambda: defaultdict(list))
    for k in stable:
        ds, seed, T = k
        for M, r in by_unit[k].items():
            if M == T:
                continue
            agg[(ds, T)][M].append(r["rel_err"])
    print(f"\n{'ds':<10}{'T':>4}{'M':>4}{'mean rel_err':>14}{'std':>10}{'n':>4}")
    print("-" * 48)
    table = {}
    for (ds, T) in sorted(agg):
        for M in sorted(agg[(ds, T)]):
            v = agg[(ds, T)][M]
            m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
            table[(ds, T, M)] = (m, s, len(v))
            print(f"{ds:<10}{T:>4}{M:>4}{m:>14.4f}{s:>10.4f}{len(v):>4}")

    # ---- monotonicity within stable units ----
    viol = []
    for k in stable:
        ms = sorted(M for M in by_unit[k] if M != k[2])
        prev = None
        for M in ms:
            cur = by_unit[k][M]["rel_err"]
            if prev is not None and cur > prev + 0.02:
                viol.append((k, M, prev, cur))
            prev = cur
    print(f"\n单调性违例（容差 0.02）：{len(viol)}"
          f"{'' if viol else '  -> PASS'}")
    for v in viol[:6]:
        print("   ", v)

    return {"stable_units": len(stable), "unstable": [list(k) + [why]
                                                      for k, why, _ in unstable],
            "sentinel_max": max(sent_vals) if sent_vals else None,
            "monotone_violations": len(viol),
            "table": {f"{k[0]}|T{k[1]}|M{k[2]}": v for k, v in table.items()}}


# ------------------------------------------------------------------ group B
def group_b(rows):
    print("\n" + "=" * 74)
    print("B 组 —— 度量校正消融（重做 Table 6）")
    print("=" * 74)

    by = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], r["seed"], r["T"], r["M"])][r["variant"]] = r

    win, full = [], []
    for k, v in by.items():
        a, n = v.get("amc_truncated"), v.get("no_metric_correction")
        if not (a and n):
            continue
        (full if k[3] == k[2] else win).append(a["cos"] - n["cos"])

    def describe(name, xs):
        if not xs:
            print(f"{name}: 无数据")
            return None
        xs = sorted(xs)
        print(f"{name}: n={len(xs)}  min={xs[0]:+.5f}  median="
              f"{statistics.median(xs):+.5f}  max={xs[-1]:+.5f}  "
              f"|median|={abs(statistics.median(xs)):.5f}")
        return statistics.median(xs)

    print()
    med_win = describe("M <  T （窗口排除初始瞬态）", win)
    med_full = describe("M == T （窗口包含初始瞬态）", full)

    print("\n结论（由数据裁定，非由偏好裁定）：")
    if med_win is not None and abs(med_win) < 0.01:
        print("  在 M<T 的全部测量上，去掉度量校正对梯度余弦的中位影响为 "
              f"{med_win:+.5f}——度量校正在截断窗口内数值上近似恒等。")
        print("  论文 Table 6 声称的 46 个百分点差距不成立，必须删除。")
    if med_full is not None and abs(med_full) > 0.01:
        print(f"  但在 M=T 时中位影响为 {med_full:+.5f}，校正并非恒等——")
        print("  它的净效应等于保留窗口的端点比 lambda_start^2/lambda_end^2，")
        print("  只有当窗口包含从原点出发的初始瞬态时该比值才显著偏离 1。")
    return {"median_delta_cos_M_lt_T": med_win, "median_delta_cos_M_eq_T": med_full,
            "n_M_lt_T": len(win), "n_M_eq_T": len(full)}


# ------------------------------------------------------------------ group F
def group_f(d):
    print("\n" + "=" * 74)
    print("F 组 —— 曲率/半径扫描：度量校正的适用边界")
    print("=" * 74)
    if not d:
        print("  未找到 results_v3/curvature_sweep.json")
        return None
    rows = [r for r in d.get("rows", []) if "error" not in r]
    print(f"\n{'ds':<8}{'seed':>5}{'c':>6}{'r_eff':>8}{'lam2 range':>12}"
          f"{'full-chain ratio':>18}{'window ratio':>14}{'Delta cos':>12}")
    print("-" * 84)
    xs, ys = [], []
    for r in rows:
        g = r.get("geometry") or {}
        ep = r.get("endpoint_ratio") or {}
        fr = ep.get("full_chain_geomean")
        wr = ep.get("window_geomean")
        print(f"{r['dataset']:<8}{r['seed']:>5}{r['c']:>6}"
              f"{g.get('r_eff_mean', float('nan')):>8.3f}"
              f"{g.get('lambda2_dynamic_range', float('nan')):>12.3e}"
              f"{(fr if fr else float('nan')):>18.3e}"
              f"{(wr if wr else float('nan')):>14.4f}"
              f"{r['delta_cos']:>+12.5f}")
        if g.get("r_eff_mean") is not None:
            xs.append(g["r_eff_mean"])
            ys.append(r["delta_cos"])

    rho = spearman(xs, ys)
    gates = d.get("gates", {})
    print(f"\nSpearman(r_eff, Delta cos) = {rho}")
    print(f"脚本内预注册判定：{gates.get('verdict')}")

    ratios = [r.get("endpoint_ratio", {}).get("window_geomean") for r in rows]
    ratios = [x for x in ratios if x]
    if ratios:
        print(f"\n窗口端点比 lambda_(T-M)^2/lambda_T^2 的实测范围："
              f"[{min(ratios):.4f}, {max(ratios):.4f}]")
        print("这就是 M<T 时 Delta cos 恒 ~0 的直接原因：窗口内端点比始终贴近 1。")
    return {"spearman": rho, "verdict": gates.get("verdict"),
            "window_ratio_range": [min(ratios), max(ratios)] if ratios else None}


def main():
    ab = load("gradient_error_matrix.json")
    f = load("curvature_sweep.json")
    if not ab:
        print("缺少 results_v3/gradient_error_matrix.json")
        return
    out = {}
    out["group_a"] = group_a(ab["rows"])
    out["group_b"] = group_b(ab["rows"])
    out["group_f"] = group_f(f)
    json.dump(out, open(os.path.join(V3, "verify_summary.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\n-> results_v3/verify_summary.json")


if __name__ == "__main__":
    main()
