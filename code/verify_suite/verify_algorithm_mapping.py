"""Four-fold verification (4/4): ALGORITHM MAPPING — pseudocode vs implementation.

Each fingerprint asserts a mechanism the paper describes is actually present in
the code. Negative checks assert a documented design choice is NOT silently
changed. Adapted from papersh verify_algorithm_mapping_template.

Mechanism under test (the analytic metric correction, Sec. 4):
  * conformal_factor(x, c) defined and used to re-anchor gradients;
  * truncated BPTT window: start_idx = max(0, T - window);
  * telescoping re-anchoring in the backward pass (lam_prev/lam_curr);
  * rho**(start_idx) forgetting term;
  * PerStepMetricCorrection applied per step in stmp.py.
Negative: the SpikeRPGNN message passing is Euclidean GIN — it must NOT call a
generic expmap( / logmap( (only expmap0/logmap0 at the readout pooling are
allowed). Changing this would be the classic mechanism drift.

Exit 0 = all steps match.  Run the injection test (bottom) after editing.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(os.path.dirname(_HERE), "spikergnn")

DVM = os.path.join(PROG, "dvm.py")
STMP = os.path.join(PROG, "stmp.py")

# (pseudocode step, source file, fingerprint regex, note)
CHECKS = [
    ("conformal factor defined", DVM, r"def conformal_factor", "lambda = 2/(1-c|x|^2)"),
    ("truncation window start_idx", DVM,
     r"start_idx\s*=\s*max\(0,\s*T\s*-\s*window\)", "truncated BPTT window"),
    ("telescoping re-anchor (prev)", DVM,
     r"lam_prev\s*=\s*conformal_factor\(p_prev(?:,\s*\w+)?\)", "pullback step t-1"),
    ("telescoping re-anchor (curr)", DVM,
     r"lam_curr\s*=\s*conformal_factor\(p_curr(?:,\s*\w+)?\)", "pullback step t"),
    ("rho forgetting term", DVM,
     r"decay\s*=\s*rho\s*\*\*\s*start_idx", "C(1-rho^(T-M)) bound"),
    ("per-step metric correction applied", STMP,
     r"PerStepMetricCorrection\.apply", "correction in the recurrent loop"),
    ("Riemannian geodesic message passing (RGC)", STMP,
     r"RiemannianGraphConv", "geodesic message passing, not Euclidean GIN"),
    # P0-B: true O(M) byte memory — window-exterior steps detached from graph
    ("P0-B truncation-window branch", STMP,
     r"_window_detach and t < start_idx", "window-exterior steps gated by start_idx"),
    ("P0-B window-exterior no_grad", STMP,
     r"with torch\.no_grad\(\)", "true O(M) bytes via graph detach (not just stored states)"),
]

# (label, file, anchor regex, forbidden pattern, note)
NEGATIVE_CHECKS = [
    ("readout pooling is Riemannian (expmap0, not generic expmap)",
     STMP, r"hyp_stack", r"expmap\(",
     "only expmap0 at readout pooling is allowed; a generic expmap( would "
     "contradict the documented Poincare readout"),
    ("message passer is not a bare Euclidean GINConv",
     STMP, r"RiemannianGraphConv", r"GINConv",
     "the Riemannian claim is satisfied by RGC, not torch_geometric GINConv"),
]


def strip_comments(src):
    src = re.sub(r"^\s*\"\"\"[\s\S]*?\"\"\"", "", src, count=1)
    return "\n".join(l.split("#", 1)[0] for l in src.split("\n"))


def main():
    srcs = {}
    for f in (DVM, STMP):
        if not os.path.exists(f):
            print(f"MISSING: {f}")
            return 1
        srcs[f] = strip_comments(open(f, encoding="utf-8").read())

    print("=" * 72)
    print("PSEUDOCODE <-> IMPLEMENTATION MAPPING CHECK")
    print("=" * 72)
    fails = []
    for label, fpath, pattern, note in CHECKS:
        ok = re.search(pattern, srcs[fpath]) is not None
        print(f"{label[:46]:46s} {'OK' if ok else '**FAIL**':>9s}  {os.path.basename(fpath)}")
        if not ok:
            fails.append((label, fpath, pattern, note))

    print("-" * 72)
    for label, fpath, anchor, forbidden, note in NEGATIVE_CHECKS:
        hit = re.search(anchor, srcs[fpath])
        if not hit:
            print(f"{label[:46]:46s} {'**FAIL**':>9s}  (anchor not found)")
            fails.append((label, fpath, anchor, note))
            continue
        seg = srcs[fpath][hit.start():hit.start() + 400]
        bad = re.search(forbidden, seg)
        ok = bad is None
        print(f"{label[:46]:46s} {'OK' if ok else '**FAIL**':>9s}  {os.path.basename(fpath)}")
        if not ok:
            fails.append((label, fpath, forbidden, note))

    print("-" * 72)
    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for label, fpath, pattern, note in fails:
            print(f"  - {label}\n      {fpath}: expected {pattern}\n      note: {note}")
        print("\nAction: update the pseudocode, or change the code and re-run.")
        return 1
    print(f"ALL {len(CHECKS) + len(NEGATIVE_CHECKS)} STEPS MATCH.")
    return 0


# ── INJECTION TEST ────────────────────────────────────────────────────────
# After adapting, confirm the harness can actually fail:
#   python -c "import verify_algorithm_mapping as v; \
#       v.CHECKS[0]=(v.CHECKS[0][0], v.DVM, r'ZZZ_ABSENT_ZZZ', v.CHECKS[0][3]); \
#       print('expect exit 1 ->', v.main())"
# Expect 1.
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
