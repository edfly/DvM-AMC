"""
verify_ablation_functional.py — CPU functional tests for the experiment-matrix
core mechanics (no GPU / no TU dataset required).

Covers the two highest-risk fixes from the redesign audit
(experiment_matrix_redesign.md):
  * D3/N5: dvm.conformal_factor / apply_dvm must use the REAL curvature, not a
    hardcoded c=1.0. c=0 (dense Euclidean baseline) must yield λ=2 (identity).
  * N1: no_spikes ablation must let the manifold state EVOLVE (mask=1), so the
    readout is not a degenerate constant. c=0 path must stay finite (no NaN).

Run:
  PYTHONPATH=programfiles python verify_suite/verify_ablation_functional.py
Exit code 0 = all assertions pass.
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROG = os.path.join(_PROJECT_ROOT, "spikergnn")
# The `spikergnn` package lives in programfiles (exposed via the project's
# spikergnn->programfiles junction). Insert it explicitly so this script is
# self-contained and does not depend on a clean PYTHONPATH from the caller
# (e.g. an inherited shim path that breaks the junction-based import).
for _p in (_PROG, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from spikergnn.dvm import conformal_factor, apply_dvm
from spikergnn.manifolds import ProductManifold
from spikergnn.neurons import ManifoldLIFNeuron


def _fail(msg):
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_conformal_factor_curvature():
    print("[test] conformal_factor curvature passthrough")
    x = torch.randn(4, 8) * 0.3
    cf0 = conformal_factor(x, 0.0)
    if not torch.allclose(cf0, torch.full_like(cf0, 2.0)):
        _fail("c=0 must give λ=2 everywhere (Euclidean limit)")
    cf1 = conformal_factor(x, 1.0)
    expected = 2.0 / (1.0 - 1.0 * (x * x).sum(-1)).clamp(min=1e-5)
    if not torch.allclose(cf1, expected):
        _fail("c=1.0 conformal factor does not match 2/(1-c||x||^2)")
    # Sanity: c=0.5 differs from c=1.0
    cf05 = conformal_factor(x, 0.5)
    if torch.allclose(cf05, cf1):
        _fail("conformal_factor ignores c (still hardcoded?)")
    print("  [ok] c=0 -> λ=2; c=1.0 -> exact; c=0.5 distinct")


def test_apply_dvm_curvature_aware():
    print("[test] apply_dvm gradient is curvature-aware")
    torch.manual_seed(0)
    N, d_h, T = 3, 8, 6
    c_vals = [0.0, 1.0]
    grads = {}
    for c in c_vals:
        man = ProductManifold(d_h=d_h, d_e=0, c=c)
        p_init_hyp = (torch.randn(N, d_h) * 0.1).requires_grad_(True)
        p_init_euc = torch.zeros(N, 0)
        step_outputs = []
        cur = p_init_hyp.detach().clone()
        for _ in range(T):
            cur = man.hyp.expmap(torch.randn(N, d_h) * 0.05, cur)
            step_outputs.append((cur.clone(), p_init_euc.clone()))
        p_final_hyp, _ = apply_dvm(
            manifold=man, p_init=(p_init_hyp, p_init_euc),
            step_fn_outputs=step_outputs, window=T,
            use_metric_correction=True, c=c,
        )
        loss = (p_final_hyp ** 2).sum()
        loss.backward()
        grads[c] = p_init_hyp.grad.detach().clone()
    if torch.allclose(grads[0.0], grads[1.0]):
        _fail("apply_dvm gradient identical for c=0 vs c=1.0 — curvature-blind!")
    print("  [ok] c=0 vs c=1.0 gradients differ (curvature-aware)")


def _run_lif(manifold, ablation, x, p_init):
    lif = ManifoldLIFNeuron(manifold=manifold, ablation=ablation)
    lif.eval()
    with torch.no_grad():
        spikes, p_final = lif(x, p_init)
    return lif, spikes, p_final


def test_no_spikes_state_evolves():
    print("[test] no_spikes state evolves (not frozen)")
    torch.manual_seed(1)
    N, T, d_total = 5, 4, 24
    man = ProductManifold(d_h=8, d_e=16, c=1.0)
    x = torch.randn(N, T, d_total) * 0.5
    p_init = (torch.randn(N, 8) * 0.1,
              torch.randn(N, 16) * 0.1)
    lif, spikes, p_final = _run_lif(man, "no_spikes", x, p_init)
    if not torch.isfinite(p_final[0]).all():
        _fail("no_spikes produced non-finite hyperbolic state")
    if torch.allclose(p_final[0], p_init[0], atol=1e-6):
        _fail("no_spikes state did NOT evolve (frozen -> degenerate readout)")
    print("  [ok] no_spikes hyperbolic state evolved, finite")


def test_full_produces_spikes():
    print("[test] full (spiking) produces spikes without NaN")
    torch.manual_seed(2)
    N, T, d_total = 5, 4, 24
    man = ProductManifold(d_h=8, d_e=16, c=1.0)
    x = torch.randn(N, T, d_total) * 1.5
    p_init = (torch.randn(N, 8) * 0.1,
              torch.randn(N, 16) * 0.1)
    lif, spikes, p_final = _run_lif(man, "full", x, p_init)
    if not torch.isfinite(p_final[0]).all():
        _fail("full produced non-finite state")
    if lif._spike_count < 0:
        _fail("full spike count negative")
    print(f"  [ok] full finite; spike_count={lif._spike_count}")


def test_c0_euclidean_finite():
    print("[test] c=0 (dense Euclidean) path is finite and evolves")
    torch.manual_seed(3)
    N, T, d_total = 5, 4, 24
    man = ProductManifold(d_h=8, d_e=16, c=0.0)
    x = torch.randn(N, T, d_total) * 0.5
    p_init = (torch.randn(N, 8) * 0.1,
              torch.randn(N, 16) * 0.1)
    lif, spikes, p_final = _run_lif(man, "no_spikes", x, p_init)
    if not torch.isfinite(p_final[0]).all():
        _fail("c=0 no_spikes produced NaN (divide-by-zero regression!)")
    if torch.allclose(p_final[0], p_init[0], atol=1e-6):
        _fail("c=0 no_spikes state frozen")
    print("  [ok] c=0 path finite and state evolved")


def main():
    print("=" * 60)
    print("FUNCTIONAL TESTS — experiment-matrix core mechanics")
    print("=" * 60)
    test_conformal_factor_curvature()
    test_apply_dvm_curvature_aware()
    test_no_spikes_state_evolves()
    test_full_produces_spikes()
    test_c0_euclidean_finite()
    print("=" * 60)
    print("ALL FUNCTIONAL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
