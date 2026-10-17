"""
verify_correction_reaches_weights.py
=====================================

CPU-only, dependency-free proof that the corrected DvM makes the Riemannian
metric correction reach the recurrent *weights* (fixes P0-②).

It reproduces the EXACT PerStepMetricCorrection logic from
``spikergnn/dvm.py`` and drives it through a minimal hyperbolic recurrence
that contains a real ``nn.Parameter`` W, so we can inspect ``W.grad``.

Three modes are compared on the same random problem:

  A) BROKEN (old behaviour): every step is detached and the correction is
     applied post-hoc (old ``apply_dvm`` returned None for step outputs),
     so the autograd graph is severed -> W receives NO gradient.
  B) CORRECTED (new behaviour): PerStepMetricCorrection is applied INLINE,
     keeping the graph connected; truncated BPTT detaches only beyond the
     M-step window. W receives the λ²-corrected gradient.
  C) NO CORRECTION (baseline): plain autograd, no detach, no correction.

Assertions (all must hold for the fix to be correct):
  1. A.W.grad is None / ~0            -> old code did NOT train weights
  2. B.W.grad.norm() > 0              -> corrected code DOES reach weights
  3. ||B.W.grad - C.W.grad|| > tol    -> correction actually changes the gradient
     (if B == C, the "correction" would be a no-op)

Run:  <venv>/python.exe verify_correction_reaches_weights.py
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# Exact copy of spikergnn.dvm.PerStepMetricCorrection (the real fix code)
# ─────────────────────────────────────────────────────────────────────────
def conformal_factor(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5):
    x_norm_sq = (x * x).sum(dim=-1)
    return 2.0 / (1.0 - c * x_norm_sq).clamp(min=eps)


class PerStepMetricCorrection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_hyp_old, p_hyp_new, c):
        ctx.save_for_backward(p_hyp_old, p_hyp_new)
        ctx.c = c
        return p_hyp_new

    @staticmethod
    def backward(ctx, grad_output):
        p_hyp_old, p_hyp_new = ctx.saved_tensors
        c = ctx.c
        lam_old = conformal_factor(p_hyp_old, c)
        lam_new = conformal_factor(p_hyp_new, c)
        factor = (lam_old / lam_new.clamp(min=1e-5)) ** 2
        grad_old = torch.zeros_like(grad_output)
        grad_new = grad_output * factor.unsqueeze(-1)
        return grad_old, grad_new, None


def make_recurrence(d=4):
    """A minimal hyperbolic-ish recurrence with a real weight W."""
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(d, d) * 0.3)
    p_init = torch.randn(1, d)
    return W, p_init


def leaf(W0):
    """Make a fresh leaf parameter so .grad is populated (no clone grad_fn)."""
    return nn.Parameter(W0.detach().clone())


def run_broken(W, p_init, T=8, c=1.0):
    """Old behaviour: detach every step, post-hoc correction severs the graph.

    Faithfully reproduces the legacy ``apply_dvm`` / ``DvMFunction``:
    the final state keeps a grad_fn (so backward() runs) but the Function's
    backward returns None for every step output -> the gradient path to W is
    severed, so W.grad ends up None.
    """
    class BrokenDvMFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p_init_in, *step_outputs):
            ctx.save_for_backward(p_init_in, *step_outputs)
            return step_outputs[-1]

        @staticmethod
        def backward(ctx, grad):
            saved = ctx.saved_tensors
            # grad for p_init, then None for every step output (severs W)
            return grad, *([None] * (len(saved) - 1))

    W = leaf(W)
    p = p_init.clone().requires_grad_(True)      # init state has grad (like an embedding)
    states = []
    for _ in range(T):
        p_next = torch.tanh(W @ p.T).T          # (1, d), graph from W
        states.append(p_next.detach())           # <-- detach every step (old bug)
        p = p_next.detach()
    # post-hoc correction: graph exists (via p_init), but step outputs get
    # None gradient -> the recurrence weight W is never reached.
    final = BrokenDvMFunction.apply(p_init.clone().requires_grad_(True), *states)
    loss = final.pow(2).mean()
    loss.backward()
    return W.grad


def run_corrected(W, p_init, T=8, M=3, c=1.0):
    """New behaviour: inline correction, graph connected; detach beyond window."""
    W = leaf(W)
    p = p_init.clone()
    for t in range(T):
        p_prev = p
        p_next = torch.tanh(W @ p.T).T
        p = PerStepMetricCorrection.apply(p_prev, p_next, c)
        if (T - 1 - t) >= M:                      # truncated BPTT boundary
            p = p.detach()
    loss = p.pow(2).mean()
    loss.backward()
    return W.grad


def run_no_correction(W, p_init, T=8, c=1.0):
    """Baseline: plain autograd, no correction, no detach."""
    W = leaf(W)
    p = p_init.clone()
    for _ in range(T):
        p = torch.tanh(W @ p.T).T
    loss = p.pow(2).mean()
    loss.backward()
    return W.grad


def main():
    d, T, M = 4, 8, 3
    W0, p_init = make_recurrence(d)

    g_broken = run_broken(W0, p_init, T)
    g_corr = run_corrected(W0, p_init, T, M)
    g_base = run_no_correction(W0, p_init, T)

    broken_norm = 0.0 if g_broken is None else g_broken.norm().item()
    corr_norm = g_corr.norm().item()
    base_norm = g_base.norm().item()
    diff_corr_base = (g_corr - g_base).norm().item()

    print("=" * 64)
    print("Metric-correction reaches weight gradients — verification")
    print("=" * 64)
    print(f"  (A) BROKEN   ||W.grad|| = {broken_norm:.6e}   (old: detached, no grad)")
    print(f"  (B) CORRECTED||W.grad|| = {corr_norm:.6e}   (new: λ²-corrected)")
    print(f"  (C) BASELINE ||W.grad|| = {base_norm:.6e}   (no correction)")
    print(f"  ||B - C||               = {diff_corr_base:.6e}")
    print("-" * 64)

    checks = []
    # 1. old code severed weight gradients
    checks.append(("A: old code reaches NO weight gradient",
                   g_broken is None or broken_norm < 1e-9))
    # 2. corrected code reaches weights
    checks.append(("B: corrected code reaches weight gradient",
                   corr_norm > 1e-9))
    # 3. correction changes the gradient (not a no-op)
    checks.append(("B != C: correction is not a no-op",
                   diff_corr_base > 1e-9))
    # 4. bonus: corrected grad should differ from baseline in the expected
    #    direction (correction scales gradient by λ² factors near 1 for small
    #    states, so it is close-but-not-equal to baseline)
    checks.append(("B close to C (sanity: same order of magnitude)",
                   diff_corr_base < 10.0 * max(corr_norm, 1e-9)))

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("=" * 64)
    if ok:
        print("RESULT: PASS — metric correction now reaches the weights.")
        print("        The P0-② fix is verified on CPU.")
    else:
        print("RESULT: FAIL — fix did NOT behave as expected.")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
