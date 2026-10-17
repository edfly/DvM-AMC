"""
Differentiation-via-Manifold (DvM) — CORRECTED implementation.

Implements truncated BPTT with per-step Riemannian metric correction
matching the DvM paper's Theorem 1 and Algorithm 2.

Key corrections over the previous version:
1. Metric correction uses telescoping (λ₀²/λ_T² directly, not geometric mean)
2. Window-external gradient uses contraction decay ρ^(T-M) + endpoint correction
3. Supports use_metric_correction=False for ablation (No Correction baseline)
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from spikergnn.manifolds import ProductManifold, Point


def conformal_factor(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """λ_x = 2 / (1 - c||x||²). c MUST be the actual manifold curvature
    (never a hardcoded 1.0) so the metric correction is curvature-aware in
    both truncated and full_bptt_mc modes."""
    x_norm_sq = (x * x).sum(dim=-1)
    return 2.0 / (1.0 - c * x_norm_sq).clamp(min=eps)


class DvMFunction(torch.autograd.Function):
    """DvM with per-step telescoping metric correction.

    Forward: stores last M steps (or all if window >= T).
    Backward: applies metric correction λ_prev²/λ_curr² at each step
              within window; beyond window uses contraction decay +
              endpoint correction λ₀²/λ_{start}² (applied ONCE, not rooted).
    """

    @staticmethod
    def forward(ctx, p_init_hyp, p_init_euc, step_fn_outputs, conformal_factors,
                window, use_metric_correction, c):
        T = len(step_fn_outputs)
        start_idx = max(0, T - window)
        ctx.save_for_backward(
            p_init_hyp, p_init_euc,
            *[step_fn_outputs[i][0] for i in range(start_idx, T)],
            *[step_fn_outputs[i][1] for i in range(start_idx, T)],
            *[conformal_factors[i] for i in range(start_idx, T)],
        )
        ctx.window = window
        ctx.T = T
        ctx.start_idx = start_idx
        ctx.use_metric_correction = use_metric_correction
        ctx.c = c
        p_final_hyp, p_final_euc = step_fn_outputs[-1]
        return p_final_hyp, p_final_euc

    @staticmethod
    def backward(ctx, grad_p_final_hyp, grad_p_final_euc):
        saved = ctx.saved_tensors
        T = ctx.T
        start_idx = ctx.start_idx
        n_steps_in_window = T - start_idx
        use_mc = ctx.use_metric_correction
        c = ctx.c

        p_init_hyp = saved[0]
        p_init_euc = saved[1]
        hyp_states = saved[2:2 + n_steps_in_window]
        euc_states = saved[2 + n_steps_in_window:2 + 2 * n_steps_in_window]
        conf_factors = saved[2 + 2 * n_steps_in_window:2 + 3 * n_steps_in_window]

        grad_hyp = grad_p_final_hyp
        grad_euc = grad_p_final_euc

        # ── Within window: per-step metric correction ────────────
        if use_mc:
            for i in range(n_steps_in_window - 1, 0, -1):
                p_prev = hyp_states[i - 1]
                p_curr = hyp_states[i]
                lam_prev = conformal_factor(p_prev, c)
                lam_curr = conformal_factor(p_curr, c)
                factor = (lam_prev / lam_curr.clamp(min=1e-5)) ** 2
                grad_hyp = grad_hyp * factor.unsqueeze(-1)

        # ── Beyond window: contraction + endpoint correction ─────
        if start_idx > 0:
            # Contraction decay
            rho = 0.9
            decay = rho ** start_idx
            grad_hyp = grad_hyp * decay
            grad_euc = grad_euc * decay

            # Endpoint correction: λ₀²/λ_{start}² (ONCE, not rooted!)
            if use_mc:
                lam_0 = conformal_factor(p_init_hyp, c)
                lam_start = conformal_factor(hyp_states[0], c)
                endpoint_corr = (lam_0 / lam_start.clamp(min=1e-5)) ** 2
                grad_hyp = grad_hyp * endpoint_corr.unsqueeze(-1)

        return grad_hyp, grad_euc, None, None, None, None, None


def apply_dvm(
    manifold: ProductManifold,
    p_init: Point,
    step_fn_outputs: List[Point],
    window: int,
    use_metric_correction: bool = True,
    c: Optional[float] = None,
) -> Point:
    """Apply DvM with correct metric correction to STMP step sequence.

    c is the actual manifold curvature (default: read from `manifold.c`).
    Passing it explicitly here closes the long-standing gap where the
    truncated-mode correction silently used c=1.0 regardless of the real
    curvature (only full_bptt_mc mode was curvature-aware via
    PerStepMetricCorrection). For c=0 (dense Euclidean baseline) this yields
    λ=2 (constant) => identity correction, which is the correct Euclidean limit.
    """
    if c is None:
        c = manifold.c.item() if hasattr(manifold.c, "item") else manifold.c
    p_init_hyp, p_init_euc = p_init
    conformal_factors = []
    for p_hyp_t, _ in step_fn_outputs:
        cf = conformal_factor(p_hyp_t.detach(), c)
        conformal_factors.append(cf)

    p_final_hyp, p_final_euc = DvMFunction.apply(
        p_init_hyp, p_init_euc,
        step_fn_outputs, conformal_factors,
        window, use_metric_correction, c,
    )
    return (p_final_hyp, p_final_euc)


# ═══════════════════════════════════════════════════════════════
# Per-Step Metric Correction — for full_bptt_mc mode
# ═══════════════════════════════════════════════════════════════

class PerStepMetricCorrection(torch.autograd.Function):
    """Apply metric correction λ_p²/λ_{p_new}² at a SINGLE transition step.

    Forward:  returns p_new unchanged (pass-through).
    Backward: multiplies incoming gradient by λ_p²/λ_{p_new}².

    This is inserted BETWEEN state transitions in the STMP loop so that
    autograd computes (Dφ)^T naturally through the computation graph,
    while this function applies the Riemannian metric correction.

    Usage in STMP full_bptt_mc mode:
        p_new = rgc_and_lif(p_old)
        p_corrected = PerStepMetricCorrection.apply(p_old, p_new, c)
    """

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

        # KEY: apply metric correction to grad_new, zero grad_old.
        # grad_new flows through autograd: (Dφ)^T · (factor·grad_output)
        # grad_old is zeroed to prevent double-counting.
        # This gives the correct single-path gradient:
        #   total_grad = (λ_old²/λ_new²) · (Dφ)^T · grad_output
        grad_old = torch.zeros_like(grad_output)
        grad_new = grad_output * factor.unsqueeze(-1)
        return grad_old, grad_new, None  # p_hyp_old, p_hyp_new, c
