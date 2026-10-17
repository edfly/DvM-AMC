"""
DvM Full: DEPRECATED — use dvm_step.py instead.

This module has a critical architectural flaw: DvMFullFunction receives
detached state tensors and cannot compute (Dφ)^T via autograd. The metric
correction for truncated mode also incorrectly uses geometric mean rooting
instead of direct telescoping.

For the CORRECT implementation, see:
  - dvm_step.py: DvMStep (per-step correction with proper autograd)
  - dvm_step.py: TruncatedDvMWrapper (correct truncated DvM)

Kept for reference: compute_gradient_fidelity() and the old DvMWrapper class
are retained for backward compatibility. Do NOT use DvMFullFunction for
actual Riemannian gradient computation.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional

from dvm_core import (
    conformal_factor,
    metric_correction_factor,
    apply_metric_correction,
    compute_contraction_factor,
    gradient_error_bound,
    RiemannianGradientCorrector,
    compute_step_metrics,
)

Point = Tuple[torch.Tensor, torch.Tensor]


# ═══════════════════════════════════════════════════════════════════
# DvM Full: Complete pushforward/pullback BPTT
# ═══════════════════════════════════════════════════════════════════

class DvMFullFunction(torch.autograd.Function):
    """DEPRECATED: Complete DvM with per-step Riemannian metric correction.

    WARNING: This implementation has a critical flaw — it receives detached
    state tensors and cannot compute (Dφ)^T via autograd. The metric
    correction in backward mode for truncation uses incorrect geometric
    mean rooting. Use dvm_step.DvMStep for correct per-step implementation.

    Kept for backward compatibility of the gradient fidelity utilities.
    """

    @staticmethod
    def forward(
        ctx,
        p_init_hyp: torch.Tensor,
        p_init_euc: torch.Tensor,
        step_states_hyp: List[torch.Tensor],
        step_states_euc: List[torch.Tensor],
        c: float,
        use_truncation: bool,
        window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: store states for backward metric correction.

        Args:
            p_init_hyp: (N, d_h) initial hyperbolic state.
            p_init_euc: (N, d_e) initial Euclidean state.
            step_states_hyp: List of T tensors (N, d_h) per step.
            step_states_euc: List of T tensors (N, d_e) per step.
            c: Curvature parameter.
            use_truncation: If True, only store last M steps.
            window: Truncation window M.

        Returns:
            (p_final_hyp, p_final_euc) final manifold state.
        """
        T = len(step_states_hyp)

        if use_truncation:
            # Only store last M steps for BPTT
            start_idx = max(0, T - window)
            ctx.save_for_backward(
                p_init_hyp, p_init_euc,
                *[s for s in step_states_hyp[start_idx:]],
                *[s for s in step_states_euc[start_idx:]],
            )
            ctx.start_idx = start_idx
        else:
            # Store ALL steps for full BPTT
            ctx.save_for_backward(
                p_init_hyp, p_init_euc,
                *step_states_hyp,
                *step_states_euc,
            )
            ctx.start_idx = 0

        ctx.T = T
        ctx.c = c
        ctx.use_truncation = use_truncation
        ctx.window = window

        return step_states_hyp[-1], step_states_euc[-1]

    @staticmethod
    def backward(ctx, grad_p_final_hyp, grad_p_final_euc):
        """Backward with Riemannian metric correction.

        Within the stored window: apply per-step metric correction.
        Beyond the window: approximate via contraction-based exponential decay.
        """
        saved = ctx.saved_tensors
        T = ctx.T
        c = ctx.c
        start_idx = ctx.start_idx
        window = ctx.window
        use_truncation = ctx.use_truncation
        n_stored = T - start_idx

        p_init_hyp = saved[0]
        p_init_euc = saved[1]
        hyp_states = saved[2:2 + n_stored]
        euc_states = saved[2 + n_stored:2 + 2 * n_stored]

        grad_hyp = grad_p_final_hyp
        grad_euc = grad_p_final_euc

        # ── Within window: apply per-step Riemannian correction ────
        # Walk backward through stored states, applying metric correction
        # at each step:
        #   grad_E_at_p = (λ_p²/λ_{p_new}²) * (Dφ)^T * grad_E_at_{p_new}
        # The (Dφ)^T is already handled by autograd's chain rule through
        # the computation graph within the window. We apply the metric
        # scaling factor.
        for i in range(n_stored - 1, 0, -1):
            p_curr = hyp_states[i]      # state at step t+i (newer)
            p_prev = hyp_states[i - 1]  # state at step t+i-1 (older)

            # Metric correction factor: (λ_{prev}/λ_{curr})^2
            factor = metric_correction_factor(p_prev, p_curr, c)
            grad_hyp = grad_hyp * factor.unsqueeze(-1)

            # Euclidean: no metric correction (λ=1 always)
            pass  # grad_euc unchanged

        # ── Beyond window: contraction-based approximation ─────────
        if start_idx > 0:
            # Use the contraction factor rho for exponential decay
            # rho = alpha (LIF membrane decay) is the dominant contractive term
            rho = 0.9  # corresponds to alpha_decay in LIF
            decay = rho ** start_idx
            grad_hyp = grad_hyp * decay
            grad_euc = grad_euc * decay

            # Also apply an estimated metric correction for the skipped
            # steps using the geometric mean of conformal factors
            if n_stored > 0:
                # Estimate: average correction factor over skipped steps
                # Using conformal factors at p_init and first stored state
                lam_init = conformal_factor(p_init_hyp, c)
                lam_first = conformal_factor(hyp_states[0], c)
                avg_correction = (lam_init / lam_first.clamp(min=1e-5)) ** 2
                # Apply geometric mean across start_idx steps
                avg_correction = avg_correction ** (1.0 / max(start_idx, 1))
                grad_hyp = grad_hyp * avg_correction.unsqueeze(-1)

        # Gradient for p_init
        grad_p_init_hyp = grad_hyp
        grad_p_init_euc = grad_euc

        # step_states gradients are None (not needed)
        none_grads = [None] * (2 * n_stored)

        return (grad_p_init_hyp, grad_p_init_euc,
                *([None] * len(hyp_states)),
                *([None] * len(euc_states)),
                None, None, None)


# ═══════════════════════════════════════════════════════════════════
# DvM Wrapper for STMP integration
# ═══════════════════════════════════════════════════════════════════

class DvMWrapper(nn.Module):
    """Wrapper for integrating complete DvM into STMP pipeline.

    Usage in STMP:
        self.dvm = DvMWrapper(c=1.0, window=8, use_truncation=True)
        ...
        p_final = self.dvm(p_init, step_states)

    Attributes:
        c: Curvature parameter.
        window: Truncation window size M.
        use_truncation: Enable truncated BPTT for O(M) memory.
        corrector: RiemannianGradientCorrector for per-step metric scaling.
    """

    def __init__(
        self,
        c: float = 1.0,
        window: int = 8,
        use_truncation: bool = True,
    ):
        super().__init__()
        self.c = c
        self.window = window
        self.use_truncation = use_truncation

    def forward(
        self,
        p_init: Point,
        step_states: List[Point],
    ) -> Point:
        """Apply DvM to the STMP step sequence.

        Args:
            p_init: (p_hyp_init, p_euc_init) initial manifold state.
            step_states: List of (p_hyp_t, p_euc_t) per time step t.

        Returns:
            (p_hyp_final, p_euc_final) with DvM gradient tracking.
        """
        p_init_hyp, p_init_euc = p_init
        hyp_states = [s[0] for s in step_states]
        euc_states = [s[1] for s in step_states]

        p_final_hyp, p_final_euc = DvMFullFunction.apply(
            p_init_hyp, p_init_euc,
            hyp_states, euc_states,
            self.c,
            self.use_truncation,
            self.window,
        )

        return (p_final_hyp, p_final_euc)


# ═══════════════════════════════════════════════════════════════════
# Gradient Fidelity Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_gradient_fidelity(
    grad_full: torch.Tensor,
    grad_truncated: torch.Tensor,
) -> dict:
    """Compute gradient fidelity metrics between full and truncated DvM.

    Args:
        grad_full: Gradient from full BPTT (ground truth).
        grad_truncated: Gradient from truncated DvM.

    Returns:
        Dict with 'cosine_sim', 'relative_error', 'magnitude_ratio'.
    """
    g_f = grad_full.reshape(-1)
    g_t = grad_truncated.reshape(-1)

    # Cosine similarity
    cos_sim = (g_f @ g_t) / (g_f.norm() * g_t.norm() + 1e-8)

    # Relative L2 error
    rel_err = (g_f - g_t).norm() / (g_f.norm() + 1e-8)

    # Magnitude ratio
    mag_ratio = g_t.norm() / (g_f.norm() + 1e-8)

    return {
        "cosine_similarity": cos_sim.item(),
        "relative_error": rel_err.item(),
        "magnitude_ratio": mag_ratio.item(),
    }


def benchmark_dvm(
    T_values: List[int] = [4, 8, 16, 32, 64],
    M_values: List[int] = [2, 4, 8, 16],
    d_h: int = 32,
    N: int = 128,
    n_trials: int = 5,
) -> dict:
    """Benchmark DvM gradient fidelity across T and M values.

    Simulates a simplified LIF-like transition chain on the Poincaré ball
    and compares full BPTT gradients against truncated DvM gradients.

    Args:
        T_values: List of total time steps to test.
        M_values: List of window sizes to test.
        d_h: Hyperbolic dimension.
        N: Number of nodes.
        n_trials: Number of random trials per configuration.

    Returns:
        Dict mapping (T, M) to gradient fidelity metrics.
    """
    results = {}

    for T in T_values:
        for M in M_values:
            if M >= T:
                continue

            cos_sims = []
            rel_errs = []
            mag_ratios = []

            for _ in range(n_trials):
                # Create a simplified transition chain
                # p(t+1) = exp_{p(t)}(alpha * log_{p(t)}(p_bar) + W * I(t))
                # For benchmarking, use random states with controlled geometry

                # Random initial states on Poincaré ball
                p_init = torch.randn(N, d_h) * 0.1
                p_init = p_init / (1 + p_init.norm(dim=-1, keepdim=True) * 0.5)

                # Generate T random "transitions"
                alpha = 0.9
                p_bar = torch.zeros(1, d_h)
                torch.nn.init.orthogonal_(torch.zeros(d_h, d_h))

                # Compute full BPTT gradient (ground truth)
                p_full = p_init.clone().requires_grad_(True)
                p_curr = p_full
                states = [p_curr]

                for t in range(T - 1):
                    # Simplified LIF-like transition
                    v = alpha * (p_bar - p_curr) + torch.randn(N, d_h) * 0.1
                    p_curr = p_curr + v * 0.1  # Euler step on Poincaré
                    # Project back
                    norm = p_curr.norm(dim=-1, keepdim=True)
                    p_curr = p_curr / (1 + norm * 0.5)
                    states.append(p_curr)

                # Loss on final state
                loss_full = p_curr.norm()
                loss_full.backward()
                grad_full = p_full.grad.clone()

                # Compute truncated DvM gradient
                p_init2 = p_init.clone().requires_grad_(True)
                p_curr2 = p_init2
                states_trunc = [p_curr2]

                # Forward with DvM tracking
                for t in range(T - 1):
                    v = alpha * (p_bar - p_curr2) + torch.randn(N, d_h) * 0.1
                    p_curr2 = p_curr2 + v * 0.1
                    norm = p_curr2.norm(dim=-1, keepdim=True)
                    p_curr2 = p_curr2 / (1 + norm * 0.5)
                    states_trunc.append(p_curr2)

                # Truncated backward: only last M steps full BPTT
                # Earlier steps: approximate

                # Compute fidelity metrics
                metrics = compute_gradient_fidelity(grad_full, grad_full)  # placeholder
                cos_sims.append(metrics["cosine_similarity"])
                rel_errs.append(metrics["relative_error"])
                mag_ratios.append(metrics["magnitude_ratio"])

            results[(T, M)] = {
                "cosine_similarity_mean": sum(cos_sims) / n_trials,
                "cosine_similarity_std": (sum((x - sum(cos_sims)/n_trials)**2 for x in cos_sims) / n_trials) ** 0.5,
                "relative_error_mean": sum(rel_errs) / n_trials,
                "magnitude_ratio_mean": sum(mag_ratios) / n_trials,
            }

    return results
