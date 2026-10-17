"""
DvM Core: Metric correction for Riemannian gradient computation.

Implements the complete pushforward/pullback mechanism on the tangent bundle
of the product manifold H^{d_h} x R^{d_e} as described in Theorem 4.1.

Key insight: For the Poincaré ball model with metric g_x = lambda_x^2 * I:
    grad_R f(p) = grad_E f(p) / lambda_p^2        (Riemannian -> Euclidean)
    grad_E f(p) = lambda_p^2 * grad_R f(p)        (Euclidean -> Riemannian)

When pulling back through φ: p -> φ(p):
    grad_E_at_p = (lambda_p^2 / lambda_{φ(p)}^2) * (Dφ)^T * grad_E_at_{φ(p)}

The term (Dφ)^T * grad_E is computed by autograd.
The DvM correction is the metric rescaling factor lambda_p^2 / lambda_{φ(p)}^2.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional

# Type aliases
Point = Tuple[torch.Tensor, torch.Tensor]     # (z_hyp, z_euc)


def conformal_factor(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Compute conformal factor for Poincaré ball: lambda_x = 2 / (1 - c||x||^2).

    Args:
        x: (..., d) point on Poincaré ball.
        c: Curvature parameter.
        eps: Numerical stability.

    Returns:
        (...,) conformal factor at each point.
    """
    x_norm_sq = (x * x).sum(dim=-1)
    return 2.0 / (1.0 - c * x_norm_sq).clamp(min=eps)


def metric_correction_factor(
    p_hyp: torch.Tensor,
    p_new_hyp: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the metric correction factor for DvM gradient pullback.

    For the Poincaré ball with metric g_x = lambda_x^2 * I:
        grad_E_at_p = (lambda_p^2 / lambda_{p_new}^2) * (Dφ)^T * grad_E_at_{p_new}

    This function returns: lambda_p^2 / lambda_{p_new}^2.

    Args:
        p_hyp: (N, d_h) state at step t.
        p_new_hyp: (N, d_h) state at step t+1.
        c: Curvature parameter.
        eps: Minimum value for denominator.

    Returns:
        (N,) correction factor per node.
    """
    lam_p = conformal_factor(p_hyp, c, eps)
    lam_p_new = conformal_factor(p_new_hyp, c, eps)
    lam_ratio = lam_p / lam_p_new
    return lam_ratio ** 2  # (lambda_p / lambda_{p_new})^2


def apply_metric_correction(
    grad_hyp: torch.Tensor,
    p_hyp: torch.Tensor,
    p_new_hyp: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """Apply the Riemannian metric correction to the hyperbolic gradient.

    Converts the Euclidean autograd gradient to the correctly-scaled
    Riemannian gradient via the pullback of the metric.

    Args:
        grad_hyp: (N, d_h) Euclidean gradient from autograd (already
                  includes (Dφ)^T from chain rule).
        p_hyp: (N, d_h) state at step t.
        p_new_hyp: (N, d_h) state at step t+1.
        c: Curvature parameter.

    Returns:
        (N, d_h) metric-corrected gradient.
    """
    factor = metric_correction_factor(p_hyp, p_new_hyp, c)
    return grad_hyp * factor.unsqueeze(-1)


def compute_contraction_factor(
    alpha_decay: float,
    grad_clip: float = 5.0,
) -> float:
    """Compute the theoretical contraction factor rho for DvM error bound.

    For LIF dynamics with membrane decay alpha:
    The state transition linearization has Jacobian spectral norm bounded
    by alpha < 1, ensuring contractivity. With weight regularization and
    gradient clipping, we have:
        rho = alpha + (1 - alpha) * ||W||_2

    In practice, gradient clipping ensures ||W||_2 <= grad_clip * factor,
    giving a conservative estimate.

    Args:
        alpha_decay: LIF membrane decay constant (0 < alpha < 1).
        grad_clip: Gradient clipping max_norm.

    Returns:
        Estimated contraction factor rho in [0, 1).
    """
    # Conservative estimate: rho = alpha (dominant LIF decay)
    # More aggressive: rho = alpha + (1-alpha)*small_weight_norm
    weight_norm_estimate = min(1.0 / grad_clip, 0.5)
    rho = alpha_decay + (1.0 - alpha_decay) * weight_norm_estimate
    return min(rho, 0.999)


def gradient_error_bound(
    T: int,
    M: int,
    grad_norm: float,
    rho: float,
    lam_ratio_sq: float = 1.0,
) -> float:
    """Compute the theoretical gradient error bound for truncated DvM.

    Theorem 1: epsilon_grad(T, M) <= C * (1 - rho^{T-M}) * (λ₀²/λ_T²)
    where C = ||∇L|| / (1 - rho).

    Args:
        T: Total time steps.
        M: Truncation window size.
        grad_norm: Norm of the loss gradient at final step ||∇_{p(T)}L||.
        rho: Contraction factor (0 <= rho < 1).
        lam_ratio_sq: Endpoint conformal factor ratio (λ₀/λ_T)².
                      Default 1.0 (Euclidean limit).

    Returns:
        Upper bound on gradient approximation error.
    """
    C = grad_norm / (1.0 - rho)
    return C * (1.0 - rho ** (T - M)) * lam_ratio_sq


class RiemannianGradientCorrector(nn.Module):
    """Applies Riemannian metric correction during backward pass.

    This module wraps a manifold state transition and applies the
    correct pushforward/pullback metric scaling during backpropagation.

    Usage:
        corrector = RiemannianGradientCorrector(c=1.0)
        p_hyp.requires_grad_(True)
        p_new_hyp = some_transition(p_hyp)  # includes (Dφ)^T in autograd
        corrected = corrector(p_hyp, p_new_hyp)
        # corrected.backward() now has correct Riemannian scaling
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(
        self,
        p_hyp: torch.Tensor,
        p_new_hyp: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Riemannian metric correction during forward/backward.

        During forward: returns p_new_hyp unchanged.
        During backward: applies the metric correction lambda_p^2/lambda_{p_new}^2
                         to the gradient flowing to p_hyp.

        Args:
            p_hyp: (N, d_h) state at step t. Must have requires_grad=True.
            p_new_hyp: (N, d_h) state at step t+1.

        Returns:
            p_new_hyp with corrected backward path.
        """
        return RiemannianPullbackFunction.apply(p_hyp, p_new_hyp, self.c, self.eps)


class RiemannianPullbackFunction(torch.autograd.Function):
    """Custom autograd function implementing the Riemannian pullback.

    Forward: identity (no change to values).
    Backward: applies metric correction (lambda_p/lambda_{p_new})^2 to
              convert Euclidean autograd gradient to Riemannian gradient.
    """

    @staticmethod
    def forward(ctx, p_hyp, p_new_hyp, c, eps):
        ctx.save_for_backward(p_hyp, p_new_hyp)
        ctx.c = c
        ctx.eps = eps
        return p_new_hyp

    @staticmethod
    def backward(ctx, grad_output):
        p_hyp, p_new_hyp = ctx.saved_tensors
        c = ctx.c
        eps = ctx.eps

        # Compute correction factor: (lambda_p / lambda_{p_new})^2
        lam_p = conformal_factor(p_hyp, c, eps)
        lam_p_new = conformal_factor(p_new_hyp, c, eps)
        lam_ratio_sq = (lam_p / lam_p_new.clamp(min=eps)) ** 2

        # Apply correction to the gradient flowing to p_hyp
        grad_p_hyp = grad_output * lam_ratio_sq.unsqueeze(-1)

        # p_new_hyp gradient passes through unchanged (it receives from above)
        return grad_p_hyp, grad_output, None, None


def compute_step_metrics(
    p_hyp: torch.Tensor,
    p_new_hyp: torch.Tensor,
    c: float = 1.0,
) -> dict:
    """Compute diagnostic metrics for a single DvM step.

    Args:
        p_hyp: (N, d_h) state at step t.
        p_new_hyp: (N, d_h) state at step t+1.
        c: Curvature parameter.

    Returns:
        Dict with 'lambda_ratio_mean', 'lambda_ratio_max', 'correction_factor_mean'.
    """
    lam_p = conformal_factor(p_hyp, c)
    lam_p_new = conformal_factor(p_new_hyp, c)
    lam_ratio = lam_p / lam_p_new.clamp(min=1e-5)
    correction = lam_ratio ** 2

    return {
        "lambda_p_mean": lam_p.mean().item(),
        "lambda_p_new_mean": lam_p_new.mean().item(),
        "lambda_ratio_mean": lam_ratio.mean().item(),
        "lambda_ratio_max": lam_ratio.max().item(),
        "correction_factor_mean": correction.mean().item(),
    }
