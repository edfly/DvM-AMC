"""
Product manifold H^{d_h} x R^{d_e} operations.

Encapsulates a PoincareManifold (hyperbolic component) and standard
Euclidean space, providing unified operations that accept/return
Tuple[Tensor, Tensor] for the two components.

All manifold operations (expmap, logmap, dist, etc.) are decomposed
into per-component operations. Distance fusion uses a learnable
gate parameter lambda: d = lambda * d_H + (1 - lambda) * d_E.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.manifolds import PoincareManifold


# Type aliases for clarity
Point = Tuple[torch.Tensor, torch.Tensor]        # (z_hyp, z_euc)
TangentVec = Tuple[torch.Tensor, torch.Tensor]   # (v_hyp, v_euc)


class ProductManifold(nn.Module):
    """Product manifold H^{d_h} x R^{d_e}.

    Wraps a PoincareManifold for the hyperbolic component and provides
    standard Euclidean operations for the Euclidean component. All
    operations work on Tuple[Tensor, Tensor] representing the two
    components independently.

    Attributes:
        hyp: PoincareManifold instance for hyperbolic component.
        d_h: Dimension of the hyperbolic component.
        d_e: Dimension of the Euclidean component.
        c: Curvature parameter for the Poincaré ball.
    """

    def __init__(
        self,
        d_h: int = 32,
        d_e: int = 96,
        c: float = 1.0,
        learnable_c: bool = False,
        eps: float = 1e-5,
    ):
        """Initialize product manifold.

        Args:
            d_h: Hyperbolic component dimension.
            d_e: Euclidean component dimension.
            c: Poincaré ball curvature parameter (initial value).
            learnable_c: If True, treat c as a learnable nn.Parameter.
            eps: Numerical stability epsilon.
        """
        super().__init__()
        self.d_h = d_h
        self.d_e = d_e
        self.eps = eps
        self.learnable_c = learnable_c

        if learnable_c:
            # Store log(c) as parameter to ensure c > 0
            self._c_log = nn.Parameter(torch.tensor(math.log(max(c, 1e-5))))
        else:
            self._c_fixed = c

        # Initialize hyp with initial c as tensor (for learnable curvature)
        # self.c property returns tensor; pass directly (PoincareManifold now accepts tensor)
        self.hyp = PoincareManifold(c=self.c.detach().clone(), eps=eps)

    @property
    def c(self) -> torch.Tensor:
        """Current curvature value (always positive)."""
        if self.learnable_c:
            return torch.exp(self._c_log)
        else:
            return torch.tensor(self._c_fixed)

    def _sync_c(self):
        """Sync self.hyp.c with current self.c for PoincareManifold methods.

        self.c returns a tensor (requires_grad=True if learnable).
        Assign directly (no detach) so gradients flow through hyp operations.
        """
        c_tensor = self.c  # Calls property: exp(_c_log) if learnable
        self.hyp.c = c_tensor
        # Debug: verify grad propagates
        if self.learnable_c:
            if not c_tensor.requires_grad:
                print(f"[WARN] _sync_c: c_tensor has no grad! "
                      f"_c_log.requires_grad={self._c_log.requires_grad}")
            elif self.hyp.c.requires_grad:
                pass  # OK
            else:
                print(f"[WARN] _sync_c: hyp.c lost grad!")

    # ── Split / Merge ─────────────────────────────────────────────

    def split(self, z: torch.Tensor) -> Point:
        """Split a concatenated tensor into (hyperbolic, euclidean) components.

        Args:
            z: (..., d_h + d_e) concatenated representation.

        Returns:
            (z_hyp, z_euc) tuple of tensors with shapes (..., d_h) and (..., d_e).
        """
        return z[..., :self.d_h], z[..., self.d_h:]

    def merge(self, z_hyp: torch.Tensor, z_euc: torch.Tensor) -> torch.Tensor:
        """Merge (hyperbolic, euclidean) components into a single tensor.

        Args:
            z_hyp: (..., d_h) hyperbolic component.
            z_euc: (..., d_e) euclidean component.

        Returns:
            (..., d_h + d_e) concatenated representation.
        """
        return torch.cat([z_hyp, z_euc], dim=-1)

    # ── Core manifold operations ──────────────────────────────────

    def projx(self, z: Point) -> Point:
        """Project onto the product manifold.

        Hyperbolic component is clipped to the open unit ball.
        Euclidean component has no constraint.

        Args:
            z: (z_hyp, z_euc) point on product manifold.

        Returns:
            Projected point on product manifold.
        """
        self._sync_c()
        z_hyp, z_euc = z
        return (self.hyp.projx(z_hyp), z_euc)

    def expmap(self, v: TangentVec, x: Point) -> Point:
        """Exponential map on the product manifold.

        For hyperbolic component: Poincaré ball expmap.
        For Euclidean component: standard addition.

        Args:
            v: (v_hyp, v_euc) tangent vector at x.
            x: (x_hyp, x_euc) base point on the manifold.

        Returns:
            Point on the product manifold.
        """
        self._sync_c()
        v_hyp, v_euc = v
        x_hyp, x_euc = x
        z_hyp = self.hyp.expmap(v_hyp, x_hyp)
        z_euc = x_euc + v_euc
        return (z_hyp, z_euc)

    def logmap(self, y: Point, x: Point) -> TangentVec:
        """Logarithmic map on the product manifold.

        For hyperbolic component: Poincaré ball logmap.
        For Euclidean component: standard subtraction.

        Args:
            y: (y_hyp, y_euc) target point on the manifold.
            x: (x_hyp, x_euc) base point on the manifold.

        Returns:
            Tangent vector at x pointing toward y.
        """
        self._sync_c()
        y_hyp, y_euc = y
        x_hyp, x_euc = x
        v_hyp = self.hyp.logmap(y_hyp, x_hyp)
        v_euc = y_euc - x_euc
        return (v_hyp, v_euc)

    def dist(self, x: Point, y: Point) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-component distances on the product manifold.

        Args:
            x: (x_hyp, x_euc) first point.
            y: (y_hyp, y_euc) second point.

        Returns:
            (d_H, d_E) tuple of distance tensors.
        """
        self._sync_c()
        x_hyp, x_euc = x
        y_hyp, y_euc = y
        d_H = self.hyp.dist(x_hyp, y_hyp)
        d_E = (x_euc - y_euc).norm(dim=-1)
        return (d_H, d_E)

    def fused_dist(
        self, x: Point, y: Point, lam: torch.Tensor
    ) -> torch.Tensor:
        """Compute fused distance: lambda * d_H + (1 - lambda) * d_E.

        Args:
            x: (x_hyp, x_euc) first point.
            y: (y_hyp, y_euc) second point.
            lam: Gate parameter in (0, 1), typically sigmoid of a learnable scalar.

        Returns:
            Fused distance tensor.
        """
        d_H, d_E = self.dist(x, y)
        return lam * d_H + (1 - lam) * d_E

    def frechet_mean(
        self,
        points: Point,
        weights: Optional[torch.Tensor] = None,
        max_iter: int = 20,
        lr: float = 0.1,
    ) -> Point:
        """Compute Fréchet mean on the product manifold.

        Hyperbolic component: Karcher flow on Poincaré ball.
        Euclidean component: arithmetic mean.

        Args:
            points: (z_hyp, z_euc) where z_hyp is (N, d_h) and z_euc is (N, d_e).
            weights: Optional (N,) weights summing to 1.
            max_iter: Maximum Karcher iterations for hyperbolic component.
            lr: Learning rate for Karcher flow.

        Returns:
            (mu_hyp, mu_euc) Fréchet mean point.
        """
        self._sync_c()
        z_hyp, z_euc = points

        # Hyperbolic Fréchet mean via Karcher flow
        mu_hyp = self.hyp.frechet_mean(z_hyp, weights=weights, max_iter=max_iter, lr=lr)

        # Euclidean arithmetic mean
        if weights is None:
            mu_euc = z_euc.mean(dim=0)
        else:
            mu_euc = (weights.unsqueeze(-1) * z_euc).sum(dim=0)

        return (mu_hyp, mu_euc)

    def origin(self, *size, **kwargs) -> Point:
        """Return the origin of the product manifold.

        Both components are zero vectors.

        Args:
            *size: Batch dimensions (e.g., batch_size).
                   The feature dimensions (d_h, d_e) are appended automatically.

        Returns:
            (zero_hyp, zero_euc) origin point.
        """
        # PoincareBall.origin(*size) just creates zeros(*size)
        # We need to include the feature dimension explicitly
        origin_hyp = self.hyp.origin(*size, self.d_h, **kwargs)
        origin_euc = torch.zeros(*size, self.d_e, **kwargs)
        if isinstance(origin_hyp, torch.Tensor) and origin_euc.device != origin_hyp.device:
            origin_euc = origin_euc.to(origin_hyp.device)
        return (origin_hyp, origin_euc)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda_x = 2 / (1 - c * ||x||^2).

        Used by DvM for tangent-bundle gradient propagation.

        Args:
            x: (..., d_h) point on the Poincaré ball.

        Returns:
            (...,) conformal factor scalar.
        """
        x_norm_sq = (x * x).sum(dim=-1)
        return 2.0 / (1.0 - self.c * x_norm_sq).clamp(min=1e-5)
