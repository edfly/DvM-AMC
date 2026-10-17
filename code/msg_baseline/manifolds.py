"""
Hyperbolic manifold operations — Pure PyTorch implementation.

Replaces the geoopt.PoincareBall wrapper with numerically stable
pure PyTorch operations. This eliminates C-level segfaults that
occurred in geoopt when points approached the Poincaré ball boundary.

All formulas follow the standard Poincaré ball model with curvature c > 0.
Key safety measures:
  - projx clamps norm to <= (1/sqrt(c) - eps) before any operation
  - arctanh arguments are clamped to <= (1 - 1e-4) to prevent inf/nan
  - Zero-norm vectors are handled with epsilon guards
  - frechet_mean uses tangent-space-at-origin approximation for stability
"""

import torch
import torch.nn as nn
from typing import Optional


class PoincareManifold:
    """
    Poincaré ball model of hyperbolic space H^d — Pure PyTorch.

    Implements all manifold operations (expmap, logmap, projx, dist,
    mobius_add, frechet_mean) in pure PyTorch with numerical safety
    guards. This replaces the geoopt-based implementation to eliminate
    C-level segfaults that occurred when points approached the ball boundary.

    Convention:
      - Ball: {x in R^d : ||x|| < 1/sqrt(c)}
      - Curvature c > 0 (actual sectional curvature is -c)
      - Conformal factor: lambda_x = 2 / (1 - c * ||x||^2)
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        """
        Args:
            c: Ball curvature parameter (c > 0 for standard Poincaré).
               Can be a float or a torch.Tensor (for learnable curvature).
               The actual hyperbolic curvature is -c.
            eps: Numerical stability epsilon for projection.
        """
        # Store c as tensor for differentiability
        if isinstance(c, torch.Tensor):
            self.c = c
        else:
            self.c = torch.tensor(c)
        self.eps = eps
        # max_norm is now computed dynamically in projx() from self.c

    # ---- Internal helpers ----

    def _safe_norm(self, x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
        """Compute L2 norm with minimum clamp to prevent division by zero.
        Default keepdim=True to ensure broadcasting works correctly.
        """
        return x.norm(dim=dim, keepdim=keepdim).clamp(min=self.eps)

    def _safe_arctanh(self, x: torch.Tensor) -> torch.Tensor:
        """arctanh with argument clamped to (-1+delta, 1-delta) to prevent inf."""
        return torch.atanh(x.clamp(min=-1.0 + 1e-4, max=1.0 - 1e-4))

    # ---- Basic maps ----

    def origin(self, *size, **kwargs) -> torch.Tensor:
        """Origin point (zero vector) in the Poincaré ball."""
        return torch.zeros(*size, **kwargs)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """Project x onto the open Poincaré ball (clip norm < 1/sqrt(c) - eps).

        This is the PRIMARY safety gate. All other operations assume inputs
        have been projected to stay well inside the ball.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # Dynamic max_norm from current c (supports learnable c)
        max_norm = 1.0 / (self.c ** 0.5) - self.eps
        # Scale factor: if norm > max_norm, shrink to max_norm; otherwise keep
        scale = torch.min(
            torch.ones_like(norm),
            max_norm / norm,
        )
        return x * scale

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent at origin to manifold.

        exp_0(v) = tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v

        This maps a tangent vector at the origin to a point on the ball.
        For c -> 0 (Euclidean limit) this reduces to exp_0(v) = v.
        """
        if float(self.c) < 1e-7:
            return self.projx(v)  # Euclidean limit: exp_0(v) = v
        v_norm = self._safe_norm(v)
        c_sqrt = self.c ** 0.5
        # tanh(c_sqrt * v_norm) / (c_sqrt * v_norm) * v
        arg = c_sqrt * v_norm
        scalar = torch.tanh(arg) / arg
        result = scalar * v
        return self.projx(result)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold to tangent at origin.

        log_0(x) = arctanh(sqrt(c) * ||x||) / (sqrt(c) * ||x||) * x

        This maps a point on the ball to a tangent vector at the origin.
        For c -> 0 (Euclidean limit) this reduces to log_0(x) = x.
        """
        if float(self.c) < 1e-7:
            return self.projx(x)  # Euclidean limit: log_0(x) = x
        x = self.projx(x)  # safety
        x_norm = self._safe_norm(x)
        c_sqrt = self.c ** 0.5
        arg = c_sqrt * x_norm
        scalar = self._safe_arctanh(arg) / arg
        return scalar * x

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition: x ⊕_c y.

        Formula:
          num = (1 + 2c<x,y> + c||y||^2) * x + (1 - c||x||^2) * y
          den = 1 + 2c<x,y> + c||x||^2 * ||y||^2
          result = num / den
        """
        x = self.projx(x)
        y = self.projx(y)
        x2 = (x * x).sum(dim=-1, keepdim=True)  # ||x||^2
        y2 = (y * y).sum(dim=-1, keepdim=True)  # ||y||^2
        xy = (x * y).sum(dim=-1, keepdim=True)  # <x,y>

        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + self.c * x2 * y2
        den = den.clamp(min=self.eps)

        result = num / den
        return self.projx(result)

    def expmap(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exponential map: T_x M -> M.  z = Exp_x(v).

        Formula (following geoopt convention):
          lambda_x = 2 / (1 - c * ||x||^2)
          second_term = (v / ||v||) * tanh(lambda_x * sqrt(c) * ||v|| / 2) / sqrt(c)
          result = x ⊕_c second_term
        For c -> 0 (Euclidean limit) this reduces to Exp_x(v) = x + v.
        """
        if float(self.c) < 1e-7:
            return self.projx(x + v)  # Euclidean limit: x + v
        x = self.projx(x)
        v_norm = self._safe_norm(v)
        c_sqrt = self.c ** 0.5

        # Conformal factor at x
        x2 = (x * x).sum(dim=-1, keepdim=True)
        lambda_x = 2.0 / (1.0 - self.c * x2).clamp(min=self.eps)

        # Compute second term
        direction = v / v_norm  # unit direction
        arg = lambda_x * c_sqrt * v_norm / 2.0
        magnitude = torch.tanh(arg) / c_sqrt
        second_term = magnitude * direction

        # Result via Möbius addition
        result = self.mobius_add(x, second_term)
        return self.projx(result)

    def logmap(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: M -> T_x M.  v = Log_x(y).

        Formula (following geoopt convention):
          lambda_x = 2 / (1 - c * ||x||^2)
          diff = (-x) ⊕_c y
          diff_norm = ||diff||
          result = (2 / (lambda_x * sqrt(c))) * arctanh(sqrt(c) * diff_norm) / diff_norm * diff
        For c -> 0 (Euclidean limit) this reduces to Log_x(y) = y - x.
        """
        if float(self.c) < 1e-7:
            return self.projx(y - x)  # Euclidean limit: y - x
        x = self.projx(x)
        y = self.projx(y)
        c_sqrt = self.c ** 0.5

        # Conformal factor at x
        x2 = (x * x).sum(dim=-1, keepdim=True)
        lambda_x = 2.0 / (1.0 - self.c * x2).clamp(min=self.eps)

        # Möbius difference: -x ⊕ y
        diff = self.mobius_add(-x, y)
        diff_norm = self._safe_norm(diff)

        # arctanh(sqrt(c) * diff_norm) / (sqrt(c) * diff_norm)
        arg = c_sqrt * diff_norm
        atanh_term = self._safe_arctanh(arg) / arg

        result = (2.0 / (lambda_x * c_sqrt)) * atanh_term * diff
        return result

    # ---- Distance ----

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance d_H(x, y).

        Formula:
          diff = (-x) ⊕_c y
          d = (2 / sqrt(c)) * arctanh(sqrt(c) * ||diff||)
        For c -> 0 (Euclidean limit) this reduces to d = ||y - x||.
        """
        if float(self.c) < 1e-7:
            return (x - y).norm(dim=-1)  # Euclidean limit
        x = self.projx(x)
        y = self.projx(y)
        c_sqrt = self.c ** 0.5

        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=self.eps)
        arg = c_sqrt * diff_norm
        return (2.0 / c_sqrt) * self._safe_arctanh(arg)

    def dist2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared geodesic distance."""
        d = self.dist(x, y)
        return d * d

    # ---- Mobius operations ----

    def mobius_matvec(
        self, m: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Möbius matrix-vector multiplication: M ⊙_c x.

        Formula:
          Mx = M @ x (standard matrix multiplication)
          result = (1/sqrt(c)) * tanh(||Mx||/||x|| * arctanh(sqrt(c)*||x||)) * Mx/||Mx||
        """
        x = self.projx(x)
        c_sqrt = self.c ** 0.5

        Mx = torch.nn.functional.linear(x, m)  # m @ x
        x_norm = self._safe_norm(x)
        Mx_norm = self._safe_norm(Mx)

        arg = Mx_norm / x_norm * self._safe_arctanh(c_sqrt * x_norm)
        scalar = torch.tanh(arg) / (c_sqrt * Mx_norm)
        result = scalar * Mx
        return self.projx(result)

    # ---- Frechet Mean ----

    def frechet_mean(
        self,
        points: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        max_iter: int = 20,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute weighted Fréchet mean on Poincaré ball.

        Uses tangent-space-at-origin approximation for maximum numerical
        stability. The Karcher flow approach (iterative logmap + expmap
        at arbitrary base points) is numerically fragile and was the
        primary source of C-level segfaults in geoopt.

        The tangent-space approximation:
          1. Map all points to tangent space at origin: v_i = logmap0(x_i)
          2. Weighted average in tangent space: v_mean = sum(w_i * v_i)
          3. Map back to manifold: mu = expmap0(v_mean)

        This is exact for points near the origin and a good approximation
        for points with moderate norms (||x|| << 1/sqrt(c)).

        Args:
            points: (N, d) — points on the Poincaré ball.
            weights: (N,) — non-negative weights summing to 1. Uniform if None.
            max_iter: Kept for API compatibility (unused in tangent-space method).
            lr: Kept for API compatibility (unused in tangent-space method).

        Returns:
            mu: (d,) — the Fréchet mean (on the manifold).
        """
        n_points = points.shape[0]

        # Fast path: single point → it IS the Fréchet mean
        if n_points == 1:
            return points.squeeze(0).clone()

        if weights is None:
            weights = torch.ones(n_points, device=points.device) / n_points

        # Tangent-space-at-origin approximation:
        # Map all points to tangent space at origin, average, map back.
        points = self.projx(points)
        log_pts = self.logmap0(points)  # (N, d) tangent vectors at origin

        # Weighted average in tangent space
        v_mean = (weights.unsqueeze(-1) * log_pts).sum(dim=0)  # (d,)

        # Map back to manifold
        mu = self.expmap0(v_mean.unsqueeze(0)).squeeze(0)  # (d,)
        return self.projx(mu.unsqueeze(0)).squeeze(0)

    # ---- Parallel transport (approximate) ----

    def transp(
        self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Transport tangent vector v from T_x to T_y.

        Uses the approximate parallel transport formula for Poincaré ball:
          v_transported = v * (lambda_x / lambda_y)
        where lambda_x = 2 / (1 - c||x||^2) is the conformal factor.

        This is exact along geodesics through the origin and a good
        approximation for small distances.
        """
        x = self.projx(x)
        y = self.projx(y)

        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)

        lambda_x = 2.0 / (1.0 - self.c * x2).clamp(min=self.eps)
        lambda_y = 2.0 / (1.0 - self.c * y2).clamp(min=self.eps)

        return v * (lambda_x / lambda_y)
