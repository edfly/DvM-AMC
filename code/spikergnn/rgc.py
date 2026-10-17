"""
Riemannian Graph Convolution (RGC).

Implements geodesic message passing on the product manifold H^{d_h} x R^{d_e}:

For each node v and neighbor u:
  1. logmap(z_v, z_u) — map neighbor to v's tangent space
  2. Aggregate in tangent space (GIN-style MLP)
  3. expmap(aggregate, z_v) — map back to manifold
  4. projx — ensure result stays on manifold

Hyperbolic component: uses exp/log from PoincareManifold.
Euclidean component: standard GCN/MLP (exp/log degenerate to identity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.utils import add_self_loops

from spikergnn.manifolds import ProductManifold, Point


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Create a 2-layer MLP for GIN convolution.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.

    Returns:
        nn.Sequential MLP.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class RiemannianGraphConv(nn.Module):
    """Riemannian Graph Convolution on the product manifold.

    Uses GIN-style message passing with geodesic exp/log mappings
    for the hyperbolic component and standard GIN for the Euclidean
    component.

    Attributes:
        manifold: ProductManifold instance.
        gin_hyp: GINConv for hyperbolic tangent space aggregation.
        gin_euc: GINConv for Euclidean space aggregation.
        ablation: Ablation variant. "euclidean" disables hyperbolic ops.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        hidden_dim: int = 128,
        num_layers: int = 1,
        ablation: str = "full",
    ):
        """Initialize RiemannianGraphConv.

        Args:
            manifold: ProductManifold instance.
            hidden_dim: MLP hidden dimension (default 2 * d_total for GIN).
            num_layers: Number of GIN layers (unused, kept for interface compat).
            ablation: Ablation variant name. "euclidean" replaces hyperbolic
                      operations with linear layers.
        """
        super().__init__()
        self.manifold = manifold
        self.d_h = manifold.d_h
        self.d_e = manifold.d_e
        self.ablation = ablation

        mlp_hidden = 2 * (self.d_h + self.d_e)

        # GIN for hyperbolic tangent space operations
        if ablation == "euclidean":
            # Ablation: replace hyperbolic ops with standard linear
            self.hyp_linear = nn.Linear(self.d_h, self.d_h)
        else:
            self.gin_hyp = GINConv(
                nn=make_mlp(self.d_h, mlp_hidden, self.d_h),
                train_eps=True,
            )

        # GIN for Euclidean space (standard)
        self.gin_euc = GINConv(
            nn=make_mlp(self.d_e, mlp_hidden, self.d_e),
            train_eps=True,
        )

        # Bug 7 fix: make residual step size learnable instead of hardcoded 0.1.
        # Using sigmoid constraint ensures alpha stays in (0, 1) for stability.
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        z: Point,
        edge_index: torch.Tensor,
    ) -> Point:
        """Perform one round of Riemannian graph convolution.

        Args:
            z: (z_hyp, z_euc) product manifold node embeddings.
               z_hyp: (N, d_h), z_euc: (N, d_e).
            edge_index: (2, E) edge indices.

        Returns:
            z_new: (z_hyp_new, z_euc_new) updated embeddings on the product manifold.
        """
        z_hyp, z_euc = z

        # Add self-loops for stable message passing
        edge_index_with_sl, _ = add_self_loops(
            edge_index, num_nodes=z_hyp.size(0)
        )

        # ── Hyperbolic component ───────────────────────────────────
        if self.ablation == "euclidean":
            # Ablation: linear layer replaces geodesic message passing
            h_hyp = self.hyp_linear(z_hyp)
            z_hyp_new = self.manifold.hyp.projx(h_hyp)
        else:
            # Step 1: Logmap all neighbors to tangent space at origin
            # GIN aggregates in tangent space (Euclidean), then we project
            # the result back to the manifold via expmap0.
            log_hyp = self.manifold.hyp.logmap0(z_hyp)

            # Step 2: GIN aggregation in tangent space
            h_hyp = self.gin_hyp(log_hyp, edge_index_with_sl)

            # Step 3: Expmap back to manifold
            z_hyp_new = self.manifold.hyp.expmap0(h_hyp)
            z_hyp_new = self.manifold.hyp.projx(z_hyp_new)

            # Residual connection with geodesic interpolation
            # Bug 7 fix: use learnable alpha (sigmoid-constrained to (0,1))
            alpha = torch.sigmoid(self.residual_alpha)
            v_hyp = self.manifold.hyp.logmap(z_hyp_new, z_hyp)
            z_hyp_new = self.manifold.hyp.expmap(alpha * v_hyp, z_hyp)
            z_hyp_new = self.manifold.hyp.projx(z_hyp_new)

        # ── Euclidean component ────────────────────────────────────
        z_euc_new = self.gin_euc(z_euc, edge_index_with_sl)

        # Residual connection (Bug 7 fix: use same learnable alpha)
        alpha_euc = torch.sigmoid(self.residual_alpha)
        z_euc_new = z_euc + alpha_euc * (z_euc_new - z_euc)

        return (z_hyp_new, z_euc_new)
