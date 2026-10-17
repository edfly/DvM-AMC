"""
Geodesic Distance Classifier (GDC).

Classifies query samples by fused geodesic distance to prototypes:
  d = lambda * d_H + (1 - lambda) * d_E

lambda is a learnable gate parameter (passed through sigmoid to
constrain to (0, 1)). Logits are negative distances (closer = higher logit).

Ablation "euclidean": pure Euclidean distance (lambda forced to 0).
"""

import torch
import torch.nn as nn
from typing import Tuple

from spikergnn.manifolds import ProductManifold, Point


class GeodesicDistClassifier(nn.Module):
    """Geodesic Distance Classifier on the product manifold.

    Computes fused distances from query embeddings to class prototypes
    using a learnable gate parameter lambda that balances hyperbolic
    and Euclidean distance contributions.

    Attributes:
        manifold: ProductManifold instance.
        lam: Learnable gate parameter (pre-sigmoid).
        ablation: Ablation variant name.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        gate_init: float = 0.5,
        ablation: str = "full",
    ):
        """Initialize GeodesicDistClassifier.

        Args:
            manifold: ProductManifold instance.
            gate_init: Initial value for the gate parameter lambda.
            ablation: Ablation variant. "euclidean" forces lambda=0.
        """
        super().__init__()
        self.manifold = manifold
        self.ablation = ablation

        # Learnable gate: lambda = sigmoid(lam_raw), constrained to (0, 1)
        # Initialize lam_raw so sigmoid(lam_raw) = gate_init
        # sigmoid(x) = 0.5 => x = 0; sigmoid(x) = p => x = log(p/(1-p))
        if gate_init <= 0 or gate_init >= 1:
            gate_init = 0.5
        lam_raw_init = torch.log(torch.tensor(gate_init / (1 - gate_init)))
        self.lam_raw = nn.Parameter(lam_raw_init.unsqueeze(0))

    @property
    def lam(self) -> torch.Tensor:
        """Current gate value (sigmoid of learnable parameter)."""
        if self.ablation == "euclidean":
            return torch.tensor(0.0, device=self.lam_raw.device)
        return torch.sigmoid(self.lam_raw).squeeze()

    def forward(
        self,
        query_z: Point,
        prototypes: Point,
    ) -> torch.Tensor:
        """Classify queries by geodesic distance to prototypes.

        Args:
            query_z: (q_hyp, q_euc) query embeddings.
                     q_hyp: (N_query, d_h), q_euc: (N_query, d_e).
            prototypes: (proto_hyp, proto_euc) class prototypes.
                        proto_hyp: (n_classes, d_h), proto_euc: (n_classes, d_e).

        Returns:
            logits: (N_query, n_classes) negative fused distances.
        """
        q_hyp, q_euc = query_z
        proto_hyp, proto_euc = prototypes
        n_query = q_hyp.size(0)
        n_classes = proto_hyp.size(0)
        device = q_hyp.device

        # Compute distances from each query to each prototype
        d_H = torch.zeros(n_query, n_classes, device=device)
        d_E = torch.zeros(n_query, n_classes, device=device)

        for c in range(n_classes):
            # Hyperbolic distance
            d_H[:, c] = self.manifold.hyp.dist(
                q_hyp, proto_hyp[c].unsqueeze(0).expand_as(q_hyp)
            )
            # Euclidean distance
            d_E[:, c] = (q_euc - proto_euc[c].unsqueeze(0).expand_as(q_euc)).norm(dim=-1)

        # Fused distance: lambda * d_H + (1 - lambda) * d_E
        lam = self.lam
        fused_dist = lam * d_H + (1 - lam) * d_E

        # Logits = negative distance (closer = higher logit)
        logits = -fused_dist

        return logits
