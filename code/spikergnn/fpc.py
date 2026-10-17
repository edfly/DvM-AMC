"""
Fréchet Prototypical Constructor (FPC).

Computes class prototypes on the product manifold H^{d_h} x R^{d_e}:
  - Hyperbolic component: Fréchet mean via Karcher flow on Poincaré ball.
  - Euclidean component: arithmetic mean.

1-shot special case: the single support sample IS the prototype.

Ablation "no_frechet": both components use arithmetic mean.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from spikergnn.manifolds import ProductManifold, Point


class FrechetProtoConstructor(nn.Module):
    """Fréchet Prototypical Constructor on the product manifold.

    Computes per-class prototypes by averaging support set embeddings:
    hyperbolic component uses Fréchet (Karcher) mean, Euclidean uses
    arithmetic mean. For ablation, both can use arithmetic mean.

    Attributes:
        manifold: ProductManifold instance.
        use_frechet: Whether to use Fréchet mean for hyperbolic component.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        use_frechet: bool = True,
    ):
        """Initialize FrechetProtoConstructor.

        Args:
            manifold: ProductManifold instance.
            use_frechet: If True, use Fréchet mean for hyperbolic component.
                         If False, use arithmetic mean (ablation: no_frechet).
        """
        super().__init__()
        self.manifold = manifold
        self.use_frechet = use_frechet

    def forward(
        self,
        support_z: Point,
        support_labels: torch.Tensor,
        n_classes: int,
    ) -> Point:
        """Compute class prototypes from support set embeddings.

        Args:
            support_z: (z_hyp, z_euc) support set embeddings.
                       z_hyp: (N_support, d_h), z_euc: (N_support, d_e).
            support_labels: (N_support,) class indices (0..n_classes-1).
            n_classes: Number of classes (N-way).

        Returns:
            prototypes: (proto_hyp, proto_euc) per-class prototypes.
                        proto_hyp: (n_classes, d_h), proto_euc: (n_classes, d_e).
        """
        z_hyp, z_euc = support_z
        device = z_hyp.device

        proto_hyp_list = []
        proto_euc_list = []

        for c in range(n_classes):
            mask = (support_labels == c)
            class_hyp = z_hyp[mask]  # (K, d_h)
            class_euc = z_euc[mask]  # (K, d_e)
            k_shot = class_hyp.size(0)

            if k_shot == 0:
                # No support examples for this class (shouldn't happen)
                proto_hyp_list.append(self.manifold.hyp.origin(self.manifold.d_h).to(device))
                proto_euc_list.append(torch.zeros(self.manifold.d_e, device=device))
                continue

            if k_shot == 1:
                # 1-shot: the sample IS the prototype
                proto_hyp_list.append(class_hyp.squeeze(0))
                proto_euc_list.append(class_euc.squeeze(0))
                continue

            # ── Hyperbolic component ───────────────────────────────
            if self.use_frechet:
                # Fréchet mean via Karcher flow
                proto_hyp_c = self.manifold.hyp.frechet_mean(class_hyp)
            else:
                # Ablation: arithmetic mean (projected to ball)
                mean_hyp = class_hyp.mean(dim=0)
                proto_hyp_c = self.manifold.hyp.projx(mean_hyp)

            # ── Euclidean component ────────────────────────────────
            proto_euc_c = class_euc.mean(dim=0)

            proto_hyp_list.append(proto_hyp_c)
            proto_euc_list.append(proto_euc_c)

        proto_hyp = torch.stack(proto_hyp_list, dim=0)  # (n_classes, d_h)
        proto_euc = torch.stack(proto_euc_list, dim=0)  # (n_classes, d_e)

        # Ensure hyperbolic prototypes are on the manifold
        proto_hyp = self.manifold.hyp.projx(proto_hyp)

        return (proto_hyp, proto_euc)
