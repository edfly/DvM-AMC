"""
Manifold-valued Spiking Prototype Encoder (MSPE).

Projects input node features to the product manifold H^{d_h} x R^{d_e}
and generates an initial spike sequence via ManifoldLIFNeuron.

Pipeline:
  1. Linear(d_in, d_h + d_e) -> ReLU -> split
  2. Hyperbolic component: expmap0 -> projx (ensure within Poincaré ball)
  3. Euclidean component: standard linear
  4. ManifoldLIFNeuron generates initial spike sequence
  5. Returns initial manifold state (z_hyp_0, z_euc_0)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from spikergnn.manifolds import ProductManifold, Point
from spikergnn.neurons import ManifoldLIFNeuron


class MSPEncoder(nn.Module):
    """Manifold-valued Spiking Prototype Encoder.

    Projects input features to the product manifold and generates
    initial spike representations for subsequent temporal processing.

    Attributes:
        manifold: ProductManifold instance.
        input_proj: Linear projection layer.
        lif: ManifoldLIFNeuron for spike generation.
    """

    def __init__(
        self,
        in_dim: int,
        manifold: ProductManifold,
        T: int = 8,
        beta_leak: float = 0.3,
        gamma_input: float = 0.5,
        alpha_decay: float = 0.9,
        u_threshold: float = 0.3,
        delta_reset: float = 1.0,
        surrogate_beta: float = 5.0,
        surrogate_type: str = "atan",
        ablation: str = "full",
        pre_dim: int = 0,
    ):
        """Initialize MSPEncoder.

        Args:
            in_dim: Input feature dimension per node.
            manifold: ProductManifold instance.
            T: Number of time steps for spike generation.
            beta_leak: LIF leak parameter.
            gamma_input: LIF input scaling parameter.
            alpha_decay: LIF membrane potential decay.
            u_threshold: Spike threshold.
            delta_reset: Refractory reset amount.
            surrogate_beta: Surrogate gradient steepness.
            surrogate_type: "atan" (recommended) or "sigmoid".
            ablation: Ablation variant name.
        """
        super().__init__()
        self.manifold = manifold
        self.T = T
        self.d_total = manifold.d_h + manifold.d_e

        # Input projection: d_in -> d_h + d_e
        # Handle case where in_dim == 0 (no node features)
        proj_in = max(in_dim, 1)
        self.proj_in = proj_in

        # [P5] Node preprocessing MLP: lift poor raw features (e.g. ENZYMES' 3-dim)
        # into a richer representation before the manifold projection. This gives
        # the encoder non-linear capacity to build discriminative features that a
        # single Linear(3,128) cannot. Disabled when pre_dim <= 0 (default).
        if pre_dim and pre_dim > 0:
            self.pre_net = nn.Sequential(
                nn.Linear(proj_in, pre_dim),
                nn.ReLU(),
                nn.Linear(pre_dim, pre_dim),
                nn.ReLU(),
            )
            self.input_proj = nn.Linear(pre_dim, self.d_total)
        else:
            self.pre_net = None
            self.input_proj = nn.Linear(proj_in, self.d_total)

        # Manifold LIF neuron for spike generation
        self.lif = ManifoldLIFNeuron(
            manifold=manifold,
            beta_leak=beta_leak,
            gamma_input=gamma_input,
            alpha_decay=alpha_decay,
            u_threshold=u_threshold,
            delta_reset=delta_reset,
            surrogate_beta=surrogate_beta,
            surrogate_type=surrogate_type,
            ablation=ablation,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Point]:
        """Encode input node features to product manifold spike sequence.

        Args:
            x: (total_nodes, d_in) node features. If None, uses constant-1.
            edge_index: (2, total_edges) edge indices (unused in projection,
                        reserved for future graph-aware encoding).
            batch: (total_nodes,) node-to-graph assignment vector.

        Returns:
            spike_seq: (total_nodes, T, d_total) spike sequence.
            p_init: (p_hyp_0, p_euc_0) initial product manifold state.
                    p_hyp_0: (total_nodes, d_h), p_euc_0: (total_nodes, d_e).
        """
        # Handle no node features
        if x is None or x.numel() == 0:
            num_nodes = batch.numel()
            x = torch.ones(num_nodes, self.proj_in, dtype=torch.float, device=edge_index.device)

        # [P5] Node preprocessing MLP (disabled if pre_net is None)
        if self.pre_net is not None:
            x = self.pre_net(x)

        # Linear projection + ReLU
        z = torch.relu(self.input_proj(x))  # (N, d_total)

        # Split into hyperbolic and Euclidean components
        z_hyp, z_euc = self.manifold.split(z)

        # Project hyperbolic component to Poincaré ball via expmap0
        z_hyp = self.manifold.hyp.expmap0(z_hyp)
        z_hyp = self.manifold.hyp.projx(z_hyp)

        # Euclidean component: standard (already in Euclidean space)
        # No additional transformation needed

        # Create initial product manifold state
        p_init = (z_hyp, z_euc)

        # Generate spike sequence via LIF neuron
        # Create input sequence: repeat projected features over T steps
        # x_seq: (N, T, d_total)
        z_merged = self.manifold.merge(z_hyp, z_euc)
        x_seq = z_merged.unsqueeze(1).expand(-1, self.T, -1).clone()

        # Add small noise to prevent identical inputs across time steps
        noise = torch.randn_like(x_seq) * 0.01
        x_seq = x_seq + noise

        spikes, p_final = self.lif(x_seq, p_init)

        return spikes, p_init
