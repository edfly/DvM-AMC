"""
SpikeRPGNN: Complete model assembly.

Assembles the 5 core modules:
  1. MSPEncoder — input projection to product manifold
  2. SpikingTemporalMP — T-step temporal message passing with LIF
  3. FrechetProtoConstructor — class prototype computation
  4. GeodesicDistClassifier — distance-based classification
  5. ProductManifold — shared manifold operations

Also provides the factory function create_spikergnn(config) that
constructs model variants based on the ablation configuration.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.manifolds import ProductManifold, Point
from spikergnn.mspe import MSPEncoder
from spikergnn.stmp import SpikingTemporalMP
from spikergnn.fpc import FrechetProtoConstructor
from spikergnn.gdc import GeodesicDistClassifier


class SpikeRPGNN(nn.Module):
    """SpikeRPGNN: Spiking Riemannian Prototypical Graph Neural Network.

    Complete few-shot graph classification model combining product manifold
    representations, spiking dynamics, and geodesic prototype classification.

    Attributes:
        config: SpikeRPGNNConfig instance.
        manifold: ProductManifold instance.
        mspe: MSPEncoder instance.
        stmp: SpikingTemporalMP instance.
        fpc: FrechetProtoConstructor instance.
        gdc: GeodesicDistClassifier instance.
    """

    def __init__(
        self,
        config: SpikeRPGNNConfig,
        in_dim: int = 7,
    ):
        """Initialize SpikeRPGNN.

        Args:
            config: SpikeRPGNNConfig with all hyperparameters.
            in_dim: Input feature dimension per node.
        """
        super().__init__()
        self.config = config

        # Shared product manifold
        self.manifold = ProductManifold(
            d_h=config.d_h,
            d_e=config.d_e,
            c=config.c_init,
            learnable_c=config.learnable_c,
        )

        # MSPE: input projection to product manifold
        self.mspe = MSPEncoder(
            in_dim=in_dim,
            manifold=self.manifold,
            T=config.T,
            beta_leak=config.beta_leak,
            gamma_input=config.gamma_input,
            alpha_decay=config.alpha_decay,
            u_threshold=config.u_threshold,
            delta_reset=config.delta_reset,
            surrogate_beta=config.surrogate_beta,
            surrogate_type=config.surrogate_type,
            ablation=config.ablation,
            pre_dim=config.node_pre_dim,
        )

        # STMP: spiking temporal message passing
        self.stmp = SpikingTemporalMP(
            manifold=self.manifold,
            hidden_dim=config.hidden_dim,
            T=config.T,
            num_layers=config.num_layers,
            beta_leak=config.beta_leak,
            gamma_input=config.gamma_input,
            alpha_decay=config.alpha_decay,
            u_threshold=config.u_threshold,
            delta_reset=config.delta_reset,
            surrogate_beta=config.surrogate_beta,
            surrogate_type=config.surrogate_type,
            use_dvm=config.use_dvm,
            dvm_window=config.dv_m_window,
            use_metric_correction=config.use_metric_correction,
            dvm_mode=config.dvm_mode,
            ablation=config.ablation,
            truncate_memory=config.truncate_memory,
        )

        # FPC: Fréchet prototype constructor
        use_frechet = config.ablation != "no_frechet"
        self.fpc = FrechetProtoConstructor(
            manifold=self.manifold,
            use_frechet=use_frechet,
        )

        # GDC: geodesic distance classifier
        self.gdc = GeodesicDistClassifier(
            manifold=self.manifold,
            gate_init=config.gate_init,
            ablation=config.ablation,
        )

        # Spike count tracking
        self._last_spike_count = 0

        # For no_meta ablation: add a linear classification head that
        # replaces the prototypical distance mechanism. This tests whether
        # episodic meta-learning contributes by using standard supervised
        # classification (linear head + nearest-neighbor) instead.
        if config.ablation == "no_meta":
            self.meta_head = nn.Sequential(
                nn.Linear(config.d_e, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.meta_head = None

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Point:
        """Encode a batch of graphs into product manifold embeddings.

        Args:
            x: (total_nodes, d_in) node features.
            edge_index: (2, total_edges) edge indices.
            batch: (total_nodes,) node-to-graph assignment vector.

        Returns:
            z_graph: (z_hyp_graph, z_euc_graph) graph-level embeddings.
        """
        # MSPE: project to product manifold and generate spikes
        spike_seq, p_init = self.mspe(x, edge_index, batch)

        # STMP: temporal message passing + pooling
        z_graph, spike_count = self.stmp(
            p_init, spike_seq, edge_index, batch
        )

        self._last_spike_count = spike_count
        return z_graph

    def classify(
        self,
        query_z: Point,
        support_z: Point,
        support_labels: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        """Classify query graphs by geodesic distance to prototypes.

        Args:
            query_z: (q_hyp, q_euc) query graph embeddings.
            support_z: (s_hyp, s_euc) support graph embeddings.
            support_labels: (N_support,) class indices.
            n_classes: Number of classes (N-way).

        Returns:
            logits: (N_query, n_classes) classification logits.
        """
        # Ablation "no_meta": use linear classification head instead of
        # prototypical distance. This bypasses the episodic prototypical
        # mechanism and tests whether meta-learning contributes.
        if self.config.ablation == "no_meta" and self.meta_head is not None:
            # Use Euclidean component for classification (standard supervised)
            q_euc = query_z[1]  # (N_query, d_e)
            logits = self.meta_head(q_euc)  # (N_query, hidden_dim)
            # Project to n_classes via a simple linear map
            # Reuse the Euclidean distance to support set as a fallback
            # but with the meta_head transformation applied
            s_euc = support_z[1]
            s_labels = support_labels
            # Transform support and query, then use nearest-neighbor
            q_trans = self.meta_head(q_euc)  # (N_query, hidden_dim)
            s_trans = self.meta_head(s_euc)  # (N_support, hidden_dim)
            # Nearest-neighbor classification in transformed space
            # (standard supervised, no Fréchet prototypes)
            logits = torch.zeros(q_trans.size(0), n_classes, device=q_trans.device)
            for c in range(n_classes):
                mask = (s_labels == c)
                if mask.sum() > 0:
                    proto_c = s_trans[mask].mean(dim=0)  # arithmetic mean
                    dist_c = (q_trans - proto_c.unsqueeze(0)).norm(dim=-1)
                    logits[:, c] = -dist_c
            return logits

        # Compute class prototypes (standard prototypical path)
        prototypes = self.fpc(support_z, support_labels, n_classes)

        # Classify by geodesic distance
        logits = self.gdc(query_z, prototypes)

        return logits

    def forward(
        self,
        support_x: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_batch: torch.Tensor,
        query_x: torch.Tensor,
        query_edge_index: torch.Tensor,
        query_batch: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Full forward pass: encode + classify.

        Args:
            support_x: Support set node features.
            support_edge_index: Support set edges.
            support_batch: Support set batch vector.
            query_x: Query set node features.
            query_edge_index: Query set edges.
            query_batch: Query set batch vector.
            support_labels: Support set class labels.
            n_way: Number of classes (N-way).

        Returns:
            logits: (N_query, n_way) classification logits.
        """
        # Encode support and query sets
        support_z = self.encode(support_x, support_edge_index, support_batch)
        query_z = self.encode(query_x, query_edge_index, query_batch)

        # Classify
        logits = self.classify(query_z, support_z, support_labels, n_way)

        return logits

    def count_spikes(self) -> int:
        """Return the total spike count from the last forward pass (full scope).

        Full scope = STMP recurrent spikes (self._last_spike_count, accumulated over
        all T steps of the last forward) PLUS the MSPE encoder spikes
        (self.mspe.lif._spike_count, emitted once per encode). This matches the
        "MSPE + STMP" accounting scope reported in the paper (tab:energy_pareto);
        the earlier implementation returned only the STMP stage, which is the
        STMP-only strict lower bound, not the full-scope figure.
        """
        stmp_count = int(self._last_spike_count)
        mspe_lif = getattr(self.mspe, "lif", None)
        mspe_count = int(getattr(mspe_lif, "_spike_count", 0) or 0)
        return stmp_count + mspe_count

    def spike_rate_surrogate(self) -> torch.Tensor:
        """Return the differentiable spike rate from the last forward pass.

        Uses the LIF neuron's surrogate spike rate, which is differentiable
        with respect to u_th_learnable for spike rate regularization (Eq.spike_rate_reg).

        Returns:
            Tensor scalar: surrogate spike rate in [0, 1].
        """
        return self.stmp.lif.spike_rate_surrogate()


def create_spikergnn(
    config: SpikeRPGNNConfig,
    in_dim: int = 7,
) -> SpikeRPGNN:
    """Factory function to create SpikeRPGNN model based on config.

    Constructs the appropriate model variant based on config.ablation:
      - "full": Complete SpikeRPGNN
      - "no_spikes": ManifoldLIFNeuron → identity activation
      - "euclidean": ProductManifold → pure Euclidean (hyperbolic → linear)
      - "no_frechet": FPC uses arithmetic mean for both components
      - "no_dvm": DvM → full BPTT (use_dvm=False)
      - "no_meta": Episodic training → standard supervised

    Args:
        config: SpikeRPGNNConfig instance.
        in_dim: Input feature dimension per node.

    Returns:
        Configured SpikeRPGNN model instance.
    """
    # Adjust config for ablation variants
    # Create a copy of config to avoid modifying the original
    ablation_config = SpikeRPGNNConfig(
        **{k: getattr(config, k) for k in config.__dataclass_fields__}
    )
    
    if config.ablation == "no_dvm":
        # Override use_dvm for no_dvm ablation
        ablation_config.use_dvm = False
    elif config.ablation == "no_spikes":
        # MSPEncoder will handle this via ablation parameter
        pass
    elif config.ablation == "euclidean":
        # MSPEncoder and other modules will handle this via ablation parameter
        pass
    elif config.ablation == "no_frechet":
        # FPC will handle this via ablation parameter
        pass
    elif config.ablation == "no_meta":
        # SpikeRPGNN will handle this via ablation parameter
        pass
    # For "full" variant, use default config (no overrides)

    model = SpikeRPGNN(config=ablation_config, in_dim=in_dim)
    return model
