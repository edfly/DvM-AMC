"""
MSG* model architecture: MSNeuron + Riemannian Prototypical Classifier.

Implements the 4-stage MSNeuron pipeline from MSG (Sun et al., NeurIPS 2024):
  1. Euclidean aggregation: GCN(s[t]; W^l)
  2. IF spiking firing: s[t] = IFModel(x[t])
  3. Temporal pooling: v = MeanPool(x[1:T])
  4. Manifold update: z^l = Exp_{z^{l-1}}(epsilon * v)

Combined with ProtoNet-style episodic readout for few-shot graph classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch_geometric.nn import GCNConv

from .manifolds import PoincareManifold


# ── Surrogate Gradient ──────────────────────────────────────────────


class SigmoidSurrogate(nn.Module):
    """
    Sigmoid-based surrogate gradient for IF spike non-differentiability.

    During forward: returns binary spike (0 or 1).
    During backward: passes sigmoid gradient with controllable steepness.
    """

    def __init__(self, beta: float = 5.0):
        super().__init__()
        self.beta = beta

    def forward(self, v_mem: torch.Tensor) -> torch.Tensor:
        # Forward: hard threshold
        spike = (v_mem >= 0).float()

        # Backward: surrogate sigmoid gradient
        if self.training:
            sig = torch.sigmoid(self.beta * v_mem)
            surrogate = sig * (1 - sig) * self.beta
            return spike - surrogate.detach() + surrogate
        return spike


# ── Integrate-and-Fire Neuron Model ─────────────────────────────────


class IFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Discrete-time dynamics:
      v[t+1] = decay * v[t] + x[t]
      s[t+1] = spike(v[t+1])

    With reset-to-zero after firing.
    """

    def __init__(self, decay: float = 0.5, surrogate_beta: float = 5.0):
        super().__init__()
        self.decay = decay
        self.surrogate = SigmoidSurrogate(surrogate_beta)

    def forward(
        self, x_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a temporal sequence of inputs.

        Args:
            x_seq: (batch, T, features) — input sequence.

        Returns:
            s_seq: (batch, T, features) — spike sequence.
            v_final: (batch, features) — final membrane potential.
        """
        T = x_seq.shape[1]
        device = x_seq.device

        # Initialize membrane potential
        v = torch.zeros_like(x_seq[:, 0])  # (batch, features)
        spikes = []

        for t in range(T):
            # Accumulate
            v = self.decay * v + x_seq[:, t]

            # Fire
            s_t = self.surrogate(v)
            spikes.append(s_t)

            # Reset (reset-to-zero)
            v = v * (1 - s_t)

        s_seq = torch.stack(spikes, dim=1)  # (batch, T, features)
        return s_seq, v


# ── MSNeuron Layer (Single Layer) ───────────────────────────────────


class MSNeuronLayer(nn.Module):
    """
    Single MSNeuron layer implementing the MSG 4-stage pipeline:

      Stage 1: h[t] = GCN(s[t-1]; W)     — Euclidean aggregation
      Stage 2: s[t] = IF(h[t])             — Spiking activation
      Stage 3: v    = MeanPool(h[1:T])     — Temporal readout
      Stage 4: z    = Exp(z_prev, eps*v)   — Manifold update via expmap
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        manifold: PoincareManifold,
        epsilon: float = 0.1,
        tau: float = 20.0,
    ):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.if_neuron = IFNeuron(decay=1.0 - 1.0/tau)
        self.manifold = manifold
        self.epsilon = epsilon

    def forward(
        self,
        s_init: torch.Tensor,
        z_prev: torch.Tensor,
        edge_index: torch.Tensor,
        batch_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run one MSNeuron layer over T time steps.

        Args:
            s_init: (num_nodes, in_dim) — initial spike state (usually from prev layer).
            z_prev: (num_graphs, out_dim) — previous layer's manifold embedding (graph-level).
            edge_index: (2, num_edges) — graph connectivity.
            batch_vec: (num_nodes,) — node-to-graph assignment vector.

        Returns:
            z_new: (num_graphs, out_dim) — new manifold graph embeddings.
        """
        T = 8  # fixed time steps; could be made configurable per config
        num_nodes = s_init.shape[0]
        device = s_init.device

        # Expand graph-level manifold embedding to each node for residual
        # This broadcasts the previous manifold state to nodes
        s_current = s_init  # (N, D_in)

        mem_outputs = []

        for t in range(T):
            # Stage 1: GCN aggregation
            h_t = self.gcn(s_current, edge_index)  # (N, D_out)

            # Stage 2: IF spiking
            s_t, _ = self.if_neuron(h_t.unsqueeze(1))  # s_t: (N, 1, D_out)
            s_t = s_t.squeeze(1)  # (N, D_out)

            s_current = s_t
            mem_outputs.append(h_t)

        # Stage 3: Temporal mean pooling to get graph-level vectors
        mem_stack = torch.stack(mem_outputs, dim=1)  # (N, T, D_out)

        if batch_vec is not None:
            # Global mean pool over nodes within each graph, then mean over time
            v_pool = []
            max_batch = batch_vec.max().item() + 1
            for g_id in range(max_batch):
                mask = (batch_vec == g_id)
                g_mem = mem_stack[mask]  # (n_g_nodes, T, D_out)
                v_mean = g_mem.mean(dim=0).mean(dim=0)  # (D_out,)
                v_pool.append(v_mean)
            v = torch.stack(v_pool, dim=0)  # (num_graphs, D_out)
        else:
            # Single-graph case: pool all nodes then time
            v = mem_stack.mean(dim=0).mean(dim=0).unsqueeze(0)  # (1, D_out)

        # Stage 4: Exponential map update on manifold
        # Scale by epsilon and move from z_prev along direction v
        z_new = self.manifold.expmap(
            self.epsilon * v, z_prev
        )
        z_new = self.manifold.projx(z_new)

        return z_new


# ── Full MSG* Encoder ────────────────────────────────────────────────


class MSGStarEncoder(nn.Module):
    """
    Complete MSG* encoder: L stacked MSNeuron layers + input projection.

    Architecture: X -> Linear(d_in -> d_h) -> [MSNeuronLayer]×L -> H^{d_h}
    Output lives on the Poincaré ball H^{d_h}.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        manifold_c: float = 1.0,
        epsilon: float = 0.1,
        tau: float = 20.0,
    ):
        super().__init__()

        self.manifold = PoincareManifold(c=manifold_c)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        # Input projection layer
        # If in_dim==0 (no node features), we will generate virtual features
        # in forward(); set input_proj to accept 1-dummy dim.
        proj_in = max(in_dim, 1)
        self.input_proj = nn.Linear(proj_in, hidden_dim)

        # Stack of MSNeuron layers
        self.layers = nn.ModuleList()
        dims = [hidden_dim] + [hidden_dim] * num_layers
        for l in range(num_layers):
            self.layers.append(
                MSNeuronLayer(
                    in_dim=dims[l],
                    out_dim=dims[l + 1],
                    manifold=self.manifold,
                    epsilon=epsilon,
                    tau=tau,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of graphs into Poincaré ball embeddings.

        Args:
            x: (total_nodes, d_in) — concatenated node features.
            edge_index: (2, total_edges) — concatenated edge indices.
            batch: (total_nodes,) — node-to-graph assignment vector.

        Returns:
            z: (num_graphs, hidden_dim) — embeddings on Poincaré ball.
        """
        # Project input features; handle graphs with no node features
        if x is None:
            # No node features: use constant-1 virtual feature.
            # Graph structure is captured by edge_index in MSNeuron layers.
            num_nodes = batch.numel()
            x = torch.ones(num_nodes, 1, dtype=torch.float, device=edge_index.device)
        s_0 = torch.relu(self.input_proj(x))
        s_0 = self.manifold.projx(s_0)  # ensure within unit ball

        # Initialize manifold embeddings at origin
        num_graphs = batch.max().item() + 1
        z = self.manifold.origin(num_graphs, self.hidden_dim).to(x.device)

        # Apply MSNeuron layers sequentially
        for layer in self.layers:
            z = layer(s_init=s_0, z_prev=z, edge_index=edge_index, batch_vec=batch)

        return z


# ── Riemannian Prototypical Classifier ──────────────────────────────


class RiemannianProtoClassifier(nn.Module):
    """
    Prototypical classifier on Riemannian manifolds.

    Computes class prototypes as Fréchet means of support embeddings,
    classifies queries by geodesic distance to prototypes.
    """

    def __init__(self, manifold: PoincareManifold):
        super().__init__()
        self.manifold = manifold

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        """
        Compute Fréchet mean prototypes per class.

        Args:
            support_embeddings: (N_support, d) — support set embeddings on manifold.
            support_labels: (N_support,) — class indices (0..n_classes-1).
            n_classes: Number of classes (N-way).

        Returns:
            prototypes: (n_classes, d) — Fréchet mean of each class on manifold.
        """
        prototypes = []
        for c in range(n_classes):
            mask = support_labels == c
            class_embeds = support_embeddings[mask]  # (K, d) for K-shot
            proto = self.manifold.frechet_mean(class_embeds)
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)  # (N_way, d)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        """
        Classify query graphs based on geodesic distance to prototypes.

        Args:
            query_embeddings: (N_query, d) — query set embeddings.
            support_embeddings: (N_support, d) — support set embeddings.
            support_labels: (N_support,) — support labels.
            n_classes: N-way for this episode.

        Returns:
            logits: (N_query, n_classes) — negative distances as classification logits.
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels, n_classes
        )  # (N_way, d)

        # Geodesic distance from each query to each prototype
        # dists: (N_query, N_way)
        dists = torch.zeros(
            query_embeddings.shape[0], n_classes,
            device=query_embeddings.device
        )
        for c in range(n_classes):
            dists[:, c] = self.manifold.dist(
                query_embeddings, prototypes[c].unsqueeze(0)
            ).squeeze(0)

        # Convert to logits: negative distance (smaller distance = larger logit)
        return -dists
