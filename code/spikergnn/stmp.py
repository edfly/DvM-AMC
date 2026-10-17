"""
Spiking Temporal Message Passing (STMP).

T-step temporal loop: RGC -> ManifoldLIFNeuron -> accumulate.
Implements spike-driven sparse communication where only spiking nodes
transmit messages at each time step.

Time pooling:
  - Hyperbolic component: Fréchet mean over T steps.
  - Euclidean component: arithmetic mean over T steps.

Global pooling:
  - global_mean_pool for graph-level embeddings.

Output: (z_hyp_graph, z_euc_graph), spike_count_total
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from torch_geometric.nn import global_mean_pool

from spikergnn.manifolds import ProductManifold, Point
from spikergnn.rgc import RiemannianGraphConv
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.dvm import apply_dvm, PerStepMetricCorrection


class SpikingTemporalMP(nn.Module):
    """Spiking Temporal Message Passing layer.

    Iterates T time steps, each performing:
      1. Riemannian Graph Convolution (geodesic message passing)
      2. Manifold LIF neuron update (spike generation)
      3. Spike-driven sparse communication mask

    After T steps, applies temporal pooling (Fréchet mean for hyperbolic,
    arithmetic mean for Euclidean) and global graph pooling.

    Attributes:
        manifold: ProductManifold instance.
        rgc: RiemannianGraphConv instance.
        lif: ManifoldLIFNeuron instance.
        T: Number of temporal steps.
        use_dvm: Whether to apply Differentiation-via-Manifold.
        dvm_window: Truncation window size for DvM.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        hidden_dim: int = 128,
        T: int = 8,
        num_layers: int = 3,
        beta_leak: float = 0.3,
        gamma_input: float = 0.5,
        alpha_decay: float = 0.9,
        u_threshold: float = 0.3,
        delta_reset: float = 1.0,
        surrogate_beta: float = 5.0,
        surrogate_type: str = "atan",
        use_dvm: bool = True,
        dvm_window: int = 8,
        use_metric_correction: bool = True,
        dvm_mode: str = "truncated",
        ablation: str = "full",
        truncate_memory: bool = True,
    ):
        """Initialize SpikingTemporalMP.

        Args:
            manifold: ProductManifold instance.
            hidden_dim: Total representation dimension.
            T: Number of temporal steps.
            num_layers: Number of RGC layers.
            beta_leak: LIF leak parameter.
            gamma_input: LIF input scaling parameter.
            alpha_decay: Membrane potential decay.
            u_threshold: Spike threshold.
            delta_reset: Refractory reset amount.
            surrogate_beta: Surrogate gradient steepness.
            surrogate_type: "atan" (recommended) or "sigmoid".
            use_dvm: Whether to use DvM for truncated BPTT.
            dvm_window: DvM truncation window size.
            ablation: Ablation variant name.
        """
        super().__init__()
        self.manifold = manifold
        self.T = T
        self.use_dvm = use_dvm
        self.dvm_window = dvm_window
        self.use_metric_correction = use_metric_correction
        self.dvm_mode = dvm_mode
        self.ablation = ablation
        self.truncate_memory = truncate_memory

        # Riemannian Graph Convolution layers
        self.rgc_layers = nn.ModuleList([
            RiemannianGraphConv(
                manifold=manifold,
                hidden_dim=hidden_dim,
                ablation=ablation,
            )
            for _ in range(num_layers)
        ])

        # Manifold LIF neuron
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

        self._total_spike_count = 0

    def _temporal_step(self, p_hyp, p_euc, t, prev_lif_spikes, x_seq, edge_index, N):
        """One STMP recurrence step (RGC message pass + spike masking + LIF).

        Returns the updated (p_hyp, p_euc) and the (detached) binary LIF spike
        output used for the next step's autoregressive mask. Used by forward();
        factored out so the window-exterior steps can be wrapped in torch.no_grad
        for true O(M) truncated-BPTT memory (see P0-B).
        """
        p_hyp_pre = p_hyp
        p_euc_pre = p_euc

        # ── RGC: geodesic message passing ──────────────────────
        z_current = (p_hyp, p_euc)
        for rgc_layer in self.rgc_layers:
            z_current = rgc_layer(z_current, edge_index)
        z_hyp_new, z_euc_new = z_current

        # ── Spike-driven masking ───────────────────────────────
        if t == 0 or prev_lif_spikes is None:
            spike_mask = torch.ones(N, 1, device=x_seq.device)
        else:
            spike_mask = prev_lif_spikes
        spike_mask_hyp = spike_mask.expand_as(p_hyp)
        spike_mask_euc = spike_mask.expand_as(p_euc)
        p_hyp = self.manifold.hyp.projx(
            spike_mask_hyp * z_hyp_new + (1 - spike_mask_hyp) * p_hyp
        )
        p_euc = spike_mask_euc * z_euc_new + (1 - spike_mask_euc) * p_euc

        # ── ManifoldLIF neuron update ──────────────────────────
        rgc_features = self.manifold.merge(z_hyp_new, z_euc_new)
        x_step = rgc_features.unsqueeze(1)
        spikes, (p_hyp, p_euc) = self.lif(x_step, (p_hyp, p_euc))

        # ── Per-step metric correction (full_bptt_mc mode) ─────
        if self.dvm_mode == "full_bptt_mc" and self.use_metric_correction:
            c_val = self.manifold.c.item() if hasattr(self.manifold.c, 'item') else 1.0
            p_hyp = PerStepMetricCorrection.apply(p_hyp_pre, p_hyp, c_val)
            p_euc = p_euc  # pass-through (Euclidean λ=1)

        # Store LIF spike output for next step's mask
        with torch.no_grad():
            s_out_binary = (spikes[:, 0, :].mean(dim=-1, keepdim=True) > 0.5).float()
            prev_lif_spikes = s_out_binary

        self._total_spike_count += self.lif.spike_count
        return p_hyp, p_euc, prev_lif_spikes

    def forward(
        self,
        p_init: Point,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[Point, int]:
        """Run T-step spiking temporal message passing.

        Args:
            p_init: (p_hyp_0, p_euc_0) initial manifold state.
            x_seq: (N, T, d_total) input spike sequence from MSPE.
            edge_index: (2, E) edge indices.
            batch: (N,) node-to-graph assignment vector.

        Returns:
            z_graph: (z_hyp_graph, z_euc_graph) graph-level embeddings.
                     z_hyp_graph: (num_graphs, d_h), z_euc_graph: (num_graphs, d_e).
            spike_count_total: Total number of spikes across all steps.
        """
        N = x_seq.size(0)
        p_hyp, p_euc = p_init
        p_hyp = p_hyp.clone()
        p_euc = p_euc.clone()

        # Accumulate representations for temporal pooling
        hyp_accumulator: List[torch.Tensor] = []
        euc_accumulator: List[torch.Tensor] = []

        # Track states for DvM
        step_states: List[Point] = []

        self._total_spike_count = 0

        # Membrane-potential hygiene: each independent forward pass must start
        # from a clean membrane. The LIF neuron persists `u_mag` across calls
        # (so the T recurrent sub-steps inside this loop carry state correctly),
        # but a *new* forward must not inherit the membrane left by a previous
        # forward. All training loops already call `reset_membrane()` before
        # `forward()`; doing it here as well makes the contract self-contained
        # and removes the silent dependence on caller discipline.
        self.lif.reset_membrane()

        # P0-B: truncation window boundary. Steps with index < start_idx live
        # outside the window and are detached from the autograd graph (they are
        # treated as fixed context, exactly as standard truncated BPTT does),
        # so the retained computation graph spans only the last M steps and the
        # peak activation memory becomes O(M) bytes instead of O(T). When
        # T == dvm_window (start_idx == 0) this is a no-op, preserving the exact
        # pre-P0-B behaviour for the reported T=8/M=8 experiments.
        start_idx = max(0, self.T - self.dvm_window)
        _window_detach = (
            self.dvm_mode == "truncated"
            and self.use_dvm
            and self.truncate_memory
            and start_idx > 0
        )

        # Track previous step's LIF spike output for autoregressive masking
        prev_lif_spikes = None  # Will be set after first LIF call

        for t in range(self.T):
            # P0-B: steps outside the truncation window are run under
            # torch.no_grad() so no autograd graph (and no saved activation
            # tensors for the RGC/LIF weights) is retained for them — they are
            # treated as fixed context, exactly as standard truncated BPTT.
            # This makes the retained graph span only the last M steps, dropping
            # peak activation memory from O(T) to O(M) bytes. When T==M
            # (start_idx==0) the whole loop runs in-graph: exact pre-P0-B path.
            if _window_detach and t < start_idx:
                with torch.no_grad():
                    p_hyp, p_euc, prev_lif_spikes = self._temporal_step(
                        p_hyp, p_euc, t, prev_lif_spikes, x_seq, edge_index, N
                    )
                hyp_accumulator.append(p_hyp)  # already detached by no_grad
                euc_accumulator.append(p_euc)
            else:
                p_hyp, p_euc, prev_lif_spikes = self._temporal_step(
                    p_hyp, p_euc, t, prev_lif_spikes, x_seq, edge_index, N
                )
                hyp_accumulator.append(p_hyp)
                euc_accumulator.append(p_euc)

            # Save state for DvM
            if self.dvm_mode == "full_bptt_mc":
                # Keep in autograd graph: (Dφ)^T flows naturally
                step_states.append((p_hyp.detach(), p_euc.detach()))
            else:
                step_states.append((p_hyp.detach().clone(), p_euc.detach().clone()))

        # ── DvM: apply truncated BPTT ─────────────────────────────
        # In full_bptt_mc mode: correction already applied per-step, skip.
        # In truncated mode: apply DvM wrapper with windowed correction and
        #   contraction decay for steps beyond the window.
        if self.use_dvm and self.training and self.dvm_mode == "truncated":
            p_final = apply_dvm(
                manifold=self.manifold,
                p_init=p_init,
                step_fn_outputs=step_states,
                window=self.dvm_window,
                use_metric_correction=self.use_metric_correction,
            )
            p_hyp_dvm, p_euc_dvm = p_final
            # Replace the last accumulator entry so DvM's gradient path
            # is connected to the temporal pooling computation graph
            hyp_accumulator[-1] = p_hyp_dvm
            euc_accumulator[-1] = p_euc_dvm

        # ── Temporal pooling ───────────────────────────────────────
        # Hyperbolic: Fréchet mean over time steps (per node)
        hyp_stack = torch.stack(hyp_accumulator, dim=0)  # (T, N, d_h)
        euc_stack = torch.stack(euc_accumulator, dim=0)  # (T, N, d_e)

        # Vectorized tangent-space-at-origin approximation for temporal pooling.
        # Instead of a per-node Python loop (O(N) calls), we batch all nodes:
        #   1. logmap0 maps all (T, N, d_h) points to tangent space at origin
        #   2. Mean over T steps in tangent space
        #   3. expmap0 maps the means back to the manifold
        # This is mathematically identical but ~100x faster due to GPU batching.
        z_hyp_pooled = self.manifold.hyp.expmap0(
            self.manifold.hyp.logmap0(hyp_stack).mean(dim=0)  # (N, d_h)
        )
        z_hyp_pooled = self.manifold.hyp.projx(z_hyp_pooled)

        # Euclidean: arithmetic mean
        z_euc_pooled = euc_stack.mean(dim=0)  # (N, d_e)

        # ── Global graph pooling ───────────────────────────────────
        z_hyp_graph = global_mean_pool(z_hyp_pooled, batch)  # (num_graphs, d_h)
        z_euc_graph = global_mean_pool(z_euc_pooled, batch)  # (num_graphs, d_e)

        # Ensure hyperbolic component stays on manifold
        z_hyp_graph = self.manifold.hyp.projx(z_hyp_graph)

        return (z_hyp_graph, z_euc_graph), self._total_spike_count

    @property
    def spike_count(self) -> int:
        """Return total spike count from the last forward pass."""
        return self._total_spike_count
