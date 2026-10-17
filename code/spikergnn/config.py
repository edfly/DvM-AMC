"""
SpikeRPGNN configuration: extends MSG baseline ExperimentConfig.

Adds hyperparameters for the product manifold, LIF neuron dynamics,
spiking thresholds, gate initialization, and ablation control.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

import sys
import os

# Ensure msg_baseline is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.config import ExperimentConfig, DATASET_CONFIGS, ALL_DATASETS


@dataclass
class SpikeRPGNNConfig(ExperimentConfig):
    """Configuration for SpikeRPGNN experiments.

    Inherits all fields from msg_baseline.config.ExperimentConfig and adds
    SpikeRPGNN-specific hyperparameters for the product manifold, LIF neuron,
    gate fusion, and ablation variants.

    Attributes:
        hidden_dim: Total representation dimension (overridden to 128 per paper §6.1).
        d_h: Hyperbolic component dimension (Poincaré ball).
        d_e: Euclidean component dimension.
        learnable_c: Whether curvature c is learnable (P2-3 curvature learning curve).
        c_init: Initial curvature c (if learnable_c=True).
        beta_leak: LIF leak decay factor (learnable initial value).
        gamma_input: LIF input scaling factor (learnable initial value).
        alpha_decay: Membrane potential decay constant.
        u_threshold: Spike firing threshold.
        delta_reset: Refractory reset amount.
        gate_init: Initial value for learnable gate fusion parameter lambda.
        ablation: Ablation variant name (controls model construction).
        use_dvm: Whether to use Differentiation-via-Manifold (truncated BPTT).
        backbone: GNN backbone type ("gin" or "gcn").
    """

    # ── Override baseline defaults ──────────────────────────────
    hidden_dim: int = 128           # Paper §6.1: total dim = 128
    backbone: str = "gin"           # Paper §6.1: GIN backbone

    # ── Product manifold dimensions ──────────────────────────────
    d_h: int = 32                   # Hyperbolic dimension (Poincaré ball)
    d_e: int = 96                   # Euclidean dimension (128 - 32)
    learnable_c: bool = False        # Whether curvature c is learnable (P2-3)
    c_init: float = 1.0            # Initial curvature c (if learnable_c=True)

    # ── LIF neuron dynamics ──────────────────────────────────
    beta_leak: float = 0.3          # Leak decay (initial for nn.Parameter)
    gamma_input: float = 0.5        # Input scaling (initial for nn.Parameter)
    alpha_decay: float = 0.9        # Membrane potential decay
    u_threshold: float = 1.0        # Spike threshold
    delta_reset: float = 1.0        # Refractory reset amount

    # ── Gate fusion ───────────────────────────────────────
    gate_init: float = 0.5          # Initial lambda for d = lambda*d_H + (1-lambda)*d_E

    # ── Node preprocessing MLP (P5 encoder repair) ──────────
    node_pre_dim: int = 0           # If >0: lift raw in_dim -> node_pre_dim via 2-layer MLP
                                    # BEFORE MSPE projection. Targets ENZYMES' 3-dim feature
                                    # poverty. 0 = disabled (keeps original behaviour).

    # ── Spike rate regularization ──────────────────────────────
    lambda_sr: float = 0.1            # Spike rate regularization coefficient (Eq. spike_rate_reg)
    r_target: float = 0.4            # Target spike rate r* (Eq. spike_rate_reg)
    surrogate_type: str = "atan"      # Surrogate gradient: "atan" (recommended) or "sigmoid"

    # ── Ablation control ───────────────────────────────────
    ablation: str = "full"          # One of: full, no_spikes, euclidean, no_frechet, no_dvm, no_meta

    # ── DvM control ───────────────────────────────────────
    use_dvm: bool = True            # Use Differentiation-via-Manifold
    use_metric_correction: bool = True  # Apply Riemannian metric correction (λ² factors)
    dvm_mode: str = "truncated"     # "truncated": O(M) memory, approx (Dφ)^T
                                    # "full_bptt_mc": O(T) memory, exact (Dφ)^T + MC
    truncate_memory: bool = True    # P0-B: in truncated mode, detach autograd graph of
                                    # steps outside the truncation window so the retained
                                    # graph spans only the last M steps => true O(M) *bytes*
                                    # (not just stored-state count). For T==M (start_idx==0)
                                    # this is a no-op, so the reported T=8/M=8 runs are
                                    # byte-for-byte identical to the pre-P0-B code path.

    # ── Override training defaults ─────────────────────────────
    epochs: int = 100               # Paper §6.1: 100 epochs

    def __post_init__(self):
        """Validate configuration consistency."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        assert self.d_h + self.d_e == self.hidden_dim, (
            f"d_h({self.d_h}) + d_e({self.d_e}) must equal hidden_dim({self.hidden_dim})"
        )
        assert self.ablation in {
            "full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta"
        }, f"Unknown ablation variant: {self.ablation}"
        assert self.dvm_mode in {"truncated", "full_bptt_mc"}, f"Unknown dvm_mode: {self.dvm_mode}"


def get_spikergnn_config(quick_test: bool = False, **overrides) -> SpikeRPGNNConfig:
    """Return SpikeRPGNN config with optional quick-test overrides.

    Args:
        quick_test: If True, reduce epochs/runs for fast debugging.
        **overrides: Additional field overrides passed to the dataclass.

    Returns:
        Fully initialized SpikeRPGNNConfig.
    """
    cfg = SpikeRPGNNConfig(**overrides)
    if quick_test:
        cfg.quick_test = True
        cfg.epochs = 3
        cfg.n_runs = 1
        cfg.train_episodes = 50
        cfg.episodes_per_epoch = 10
        cfg.test_episodes = 20
    return cfg
