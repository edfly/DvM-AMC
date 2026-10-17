"""
Configuration constants for MSG* baseline experiments.

All hyperparameters follow the experimental protocol defined in
SpikeRPGNN v19 §6 (Experiments section) and MSG original paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExperimentConfig:
    """Global experiment configuration."""

    # ── Model architecture ──────────────────────────────────────
    hidden_dim: int = 32          # MSG original representation dim
    num_layers: int = 3           # GIN/GCN backbone layers
    manifold_c: float = 1.0       # Poincaré curvature (c > 0 => H^(-c))

    # ── Spiking parameters (from MSG paper) ────────────────────
    T: int = 8                    # Time steps (MSG grid: {5,15}, we use 8)
    epsilon: float = 0.1          # Exponential map step size (per-step scale)
    tau: float = 20.0             # Membrane time constant (ms)
    v_threshold: float = 1.0      # IF firing threshold
    v_reset: float = 0.0          # Reset potential after spike
    surrogate_beta: float = 5.0   # Surrogate gradient steepness

    # ── Training hyperparameters ────────────────────────────────
    epochs: int = 50              # Training epochs (reduced from 100 for baseline)
    lr: float = 0.001             # Adam learning rate
    weight_decay: float = 5e-4    # L2 regularization
    train_episodes: int = 5000    # Total meta-training episodes
    episodes_per_epoch: int = 30  # Episodes sampled per epoch (reduced from 50 for speed)
    grad_clip: float = 5.0        # Gradient clipping norm

    # ── Evaluation protocol ─────────────────────────────────────
    test_episodes: int = 50       # Meta-test episodes per evaluation (reduced from 100)
    n_runs: int = 5               # Independent runs (for mean ± std)
    q_query: int = 5              # Query graphs per class per episode
    train_ratio: float = 0.80     # Train/test split ratio

    # ── DvM truncation ─────────────────────────────────────────
    dv_m_window: int = 8          # DvM backward truncation window

    # ── Reproducibility ─────────────────────────────────────────
    seed_base: int = 42
    device: str = "cpu"           # CPU-only (no GPU assumed)

    # ── Quick-test mode override ────────────────────────────────
    quick_test: bool = False      # Set True for fast debugging


# ── Dataset-specific configurations ──────────────────────────────

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MUTAG": {
        "n_way": 2,
        "shots": [1, 5],
        "n_graphs": 188,
        "n_classes": 2,
        "tu_name": "MUTAG",
    },
    "PROTEINS": {
        "n_way": 2,
        "shots": [1, 5],
        "n_graphs": 1113,
        "n_classes": 2,
        "tu_name": "PROTEINS",
    },
    "DD": {
        "n_way": 2,
        "shots": [1, 5],
        "n_graphs": 1178,
        "n_classes": 2,
        "tu_name": "DD",
    },
    "NCI1": {
        # Registered here so the single source of truth (DATASET_CONFIGS /
        # ALL_DATASETS) is consistent across every entry point: spikergnn.
        # run_experiment (argv validation), msg_baseline.run_experiment,
        # run_final_matrix, and train_gnn_baselines. Previously NCI1 was
        # listed in rerun_28configs.DATASETS but absent from this config, so
        # (a) the No-Spikes arm crashed on argv validation and (b)
        # train_gnn_baselines silently DROPPED the NCI1 GNN baselines via its
        # `if d in DATASET_CONFIGS` filter. TUDataset supports NCI1 directly.
        "n_way": 2,
        "shots": [1, 5],
        "n_graphs": 4110,
        "n_classes": 2,
        "tu_name": "NCI1",
    },
    "ENZYMES": {
        "n_way": 5,
        "shots": [1, 5],
        "n_graphs": 600,
        "n_classes": 6,
        "tu_name": "ENZYMES",
    },
    "REDDIT-M5K": {
        "n_way": 5,
        "shots": [1, 5],
        "n_graphs": 4999,
        "n_classes": 5,
        "tu_name": "REDDIT-MULTI-5K",
    },
}

# All dataset names in canonical order
ALL_DATASETS = list(DATASET_CONFIGS.keys())


def get_config(quick_test: bool = False) -> ExperimentConfig:
    """Return config with optional quick-test overrides."""
    cfg = ExperimentConfig()
    if quick_test:
        cfg.quick_test = True
        cfg.epochs = 3
        cfg.n_runs = 1
        cfg.train_episodes = 50
        cfg.episodes_per_epoch = 10
        cfg.test_episodes = 20
    return cfg
