"""
Episodic training loop for SpikeRPGNN.

Implements ProtoNet-style N-way K-shot meta-learning training with:
  - Adam optimizer + CosineAnnealingLR scheduler
  - Gradient clipping (max_norm=5.0)
  - DvM-aware backward pass (handled by DvMFunction inside STMP)
  - Spike count tracking for energy estimation
  - Cross-entropy loss for episodic classification

Reuses msg_baseline.data_utils for data loading and episode sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional

from tqdm import tqdm

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN, create_spikergnn


class EpisodicTrainer:
    """Episodic trainer for SpikeRPGNN meta-learning experiments.

    Wraps the training, evaluation, and experiment execution logic into
    a single class for convenient use. Delegates to the module-level
    train_one_epoch, evaluate, and run_single_experiment functions.

    Attributes:
        model: SpikeRPGNN model instance.
        config: SpikeRPGNNConfig instance.
        device: Device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: SpikeRPGNN,
        config: SpikeRPGNNConfig,
        device: str = "cpu",
    ):
        """Initialize EpisodicTrainer.

        Args:
            model: SpikeRPGNN model instance.
            config: SpikeRPGNNConfig instance.
            device: Device to run on.
        """
        self.model = model
        self.config = config
        self.device = device

    def train_one_epoch(
        self,
        sampler: EpisodeSampler,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch using episodic sampling.

        Args:
            sampler: EpisodeSampler instance.
            optimizer: PyTorch optimizer.

        Returns:
            Dict with 'mean_acc', 'mean_loss', 'spike_count'.
        """
        return train_one_epoch(
            model=self.model,
            sampler=sampler,
            optimizer=optimizer,
            config=self.config,
            device=self.device,
        )

    def evaluate(
        self,
        sampler: EpisodeSampler,
    ) -> Dict[str, float]:
        """Evaluate model on test episodes.

        Args:
            sampler: EpisodeSampler instance.

        Returns:
            Dict with 'mean_acc', 'spike_count'.
        """
        return evaluate(
            model=self.model,
            sampler=sampler,
            config=self.config,
            device=self.device,
        )

    def run(
        self,
        dataset_name: str,
        n_way: int,
        k_shot: int,
    ) -> Dict:
        """Run a single experiment.

        Args:
            dataset_name: Name of the TUDataset (e.g., "MUTAG").
            n_way: Number of classes per episode.
            k_shot: Number of support examples per class.

        Returns:
            Dict with experiment results.
        """
        return run_single_experiment(
            dataset_name=dataset_name,
            shot=k_shot,
            run_id=0,
            config=self.config,
            device=self.device,
        )


def train_one_epoch(
    model: SpikeRPGNN,
    sampler: EpisodeSampler,
    optimizer: torch.optim.Optimizer,
    config: SpikeRPGNNConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train for one epoch using episodic sampling.

    Args:
        model: SpikeRPGNN model.
        sampler: EpisodeSampler instance.
        optimizer: PyTorch optimizer.
        config: SpikeRPGNNConfig instance.
        device: Device to run on.

    Returns:
        Dict with 'mean_acc', 'mean_loss', 'spike_count'.
    """
    model.train()
    total_acc = 0.0
    total_loss = 0.0
    total_spikes = 0
    n_episodes = config.episodes_per_epoch

    for ep in range(n_episodes):
        episode = sampler.sample_episode(mode="train")
        support_batch, support_labels, query_batch, query_labels = batch_episode_data(
            episode, device=device
        )

        optimizer.zero_grad()

        # Encode support set
        support_z = model.encode(
            support_batch.x, support_batch.edge_index, support_batch.batch
        )

        # Encode query set
        query_z = model.encode(
            query_batch.x, query_batch.edge_index, query_batch.batch
        )

        # Classify
        logits = model.classify(
            query_z, support_z, support_labels, sampler.n_way
        )

        # Loss
        loss = nn.functional.cross_entropy(logits, query_labels)

        # Backward
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

        optimizer.step()

        # Metrics
        acc = compute_accuracy(logits.detach(), query_labels)
        total_acc += acc
        total_loss += loss.item()
        total_spikes += model.count_spikes()

    n_ep = max(n_episodes, 1)
    return {
        "mean_acc": total_acc / n_ep,
        "mean_loss": total_loss / n_ep,
        "spike_count": total_spikes,
    }


def evaluate(
    model: SpikeRPGNN,
    sampler: EpisodeSampler,
    config: SpikeRPGNNConfig,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on test episodes.

    Args:
        model: SpikeRPGNN model.
        sampler: EpisodeSampler instance.
        config: SpikeRPGNNConfig instance.
        device: Device to run on.

    Returns:
        Dict with 'mean_acc', 'spike_count'.
    """
    model.eval()
    total_acc = 0.0
    total_spikes = 0
    n_episodes = config.test_episodes

    with torch.no_grad():
        for ep in range(n_episodes):
            episode = sampler.sample_episode(mode="test")
            support_batch, support_labels, query_batch, query_labels = batch_episode_data(
                episode, device=device
            )

            # Reset membrane potential before each episode to ensure
            # clean state (persistent _u_mag buffer carries over from training)
            model.stmp.lif.reset_membrane()

            # Encode and classify
            support_z = model.encode(
                support_batch.x, support_batch.edge_index, support_batch.batch
            )
            support_spikes = model.count_spikes()

            query_z = model.encode(
                query_batch.x, query_batch.edge_index, query_batch.batch
            )
            query_spikes = model.count_spikes()

            logits = model.classify(
                query_z, support_z, support_labels, sampler.n_way
            )

            acc = compute_accuracy(logits, query_labels)
            total_acc += acc
            # Count spikes from both support and query encoding
            total_spikes += support_spikes + query_spikes

    n_ep = max(n_episodes, 1)
    return {
        "mean_acc": total_acc / n_ep,
        "spike_count": total_spikes,
    }


def run_single_experiment(
    dataset_name: str,
    shot: int,
    run_id: int,
    config: SpikeRPGNNConfig,
    device: str = "cpu",
) -> Dict:
    """Run a single experiment (one dataset, one shot, one run).

    Args:
        dataset_name: Name of the TUDataset (e.g., "MUTAG").
        shot: Number of support examples per class (1 or 5).
        run_id: Run index for reproducibility.
        config: SpikeRPGNNConfig instance.
        device: Device to run on.

    Returns:
        Dict with experiment results including accuracy, spike counts,
        and training history.
    """
    from msg_baseline.config import DATASET_CONFIGS

    ds_cfg = DATASET_CONFIGS[dataset_name]
    n_way = ds_cfg["n_way"]

    # Set seed for reproducibility
    seed = config.seed_base + run_id * 100 + shot * 10
    set_seed(seed)

    # Load dataset
    dataset = TUGraphDataset(
        name=dataset_name,
        tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio,
        seed=seed,
    )

    # Create episode sampler
    sampler = EpisodeSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=shot,
        q_query=config.q_query,
        seed=seed,
    )

    # Create model
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(config, in_dim=in_dim)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    # Training loop
    best_acc = 0.0
    train_history = []
    spike_history = []

    for epoch in tqdm(range(config.epochs), desc=f"{dataset_name} {shot}-shot run{run_id}"):
        train_metrics = train_one_epoch(model, sampler, optimizer, config, device)
        scheduler.step()

        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            eval_metrics = evaluate(model, sampler, config, device)
            if eval_metrics["mean_acc"] > best_acc:
                best_acc = eval_metrics["mean_acc"]
            train_history.append({
                "epoch": epoch,
                "train_acc": train_metrics["mean_acc"],
                "train_loss": train_metrics["mean_loss"],
                "test_acc": eval_metrics["mean_acc"],
            })

        spike_history.append(train_metrics["spike_count"])

    # Final evaluation
    final_metrics = evaluate(model, sampler, config, device)

    return {
        "dataset": dataset_name,
        "shot": shot,
        "run_id": run_id,
        "n_way": n_way,
        "final_acc": final_metrics["mean_acc"],
        "best_acc": best_acc,
        "total_spikes": final_metrics["spike_count"],
        # ── P1-2 honesty: separated spike accounting ──
        # stmp_spikes: STMP recurrent stage, all T steps (model._last_spike_count).
        # mspe_spikes: MSPE encoder LIF (model.mspe.lif._spike_count).
        # These let the energy table be reproduced with the MSPE+STMP split
        # that the paper (tab:energy_pareto, Sec.6.3) discloses.
        "stmp_spikes": int(getattr(model, "_last_spike_count", 0) or 0),
        "mspe_spikes": int(
            getattr(getattr(model, "mspe", None), "lif", None)
            and (getattr(model.mspe.lif, "_spike_count", 0) or 0)
        ),
        "spike_history": spike_history,
        "train_history": train_history,
        "config": {
            "ablation": config.ablation,
            "d_h": config.d_h,
            "d_e": config.d_e,
            "T": config.T,
            "epochs": config.epochs,
        },
    }
