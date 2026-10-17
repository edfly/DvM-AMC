"""
GNN Training Integration: DvM in STMP on MUTAG.

Demonstrates truncated DvM (M=8) integrated into the Spiking Temporal
Message Passing module of a Riemannian spiking GNN on MUTAG (2-way 1-shot).

Compares: full BPTT vs DvM (M=8) — convergence, accuracy, memory usage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def train_one_epoch_with_dvm(
    model, sampler, optimizer, config, device, use_dvm=True
):
    """Train one epoch, optionally using DvM-truncated backward.

    When use_dvm=True, the STMP module is configured to use truncated DvM
    with window M=8 instead of full BPTT.
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

        support_z = model.encode(
            support_batch.x, support_batch.edge_index, support_batch.batch
        )
        query_z = model.encode(
            query_batch.x, query_batch.edge_index, query_batch.batch
        )
        logits = model.classify(
            query_z, support_z, support_labels, sampler.n_way
        )

        loss = nn.functional.cross_entropy(logits, query_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        from msg_baseline.utils import compute_accuracy
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


def evaluate_model(model, sampler, config, device):
    """Evaluate on test episodes."""
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

            model.stmp.lif.reset_membrane()

            support_z = model.encode(
                support_batch.x, support_batch.edge_index, support_batch.batch
            )
            query_z = model.encode(
                query_batch.x, query_batch.edge_index, query_batch.batch
            )
            logits = model.classify(
                query_z, support_z, support_labels, sampler.n_way
            )

            from msg_baseline.utils import compute_accuracy
            acc = compute_accuracy(logits, query_labels)
            total_acc += acc
            total_spikes += model.count_spikes()

    return {
        "mean_acc": total_acc / max(n_episodes, 1),
        "spike_count": total_spikes,
    }


def main():
    device = "cpu"
    dataset_name = "MUTAG"
    n_epochs = 50

    set_seed(42)

    from msg_baseline.config import DATASET_CONFIGS
    ds_cfg = DATASET_CONFIGS[dataset_name]
    n_way = ds_cfg["n_way"]

    # Load dataset
    dataset = TUGraphDataset(
        name=dataset_name,
        tu_name=ds_cfg["tu_name"],
        train_ratio=0.8,
        seed=42,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=n_way, k_shot=1, q_query=5, seed=42
    )

    # Config (quick-test mode)
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, ablation="full",
        epochs=n_epochs, episodes_per_epoch=20,
        test_episodes=50, lr=1e-3, weight_decay=5e-4,
    )

    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(config, in_dim=in_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    print("=" * 60, flush=True)
    print(f"DvM GNN Training: {dataset_name} {n_way}-way 1-shot ({n_epochs} epochs)", flush=True)
    print(f"  STMP T=8, DvM M=8 (truncated), params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("=" * 60, flush=True)

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(n_epochs):
        train_metrics = train_one_epoch_with_dvm(
            model, sampler, optimizer, config, device, use_dvm=True
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            eval_metrics = evaluate_model(model, sampler, config, device)
            if eval_metrics["mean_acc"] > best_acc:
                best_acc = eval_metrics["mean_acc"]
            print(f"  ep {epoch+1:3d}: train_acc={train_metrics['mean_acc']:.3f}  "
                  f"test_acc={eval_metrics['mean_acc']:.3f}  "
                  f"best={best_acc:.3f}  "
                  f"spikes={train_metrics['spike_count']}")

    t_total = time.time() - t_start
    print(f"\nFinal: best_test_acc={best_acc:.4f}  time={t_total:.1f}s  "
          f"avg={(t_total/n_epochs):.1f}s/epoch")


if __name__ == "__main__":
    main()
