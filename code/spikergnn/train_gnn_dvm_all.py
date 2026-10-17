"""
GNN Training Integration: DvM in STMP on all TU datasets.

Runs truncated DvM (M=8) on ENZYMES, PROTEINS, DD sequentially.
Logs results to gnn_train_all.log.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy
from msg_baseline.config import DATASET_CONFIGS
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def train_one_epoch(model, sampler, optimizer, config, device):
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
        support_z = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
        query_z = model.encode(query_batch.x, query_batch.edge_index, query_batch.batch)
        logits = model.classify(query_z, support_z, support_labels, sampler.n_way)
        loss = nn.functional.cross_entropy(logits, query_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
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
            support_z = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
            query_z = model.encode(query_batch.x, query_batch.edge_index, query_batch.batch)
            logits = model.classify(query_z, support_z, support_labels, sampler.n_way)
            acc = compute_accuracy(logits, query_labels)
            total_acc += acc
            total_spikes += model.count_spikes()

    return {
        "mean_acc": total_acc / max(n_episodes, 1),
        "spike_count": total_spikes,
    }


def train_on_dataset(dataset_name, n_epochs=50):
    device = "cpu"
    ds_cfg = DATASET_CONFIGS[dataset_name]
    n_way = ds_cfg["n_way"]

    set_seed(42)

    dataset = TUGraphDataset(
        name=dataset_name, tu_name=ds_cfg["tu_name"],
        train_ratio=0.8, seed=42,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=n_way, k_shot=1, q_query=5, seed=42
    )

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

    header = f"DvM GNN Training: {dataset_name} {n_way}-way 1-shot ({n_epochs} epochs)"
    print("=" * 60, flush=True)
    print(header, flush=True)
    print(f"  Graphs={ds_cfg['n_graphs']}, Features={in_dim}, Classes={ds_cfg['n_classes']}", flush=True)
    print(f"  STMP T=8, DvM M=8, params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("=" * 60, flush=True)

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(model, sampler, optimizer, config, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            eval_metrics = evaluate_model(model, sampler, config, device)
            if eval_metrics["mean_acc"] > best_acc:
                best_acc = eval_metrics["mean_acc"]
            print(f"  ep {epoch+1:3d}: train_acc={train_metrics['mean_acc']:.3f}  "
                  f"test_acc={eval_metrics['mean_acc']:.3f}  "
                  f"best={best_acc:.3f}  "
                  f"spikes={train_metrics['spike_count']}", flush=True)

    t_total = time.time() - t_start
    result = f"  => best_test_acc={best_acc:.4f}  time={t_total:.1f}s  avg={t_total/n_epochs:.1f}s/epoch"
    print(result, flush=True)
    print(flush=True)
    return best_acc, t_total


def main():
    datasets = ["ENZYMES", "PROTEINS", "DD"]
    n_epochs = 50

    print(f"\n{'#'*60}")
    print(f"# DvM GNN Multi-Dataset Training")
    print(f"# Datasets: {datasets}")
    print(f"# Epochs: {n_epochs}, 1-shot")
    print(f"# Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n", flush=True)

    results = {}
    t_total = time.time()

    for ds_name in datasets:
        print(f"\n>>> Starting {ds_name} ...\n", flush=True)
        try:
            best_acc, ds_time = train_on_dataset(ds_name, n_epochs)
            results[ds_name] = {"best_acc": best_acc, "time": ds_time, "status": "OK"}
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[ds_name] = {"best_acc": 0.0, "time": 0.0, "status": f"ERROR: {e}"}

    t_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for ds_name, r in results.items():
        if r["status"] == "OK":
            print(f"  {ds_name:12s}: best_acc={r['best_acc']:.4f}  time={r['time']:.0f}s")
        else:
            print(f"  {ds_name:12s}: {r['status']}")
    print(f"  Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
