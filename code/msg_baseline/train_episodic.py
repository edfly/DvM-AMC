"""
Episodic training loop for MSG* baseline.

Implements ProtoNet-style episodic training:
  1. Sample N-way K-shot episode
  2. Encode support + query graphs through MSG* encoder
  3. Compute Fréchet mean prototypes from support
  4. Classify queries by geodesic distance to prototypes
  5. Cross-entropy loss → Adam update
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .models import MSGStarEncoder, RiemannianProtoClassifier
from .data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from .config import ExperimentConfig, DATASET_CONFIGS, get_config
from .utils import set_seed, compute_accuracy, ResultTracker


def train_one_epoch(
    encoder: MSGStarEncoder,
    classifier: RiemannianProtoClassifier,
    sampler: EpisodeSampler,
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
) -> float:
    """
    Train for one epoch (a fixed number of episodes).

    Returns:
        Mean accuracy over this epoch's episodes.
    """
    encoder.train()
    classifier.train()  # classifier has no learnable params but keep consistent

    total_acc = 0.0
    n_episodes = config.episodes_per_epoch
    n_way = sampler.n_way

    for _ in range(n_episodes):
        # Sample episode
        episode = sampler.sample_episode(mode="train")
        supp_batch, supp_lbls, qry_batch, qry_lbls = batch_episode_data(
            episode, device=config.device
        )

        # Forward pass
        supp_z = encoder(supp_batch.x, supp_batch.edge_index, supp_batch.batch)
        qry_z = encoder(qry_batch.x, qry_batch.edge_index, qry_batch.batch)

        logits = classifier(qry_z, supp_z, supp_lbls, n_classes=n_way)

        # Loss & backward
        loss = nn.CrossEntropyLoss()(logits, qry_lbls)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), max_norm=config.grad_clip
        )

        optimizer.step()

        total_acc += compute_accuracy(logits, qry_lbls)

    return total_acc / n_episodes


@torch.no_grad()
def evaluate(
    encoder: MSGStarEncoder,
    classifier: RiemannianProtoClassifier,
    dataset: TUGraphDataset,
    config: ExperimentConfig,
    shot: int,
    seed_offset: int = 0,
) -> float:
    """
    Evaluate on meta-test set with fresh episode sampler.

    Returns:
        Mean accuracy over test_episodes.
    """
    encoder.eval()
    classifier.eval()

    ds_cfg = DATASET_CONFIGS[dataset.name]
    test_sampler = EpisodeSampler(
        dataset=dataset,
        n_way=ds_cfg["n_way"],
        k_shot=shot,
        q_query=config.q_query,
        seed=config.seed_base + seed_offset + 10000,
    )

    total_acc = 0.0
    n_eval = config.test_episodes
    n_way = test_sampler.n_way

    for _ in range(n_eval):
        episode = test_sampler.sample_episode(mode="test")
        supp_batch, supp_lbls, qry_batch, qry_lbls = batch_episode_data(
            episode, device=config.device
        )

        supp_z = encoder(supp_batch.x, supp_batch.edge_index, supp_batch.batch)
        qry_z = encoder(qry_batch.x, qry_batch.edge_index, qry_batch.batch)

        logits = classifier(qry_z, supp_z, supp_lbls, n_classes=n_way)
        total_acc += compute_accuracy(logits, qry_lbls)

    return total_acc / n_eval


def run_single_experiment(
    dataset_name: str,
    shot: int,
    run_id: int,
    config: ExperimentConfig,
) -> dict:
    """
    Run a single (dataset, shot, run_id) experiment end-to-end.

    Returns:
        Dict with 'accuracy' and optional training history.
    """
    ds_cfg = DATASET_CONFIGS[dataset_name]
    run_seed = config.seed_base + run_id * 100 + shot * 10
    set_seed(run_seed)

    print(f"\n{'='*60}")
    print(f"  {dataset_name} | {shot}-shot | Run {run_id+1}/{config.n_runs}")
    print(f"{'='*60}")

    # Load data
    dataset = TUGraphDataset(
        name=dataset_name,
        tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio,
        seed=config.seed_base,
    )
    ds_config = DATASET_CONFIGS[dataset_name]

    # Training sampler
    sampler = EpisodeSampler(
        dataset=dataset,
        n_way=ds_config["n_way"],
        k_shot=shot,
        q_query=config.q_query,
        seed=run_seed,
    )

    # Model
    encoder = MSGStarEncoder(
        in_dim=dataset.num_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        manifold_c=config.manifold_c,
        epsilon=config.epsilon,
        tau=config.tau,
    ).to(config.device)

    classifier = RiemannianProtoClassifier(manifold=encoder.manifold)

    optimizer = optim.Adam(
        encoder.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    # Training loop
    best_acc = 0.0
    epoch_accs = []
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        train_acc = train_one_epoch(
            encoder, classifier, sampler, optimizer, config, epoch
        )
        scheduler.step()

        # Periodic evaluation
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(encoder, classifier, dataset, config, shot)
            epoch_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{config.epochs} | "
                f"Train: {train_acc:.1%} | Test: {test_acc:.1%} | "
                f"Best: {best_acc:.1%} | {elapsed:.0f}s"
            )

    # Final evaluation
    final_acc = evaluate(encoder, classifier, dataset, config, shot)
    elapsed = time.time() - t0

    print(f"  >>> Final Accuracy: {final_acc:.1%} ({elapsed:.1f}s)")

    return {
        "accuracy": final_acc,
        "best_acc": best_acc,
        "epoch_accuracies": epoch_accs,
        "time_seconds": elapsed,
    }
