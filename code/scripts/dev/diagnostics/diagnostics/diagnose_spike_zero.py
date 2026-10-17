#!/usr/bin/env python3
"""
Diagnose why spike_counts=0 after 100 epoch training.
Run 10 epoch training, then inspect:
1. u_th_learnable value after training
2. w_input_mag magnitude during evaluate
3. u_mag accumulation during evaluate
4. Whether reset_membrane fixes the issue
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from spikergnn.train import run_single_experiment
from spikergnn.train import train_one_epoch, evaluate
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy
from msg_baseline.config import DATASET_CONFIGS
import torch.optim as optim
from tqdm import tqdm


def diagnose_spike_zero():
    """Run 10 epoch training and diagnose spike production."""
    print("=" * 70)
    print("DIAGNOSIS: spike_counts=0 after training")
    print("=" * 70)

    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=10,
        episodes_per_epoch=10,
        test_episodes=10,
    )

    # Set seed and load data
    seed = 42
    set_seed(seed)
    ds_cfg = DATASET_CONFIGS["ENZYMES"]
    dataset = TUGraphDataset(
        name="ENZYMES", tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio, seed=seed,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"],
        k_shot=1, q_query=config.q_query, seed=seed,
    )

    # Create model
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(config, in_dim=in_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    u_th_history = []
    spike_history = []

    # Training loop
    for epoch in tqdm(range(config.epochs), desc="Training"):
        model.train()
        train_metrics = train_one_epoch(model, sampler, optimizer, config, "cpu")
        scheduler.step()
        u_th = model.stmp.lif.u_th_learnable.item()
        u_th_history.append(u_th)
        spike_history.append(train_metrics['spike_count'])

    print(f"\n{'='*70}")
    print("POST-TRAINING MODEL STATE")
    print(f"{'='*70}")
    print(f"  u_th_learnable: {model.stmp.lif.u_th_learnable.item():.6f}")
    print(f"  u_th history: {[f'{v:.4f}' for v in u_th_history]}")
    print(f"  train spike history: {spike_history}")
    print(f"  _u_mag buffer: {model.stmp.lif._u_mag}")

    # ── Test 1: Evaluate WITHOUT reset (current behavior) ──
    print(f"\n{'='*70}")
    print("TEST 1: Evaluate WITHOUT reset_membrane (current behavior)")
    print(f"{'='*70}")

    model.eval()
    total_spikes_no_reset = 0
    with torch.no_grad():
        for ep in range(5):
            episode = sampler.sample_episode(mode="test")
            support_batch, support_labels, query_batch, query_labels = batch_episode_data(episode, device="cpu")

            support_z = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
            support_spikes = model.count_spikes()

            query_z = model.encode(query_batch.x, query_batch.edge_index, query_batch.batch)
            query_spikes = model.count_spikes()

            total_spikes_no_reset += query_spikes
            print(f"  Episode {ep+1}: support_spikes={support_spikes}, query_spikes={query_spikes}, "
                  f"_u_mag[:3]={model.stmp.lif._u_mag[:3].tolist() if model.stmp.lif._u_mag is not None else 'None'}")

    print(f"  Total spikes (no reset): {total_spikes_no_reset}")

    # ── Test 2: Evaluate WITH reset before each encode ──
    print(f"\n{'='*70}")
    print("TEST 2: Evaluate WITH reset_membrane before each encode")
    print(f"{'='*70}")

    total_spikes_with_reset = 0
    with torch.no_grad():
        for ep in range(5):
            episode = sampler.sample_episode(mode="test")
            support_batch, support_labels, query_batch, query_labels = batch_episode_data(episode, device="cpu")

            # Reset before support encoding
            model.stmp.lif.reset_membrane()
            support_z = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
            support_spikes = model.count_spikes()

            # Reset before query encoding
            model.stmp.lif.reset_membrane()
            query_z = model.encode(query_batch.x, query_batch.edge_index, query_batch.batch)
            query_spikes = model.count_spikes()

            total_spikes_with_reset += support_spikes + query_spikes
            print(f"  Episode {ep+1}: support_spikes={support_spikes}, query_spikes={query_spikes}")

    print(f"  Total spikes (with reset): {total_spikes_with_reset}")

    # ── Test 3: Inspect w_input_mag and u_mag during evaluate ──
    print(f"\n{'='*70}")
    print("TEST 3: Inspect internal values during evaluate")
    print(f"{'='*70}")

    model.stmp.lif.reset_membrane()
    model.eval()

    episode = sampler.sample_episode(mode="test")
    support_batch, support_labels, query_batch, query_labels = batch_episode_data(episode, device="cpu")

    # Manually run encode to inspect internals
    with torch.no_grad():
        x = query_batch.x
        edge_index = query_batch.edge_index
        batch = query_batch.batch

        # MSPE
        spike_seq, p_init = model.mspe(x, edge_index, batch)
        print(f"  MSPE spike_seq shape: {spike_seq.shape}")
        print(f"  MSPE spike_seq sum: {spike_seq.sum().item():.4f}")

        # STMP internals
        p_hyp, p_euc = p_init
        N = p_hyp.size(0)
        model.stmp._total_spike_count = 0

        # Check u_th
        u_th = model.stmp.lif.u_th_learnable.item()
        print(f"  u_th_learnable: {u_th:.6f}")

        # Check w_input_mag for each step
        for t in range(model.stmp.T):
            z_current = (p_hyp, p_euc)
            for rgc_layer in model.stmp.rgc_layers:
                z_current = rgc_layer(z_current, edge_index)
            z_hyp_new, z_euc_new = z_current

            rgc_features = model.stmp.manifold.merge(z_hyp_new, z_euc_new)
            x_step = rgc_features.unsqueeze(1)

            # Check w_input_mag before LIF
            w_input = model.stmp.lif.W(rgc_features)
            w_input_mag = w_input.norm(dim=-1)
            w_input_mag = w_input_mag / np.sqrt(model.stmp.lif.d_total)
            w_input_mag_clamped = torch.clamp(w_input_mag, max=u_th * 0.6)

            u_mag = model.stmp.lif._get_u_mag(N, x.device)
            u_mag = 0.9 * u_mag + w_input_mag_clamped
            u_mag = torch.clamp(u_mag, max=u_th * 1.5)

            fires = (u_mag >= u_th).sum().item()
            print(f"  Step {t}: w_input_mag mean={w_input_mag.mean().item():.4f}, "
                  f"clamped mean={w_input_mag_clamped.mean().item():.4f}, "
                  f"u_mag mean={u_mag.mean().item():.4f}, u_th={u_th:.4f}, "
                  f"fires={fires}/{N}")

            # Actually run LIF
            spikes, (p_hyp, p_euc) = model.stmp.lif(x_step, (p_hyp, p_euc))

    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    print(f"  u_th_learnable after training: {u_th:.6f}")
    print(f"  Spikes without reset: {total_spikes_no_reset}")
    print(f"  Spikes with reset: {total_spikes_with_reset}")

    if total_spikes_no_reset == 0 and total_spikes_with_reset > 0:
        print("\n  >> ROOT CAUSE: _u_mag not reset before evaluate")
        print("  >> FIX: Call reset_membrane() at start of evaluate()")
    elif total_spikes_no_reset == 0 and total_spikes_with_reset == 0:
        print("\n  >> ROOT CAUSE: w_input_mag too small or u_th too large after training")
        print(f"  >> w_input_mag mean: {w_input_mag.mean().item():.4f}, u_th: {u_th:.4f}")
    else:
        print("\n  >> Spikes are produced, issue may be in counting logic")


if __name__ == '__main__':
    diagnose_spike_zero()
