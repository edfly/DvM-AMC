#!/usr/bin/env python3
"""
Diagnose why spike_counts=0 in experiment results.

Checks:
1. Whether LIF neuron produces spikes with default config (untrained model)
2. Whether u_th_learnable grows during training
3. Whether evaluate() correctly reports spikes
4. Whether the persistent _u_mag buffer causes issues across episodes
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def test_untrained_spikes():
    """Test 1: Does an untrained model produce spikes?"""
    print("=" * 70)
    print("TEST 1: Untrained model spike production")
    print("=" * 70)
    
    config = SpikeRPGNNConfig(
        hidden_dim=128,
        num_layers=3,
        tau=20.0,
        use_dvm=True,
        ablation='full',
        epochs=100,
    )
    
    model = create_spikergnn(config, in_dim=3)  # ENZYMES has 3 features
    model.train()
    
    # Check initial u_th_learnable
    u_th_init = model.stmp.lif.u_th_learnable.item()
    print(f"  Initial u_th_learnable: {u_th_init:.6f}")
    print(f"  Initial beta: {model.stmp.lif.beta.item():.6f}")
    print(f"  Initial gamma: {model.stmp.lif.gamma.item():.6f}")
    
    # Simulate forward pass with realistic input
    N = 30  # nodes
    x = torch.randn(N, 3)
    edge_index = torch.randint(0, N, (2, 60))
    batch = torch.zeros(N, dtype=torch.long)
    
    # Forward
    z = model.encode(x, edge_index, batch)
    spike_count = model.count_spikes()
    
    print(f"\n  Forward pass result:")
    print(f"    Spike count: {spike_count}")
    print(f"    LIF _spike_count: {model.stmp.lif._spike_count}")
    print(f"    LIF _spike_rate: {model.stmp.lif._spike_rate:.6f}")
    print(f"    LIF _u_mag after forward: {model.stmp.lif._u_mag}")
    print(f"    u_th_learnable after forward: {model.stmp.lif.u_th_learnable.item():.6f}")
    
    return spike_count > 0


def test_eval_mode_spikes():
    """Test 2: Does model produce spikes in eval mode?"""
    print("\n" + "=" * 70)
    print("TEST 2: Eval mode spike production")
    print("=" * 70)
    
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=100,
    )
    
    model = create_spikergnn(config, in_dim=3)
    model.eval()
    
    # Reset membrane potential
    model.stmp.lif.reset_membrane()
    
    N = 30
    x = torch.randn(N, 3)
    edge_index = torch.randint(0, N, (2, 60))
    batch = torch.zeros(N, dtype=torch.long)
    
    with torch.no_grad():
        z = model.encode(x, edge_index, batch)
    
    spike_count = model.count_spikes()
    print(f"  Eval mode spike count: {spike_count}")
    print(f"  LIF _spike_count: {model.stmp.lif._spike_count}")
    print(f"  LIF _u_mag: {model.stmp.lif._u_mag}")
    
    return spike_count > 0


def test_training_u_th_growth():
    """Test 3: Does u_th_learnable grow during training?"""
    print("\n" + "=" * 70)
    print("TEST 3: u_th_learnable growth during short training")
    print("=" * 70)
    
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=5,
        episodes_per_epoch=5,
    )
    
    model = create_spikergnn(config, in_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    u_th_history = [model.stmp.lif.u_th_learnable.item()]
    spike_history = []
    
    for step in range(20):
        model.train()
        model.stmp.lif.reset_membrane()
        
        N = 30
        x = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 60))
        batch = torch.zeros(N, dtype=torch.long)
        labels = torch.randint(0, 5, (1,))
        
        optimizer.zero_grad()
        z = model.encode(x, edge_index, batch)
        # Dummy loss to create gradient
        loss = z[0].sum() + z[1].sum()
        loss.backward()
        optimizer.step()
        
        u_th = model.stmp.lif.u_th_learnable.item()
        u_th_history.append(u_th)
        spike_history.append(model.count_spikes())
    
    print(f"  u_th_learnable history: {[f'{v:.4f}' for v in u_th_history]}")
    print(f"  Spike count history: {spike_history}")
    print(f"  u_th change: {u_th_history[0]:.4f} -> {u_th_history[-1]:.4f} (delta={u_th_history[-1]-u_th_history[0]:.4f})")
    
    return u_th_history[-1] < 5.0  # Not exploded


def test_u_mag_persistence():
    """Test 4: Check if _u_mag buffer persists across episodes (bug?)"""
    print("\n" + "=" * 70)
    print("TEST 4: _u_mag persistence across forward calls")
    print("=" * 70)
    
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=100,
    )
    
    model = create_spikergnn(config, in_dim=3)
    model.eval()
    model.stmp.lif.reset_membrane()
    
    N = 30
    edge_index = torch.randint(0, N, (2, 60))
    batch = torch.zeros(N, dtype=torch.long)
    
    print(f"  Initial _u_mag: {model.stmp.lif._u_mag}")
    
    for i in range(3):
        x = torch.randn(N, 3)
        with torch.no_grad():
            z = model.encode(x, edge_index, batch)
        print(f"  After episode {i+1}: _u_mag[0:5]={model.stmp.lif._u_mag[:5].tolist() if model.stmp.lif._u_mag is not None else 'None'}, spikes={model.count_spikes()}")
    
    return True


def test_realistic_experiment():
    """Test 5: Run a realistic mini-experiment like the actual training."""
    print("\n" + "=" * 70)
    print("TEST 5: Realistic mini-experiment (2 epochs, 5 episodes each)")
    print("=" * 70)
    
    from spikergnn.train import run_single_experiment
    
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=2,
        episodes_per_epoch=5,
        test_episodes=10,
    )
    
    try:
        result = run_single_experiment(
            dataset_name='ENZYMES',
            shot=1,
            run_id=0,
            config=config,
            device='cpu',
        )
        print(f"  final_acc: {result['final_acc']:.4f}")
        print(f"  total_spikes: {result['total_spikes']}")
        print(f"  spike_history: {result['spike_history']}")
        return result['total_spikes'] > 0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("SPIKE DIAGNOSTIC REPORT")
    print("=" * 70)
    
    results = {}
    results['untrained'] = test_untrained_spikes()
    results['eval_mode'] = test_eval_mode_spikes()
    results['u_th_growth'] = test_training_u_th_growth()
    results['u_mag_persistence'] = test_u_mag_persistence()
    results['realistic'] = test_realistic_experiment()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:25s}: {status}")
    
    if not results['untrained']:
        print("\n  >> Untrained model produces NO spikes — LIF dynamics broken")
    if not results['eval_mode']:
        print("\n  >> Eval mode produces NO spikes — surrogate or threshold issue")
    if not results['u_th_growth']:
        print("\n  >> u_th_learnable exploded — needs regularization")
    if not results['realistic']:
        print("\n  >> Realistic experiment produces NO spikes — confirmed bug")
