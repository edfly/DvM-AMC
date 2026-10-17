#!/usr/bin/env python3
"""
Extended verification with more epochs to confirm variant differentiation.
Focus on the variants that were identical before: full vs no_dvm vs no_meta.
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.train import run_single_experiment


def test_extended_epochs():
    """Run 10 epochs for key variants to verify differentiation."""
    print("=" * 70)
    print("Extended test: 10 epochs for key variants")
    print("=" * 70)
    
    base_config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=10,
        episodes_per_epoch=10,
        test_episodes=20,
    )
    
    ablation_configs = get_all_ablation_configs(base_config)
    
    # Test the 4 variants that were identical before
    key_variants = ['full', 'no_dvm', 'no_meta', 'no_frechet']
    
    results = {}
    for variant in key_variants:
        config = ablation_configs[variant]
        print(f"\n  Running {variant} (10 epochs)...")
        try:
            result = run_single_experiment(
                dataset_name='ENZYMES',
                shot=1,
                run_id=0,
                config=config,
                device='cpu',
            )
            results[variant] = {
                'acc': result['final_acc'],
                'spikes': result['total_spikes'],
                'spike_history': result['spike_history'],
            }
            print(f"    acc={result['final_acc']:.4f}, spikes={result['total_spikes']}")
            print(f"    spike_history: {result['spike_history']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Results comparison:")
    print("=" * 70)
    for variant, res in results.items():
        print(f"  {variant:15s}: acc={res['acc']:.6f}, spikes={res['spikes']}")
    
    # Check uniqueness
    accs = {v: r['acc'] for v, r in results.items() if r['acc'] is not None}
    unique_accs = set(round(a, 6) for a in accs.values())
    print(f"\n  Unique accuracies: {len(unique_accs)} / {len(accs)}")
    
    # Expected: full == no_frechet (1-shot design), but no_dvm and no_meta should differ
    if 'full' in accs and 'no_frechet' in accs:
        frechet_same = abs(accs['full'] - accs['no_frechet']) < 0.001
        print(f"  full == no_frechet (expected for 1-shot): {frechet_same}")
    
    if 'full' in accs and 'no_dvm' in accs:
        dvm_differs = abs(accs['full'] - accs['no_dvm']) > 0.001
        print(f"  full != no_dvm (DvM effect): {dvm_differs}")
    
    if 'full' in accs and 'no_meta' in accs:
        meta_differs = abs(accs['full'] - accs['no_meta']) > 0.001
        print(f"  full != no_meta (meta-learning effect): {meta_differs}")
    
    return len(unique_accs) >= 3  # At least 3 unique values


if __name__ == '__main__':
    passed = test_extended_epochs()
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
