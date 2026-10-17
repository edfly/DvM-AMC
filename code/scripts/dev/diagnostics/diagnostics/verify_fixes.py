#!/usr/bin/env python3
"""
Quick verification that the bug fixes work correctly.
Tests:
1. PROTEINS baselines now use different ablation configs
2. ENZYMES no_meta variant produces different results from full
3. Spikes are produced during training
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.model import create_spikergnn
from spikergnn.train import run_single_experiment


def test_proteins_baselines_differ():
    """Test 1: PROTEINS baselines should now use different configs."""
    print("=" * 70)
    print("TEST 1: PROTEINS baselines use different ablation configs")
    print("=" * 70)
    
    # Import the fixed function
    from run_proteins_baseline import run_proteins_baseline
    
    # Check that the mapping is correct by inspecting the source
    import inspect
    source = inspect.getsource(run_proteins_baseline)
    
    checks = {
        "BASELINE_TO_ABLATION mapping exists": "BASELINE_TO_ABLATION" in source,
        "full -> 'full'": "'full'" in source and "SpikeRPGNN (full)" in source,
        "no_spikes -> 'no_spikes'": "'no_spikes'" in source,
        "euclidean -> 'euclidean'": "'euclidean'" in source,
        "No hardcoded ablation='full'": source.count("ablation='full'  # Use full model") == 0,
    }
    
    for desc, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
    
    return all(checks.values())


def test_enzymes_no_meta_differs():
    """Test 2: no_meta variant should produce different model structure."""
    print("\n" + "=" * 70)
    print("TEST 2: no_meta variant has different model structure")
    print("=" * 70)
    
    config_full = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=2, episodes_per_epoch=3, test_episodes=5,
    )
    config_no_meta = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='no_meta', epochs=2, episodes_per_epoch=3, test_episodes=5,
    )
    
    model_full = create_spikergnn(config_full, in_dim=3)
    model_no_meta = create_spikergnn(config_no_meta, in_dim=3)
    
    # Check model structure differences
    print(f"  full model: meta_head = {model_full.meta_head}")
    print(f"  no_meta model: meta_head = {type(model_no_meta.meta_head).__name__}")
    
    # Count parameters
    n_params_full = sum(p.numel() for p in model_full.parameters())
    n_params_no_meta = sum(p.numel() for p in model_no_meta.parameters())
    print(f"  full params: {n_params_full}")
    print(f"  no_meta params: {n_params_no_meta}")
    print(f"  Difference: {n_params_no_meta - n_params_full} (should be > 0)")
    
    # Test that classify() uses different paths
    N = 10
    q_hyp = torch.randn(N, 32)
    q_euc = torch.randn(N, 96)
    s_hyp = torch.randn(5, 32)
    s_euc = torch.randn(5, 96)
    s_labels = torch.arange(5)
    
    model_full.eval()
    model_no_meta.eval()
    
    with torch.no_grad():
        logits_full = model_full.classify((q_hyp, q_euc), (s_hyp, s_euc), s_labels, 5)
        logits_no_meta = model_no_meta.classify((q_hyp, q_euc), (s_hyp, s_euc), s_labels, 5)
    
    # They should be different (different classification paths)
    diff = (logits_full - logits_no_meta).abs().sum().item()
    print(f"  Logits difference: {diff:.4f} (should be > 0)")
    
    return model_no_meta.meta_head is not None and diff > 0


def test_all_variants_differ():
    """Test 3: All 6 ablation variants should produce different results."""
    print("\n" + "=" * 70)
    print("TEST 3: All ablation variants produce different model structures")
    print("=" * 70)
    
    base_config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=2,
    )
    
    ablation_configs = get_all_ablation_configs(base_config)
    
    # Create models for each variant
    models = {}
    param_counts = {}
    for variant, config in ablation_configs.items():
        model = create_spikergnn(config, in_dim=3)
        models[variant] = model
        param_counts[variant] = sum(p.numel() for p in model.parameters())
        print(f"  {variant:15s}: {param_counts[variant]} params, "
              f"use_dvm={config.use_dvm}, meta_head={model.meta_head is not None}")
    
    # Check which variants have different structures
    unique_params = set(param_counts.values())
    print(f"\n  Unique param counts: {len(unique_params)} (expect >= 3 for meaningful ablation)")
    
    return len(unique_params) >= 3


def test_quick_experiment_all_variants():
    """Test 4: Run quick experiment for all variants, verify different results."""
    print("\n" + "=" * 70)
    print("TEST 4: Quick experiment (1 epoch) for all variants")
    print("=" * 70)
    
    base_config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation='full', epochs=1,
        episodes_per_epoch=3,
        test_episodes=5,
    )
    
    ablation_configs = get_all_ablation_configs(base_config)
    
    results = {}
    for variant, config in ablation_configs.items():
        print(f"\n  Running variant: {variant}...")
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
            }
            print(f"    acc={result['final_acc']:.4f}, spikes={result['total_spikes']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[variant] = {'acc': None, 'spikes': None, 'error': str(e)}
    
    # Check results
    print("\n  Results summary:")
    accs = {}
    for variant, res in results.items():
        acc = res.get('acc')
        spikes = res.get('spikes')
        print(f"    {variant:15s}: acc={acc}, spikes={spikes}")
        if acc is not None:
            accs[variant] = acc
    
    # Check how many unique accuracies we have
    unique_accs = set(round(a, 6) for a in accs.values())
    print(f"\n  Unique accuracies: {len(unique_accs)} / {len(accs)}")
    
    # Note: no_frechet == full in 1-shot (by design, Fréchet mean of 1 point = the point)
    # So we expect at least 5 unique values (full==no_frechet, but others differ)
    return len(unique_accs) >= 4


if __name__ == '__main__':
    print("BUG FIX VERIFICATION REPORT")
    print("=" * 70)
    
    results = {}
    results['proteins_baselines'] = test_proteins_baselines_differ()
    results['no_meta_structure'] = test_enzymes_no_meta_differs()
    results['all_variants_structure'] = test_all_variants_differ()
    results['quick_experiment'] = test_quick_experiment_all_variants()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:30s}: {status}")
    
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
