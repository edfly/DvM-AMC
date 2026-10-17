"""
Quick test: Run 2 variants with 10 epochs each, compare training curves.
"""
import sys
sys.path.insert(0, '.')

import torch
import json
import numpy as np
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.train import run_single_experiment

def test_2_variants():
    """Test if 2 variants produce DIFFERENT training curves."""
    
    # Create base config
    base_config = SpikeRPGNNConfig(
        hidden_dim=128,
        num_layers=3,
        tau=20.0,
        use_dvm=True,
        ablation='full',
        epochs=10,  # 10 epochs for meaningful comparison
        episodes_per_epoch=10,
        test_episodes=50,
    )
    
    # Generate all ablation configs
    configs = get_all_ablation_configs(base_config)
    
    # Test 2 variants: 'full' vs 'no_spikes'
    variants_to_test = ['full', 'no_spikes']
    
    print("=" * 80)
    print("Quick Test: Comparing training curves for 2 variants (10 epochs each)...")
    print("=" * 80)
    
    results = {}
    
    for variant in variants_to_test:
        config = configs[variant]
        print(f"\n--- Variant: {variant} ---")
        print(f"  config.ablation = {config.ablation}")
        
        # Run experiment (1 run only)
        try:
            exp_result = run_single_experiment(
                dataset_name='ENZYMES',
                shot=1,
                run_id=999,  # Test run
                config=config,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            
            # Store results
            results[variant] = {
                'final_acc': exp_result['final_acc'],
                'best_acc': exp_result['best_acc'],
                'train_history': exp_result['train_history'],
                'total_spikes': exp_result.get('total_spikes', 0),
            }
            
            print(f"  ✓ Final acc: {exp_result['final_acc']:.4f}")
            print(f"  ✓ Best acc: {exp_result['best_acc']:.4f}")
            print(f"  ✓ Total spikes: {exp_result.get('total_spikes', 0)}")
            
            # Print training curve (last 5 epochs)
            train_history = exp_result['train_history']
            print(f"  Training curve (last 5 epochs):")
            for i, acc in enumerate(train_history[-5:], start=max(0, len(train_history)-5)):
                print(f"    Epoch {i+1}: {acc:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare training curves
    print("\n" + "=" * 80)
    print("Comparison:")
    print("=" * 80)
    
    if len(results) == 2:
        full_acc = results['full']['final_acc']
        no_spikes_acc = results['no_spikes']['final_acc']
        
        print(f"\n  Full SpikeRPGNN: {full_acc:.4f}")
        print(f"  w/o Spikes:       {no_spikes_acc:.4f}")
        print(f"  Difference:        {abs(full_acc - no_spikes_acc):.4f}")
        
        if abs(full_acc - no_spikes_acc) > 0.01:
            print(f"\n  ✓ Variants produce DIFFERENT results (good!)")
        else:
            print(f"\n  🚨 WARNING: Variants produce SIMILAR results!")
            print(f"     This means the ablation is NOT working correctly!")
        
        # Compare training curves
        full_curve = results['full']['train_history']
        no_spikes_curve = results['no_spikes']['train_history']
        
        print(f"\n  Training curve comparison (first 5 epochs):")
        for i in range(min(5, len(full_curve))):
            print(f"    Epoch {i+1}: Full={full_curve[i]:.4f}, NoSpikes={no_spikes_curve[i]:.4f}, Diff={abs(full_curve[i]-no_spikes_curve[i]):.4f}")
    
    # Save results
    with open('spikergnn_results/enzymes_2variants_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Test complete! Results saved to spikergnn_results/enzymes_2variants_test.json")
    print("=" * 80)

if __name__ == "__main__":
    test_2_variants()
