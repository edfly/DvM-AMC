"""
快速测试脚本：验证ENZYMES消融实验的6个变体是否产生不同结果
只运行1个epoch，1次运行，用于快速验证bug修复
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import torch

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.train import run_single_experiment

def test_all_variants_quick(n_runs=1, epochs=1, device="cpu"):
    """
    快速测试所有变体，只运行少量epoch和次数
    """
    print("=" * 80)
    print("Quick test: Running ALL 6 ablation variants (1 epoch, 1 run each)...")
    print("=" * 80)
    
    # Base config
    base_config = SpikeRPGNNConfig(
        hidden_dim=128,
        num_layers=3,
        tau=20.0,
        use_dvm=True,
        ablation="full",
        epochs=epochs,  # Only 1 epoch for quick test
    )
    
    ablation_configs = get_all_ablation_configs(base_config)
    
    results = {}
    dataset_name = "ENZYMES"
    shot = 1
    
    for variant, config in ablation_configs.items():
        print(f"\n{'=' * 80}")
        print(f"Variant: {variant}")
        print(f"{'=' * 80}")
        print(f"  config.ablation = {config.ablation}")
        print(f"  config.use_dvm = {config.use_dvm}")
        
        start_time = time.time()
        
        try:
            exp_result = run_single_experiment(
                dataset_name=dataset_name,
                shot=shot,
                run_id=999,  # Test run ID
                config=config,
                device=device,
            )
            acc = exp_result['final_acc']
            spike_count = exp_result.get("total_spikes", 0)
            elapsed = time.time() - start_time
            
            results[variant] = {
                "acc": acc,
                "spike_count": spike_count,
                "time": elapsed,
            }
            
            print(f"  ✓ acc = {acc:.4f}")
            print(f"  ✓ total_spikes = {spike_count}")
            print(f"  ✓ time = {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results[variant] = {"acc": None, "error": str(e)}
        
        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Checking if all variants produce DIFFERENT results...")
    print(f"{'=' * 80}")
    
    accs = {}
    for variant, res in results.items():
        if res.get("acc") is not None:
            accs[variant] = res["acc"]
            print(f"  {variant:20s}: acc = {res['acc']:.4f}, spikes = {res.get('spike_count', 'N/A')}")
    
    # Check if all accuracies are different
    acc_values = list(accs.values())
    acc_set = set([round(a, 4) for a in acc_values])
    
    print(f"\n  Number of unique accuracies: {len(acc_set)}")
    if len(acc_set) == len(accs):
        print("  ✓ ALL VARIANTS PRODUCE DIFFERENT RESULTS! (Good!)")
    else:
        print("  🚨 WARNING: Some variants have IDENTICAL accuracies!")
        print("     This means the ablation is NOT working correctly!")
    
    # Save results
    output_file = "spikergnn_results/enzymes_ablation_test.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    test_all_variants_quick(n_runs=1, epochs=1, device="cpu")
