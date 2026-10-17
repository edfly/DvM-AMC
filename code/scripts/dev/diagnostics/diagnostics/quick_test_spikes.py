"""
Quick test: Check if spikes are produced after lowering threshold.
"""
import sys
sys.path.insert(0, '.')

import torch
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.train import run_single_experiment

# Create config
config = SpikeRPGNNConfig(
    hidden_dim=128,
    num_layers=3,
    tau=20.0,
    use_dvm=True,
    ablation='full',  # Test full variant
    epochs=2,  # Just 2 epochs for quick test
    episodes_per_epoch=5,
    test_episodes=20,
)

print("=" * 80)
print("Quick Test: Checking if spikes are produced (threshold=0.3)...")
print("=" * 80)

# Run experiment (1 run only)
print("\nRunning experiment (2 epochs, 5 episodes per epoch)...")
try:
    exp_result = run_single_experiment(
        dataset_name='ENZYMES',
        shot=1,
        run_id=999,  # Test run
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"\n  ✓ Final acc: {exp_result['final_acc']:.4f}")
    print(f"  ✓ Best acc: {exp_result['best_acc']:.4f}")
    print(f"  ✓ Total spikes: {exp_result.get('total_spikes', 0)}")
    
    if exp_result.get('total_spikes', 0) > 0:
        print(f"\n  ✓✓ SPIKES PRODUCED! Fix worked!")
        print(f"     Spike count: {exp_result['total_spikes']}")
    else:
        print(f"\n  🚨 WARNING: Still no spikes produced!")
        print(f"     Need further debugging.")
    
except Exception as e:
    print(f"\n  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Quick test complete!")
print("=" * 80)
