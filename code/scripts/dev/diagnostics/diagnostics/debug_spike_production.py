"""
Debug: Check if spikes are produced in both MSPE and STMP.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.model import create_spikergnn

def debug_spikes():
    """Debug spike production in MSPE and STMP."""
    
    # Create config
    config = SpikeRPGNNConfig(
        hidden_dim=128,
        num_layers=3,
        tau=20.0,
        use_dvm=True,
        ablation='full',
        epochs=2,
    )
    
    print("=" * 80)
    print("Debug: Checking spike production in MSPE and STMP...")
    print("=" * 80)
    
    # Create model
    model = create_spikergnn(config, in_dim=3)  # ENZYMES has 3 features
    model.train()
    
    # Check if model has spiking neurons
    print("\n1. Checking model structure...")
    print(f"  model.mspe.lif: {type(model.mspe.lif).__name__}")
    print(f"  model.stmp.lif: {type(model.stmp.lif).__name__}")
    print(f"  model.mspe.lif.ablation: {model.mspe.lif.ablation}")
    print(f"  model.stmp.lif.ablation: {model.stmp.lif.ablation}")
    
    # Create dummy input
    print("\n2. Running forward pass with dummy input...")
    batch_size = 5  # 5-way
    num_nodes = 20
    
    # Simulate one episode
    support_x = torch.randn(batch_size * 1 * 5, 3)  # 5-way 1-shot
    support_edge_index = torch.randint(0, num_nodes, (2, 50))
    support_batch = torch.zeros(num_nodes * batch_size * 1, dtype=torch.long)
    support_labels = torch.arange(batch_size).repeat_interleave(1)
    
    query_x = torch.randn(batch_size * 3 * 5, 3)  # 5-way 3-shot
    query_edge_index = torch.randint(0, num_nodes, (2, 50))
    query_batch = torch.zeros(num_nodes * batch_size * 3, dtype=torch.long)
    query_labels = torch.arange(batch_size).repeat_interleave(3)
    
    # Forward pass
    try:
        logits = model(
            support_x, support_edge_index, support_batch,
            support_labels, query_x, query_edge_index, query_batch,
            batch_size
        )
        
        print(f"  ✓ Forward pass succeeded")
        print(f"  ✓ Logits shape: {logits.shape}")
        
        # Check spike counts
        print(f"\n3. Checking spike counts...")
        print(f"  model.mspe.lif._spike_count: {model.mspe.lif._spike_count}")
        print(f"  model.stmp.lif._spike_count: {model.stmp.lif._spike_count}")
        print(f"  model.count_spikes(): {model.count_spikes()}")
        
        # Check if STMP's lif neuron is producing spikes
        print(f"\n4. Checking STMP.lif neuron...")
        lif = model.stmp.lif
        print(f"  lif.ablation: {lif.ablation}")
        print(f"  lif.u_th_learnable: {lif.u_th_learnable.item():.6f}")
        print(f"  lif._spike_count: {lif._spike_count}")
        print(f"  lif._spike_rate: {lif._spike_rate:.6f}")
        
        if lif._spike_count == 0:
            print(f"\n  🚨 WARNING: STMP.lif produced NO spikes!")
            print(f"     Possible reasons:")
            print(f"     1. Membrane potential never reached threshold")
            print(f"     2. Surrogate function output is too small")
            print(f"     3. Neuron is in 'no_spikes' mode (ablation={lif.ablation})")
            
            # Let's check the surrogate function
            print(f"\n5. Testing surrogate function...")
            test_input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
            test_output = lif.surrogate(test_input)
            print(f"  Test input: {test_input.tolist()}")
            print(f"  Test output: {test_output.tolist()}")
            print(f"  Surrogate beta: {lif.surrogate.beta}")
            
            # Check if threshold is too high
            print(f"\n6. Checking threshold...")
            print(f"  u_th_learnable: {lif.u_th_learnable.item():.6f}")
            print(f"  u_th (fixed): {lif.u_th:.6f}")
            
            # The issue might be that u_mag never exceeds u_th
            print(f"\n  💡 Suggestion: Try lowering u_th or increasing input magnitude")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)

if __name__ == "__main__":
    debug_spikes()
