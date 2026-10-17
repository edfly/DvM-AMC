"""
Debug script: Check if spiking neurons actually produce spikes during training.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.model import create_spikergnn
from spikergnn.sampler import EpisodicSampler

def test_spikes_during_forward():
    """Test if spikes are actually produced during forward pass."""
    
    # Create config for ENZYMES (6 classes, 5-way valid)
    config = SpikeRPGNNConfig(
        hidden_dim=128,
        num_layers=3,
        tau=20.0,
        use_dvm=True,
        ablation='full',  # Test full variant
        epochs=2,  # Just 2 epochs for debugging
        episodes_per_epoch=2,  # Few episodes
        test_episodes=10,
    )
    
    print("=" * 80)
    print("Debug: Checking if spikes are produced during forward pass...")
    print("=" * 80)
    
    # Create model
    model = create_spikergnn(config, in_dim=3)  # ENZYMES has 3 features
    model.train()
    
    # Check if model has spiking neurons
    has_spikes = False
    for name, module in model.named_modules():
        if 'ManifoldLIFNeuron' in str(type(module)):
            has_spikes = True
            print(f"  Found spiking neuron: {name}")
    
    if not has_spikes:
        print("  ❌ ERROR: Model doesn't have any spiking neurons!")
        return
    
    # Create dummy data (simulate one episode)
    print("\n  Testing forward pass with dummy data...")
    batch_size = 5  # 5-way
    num_support = 1  # 1-shot
    num_query = 3  # 3 query per class
    num_nodes_per_graph = 20
    
    # Simulate support set
    support_x = torch.randn(batch_size * num_support * 5, 3)
    support_edge_index = torch.randint(0, num_nodes_per_graph, (2, 50))
    support_batch = torch.zeros(num_nodes_per_graph * batch_size * num_support, dtype=torch.long)
    support_labels = torch.arange(batch_size).repeat_interleave(num_support)
    
    # Simulate query set
    query_x = torch.randn(batch_size * num_query * 5, 3)
    query_edge_index = torch.randint(0, num_nodes_per_graph, (2, 50))
    query_batch = torch.zeros(num_nodes_per_graph * batch_size * num_query, dtype=torch.long)
    query_labels = torch.arange(batch_size).repeat_interleave(num_query)
    
    # Forward pass
    try:
        logits = model(support_x, support_edge_index, support_batch,
                      support_labels, query_x, query_edge_index, query_batch,
                      batch_size)
        
        print(f"  ✓ Forward pass succeeded")
        print(f"  ✓ Logits shape: {logits.shape}")
        
        # Check spike count
        spike_count = model.count_spikes()
        print(f"  ✓ Spike count: {spike_count}")
        
        if spike_count == 0:
            print("\n  🚨 WARNING: Spike count is 0!")
            print("     This means no spikes were produced during forward pass.")
            print("     Possible reasons:")
            print("     1. Membrane potential never reached threshold")
            print("     2. Spike counting is not implemented correctly")
            print("     3. Neurons are in 'no_spikes' ablation mode")
            
            # Check if neurons are in correct mode
            for name, module in model.named_modules():
                if 'ManifoldLIFNeuron' in str(type(module)):
                    print(f"\n  Checking {name}:")
                    print(f"    ablation = {module.ablation}")
                    print(f"    u_th = {module.u_th}")
                    print(f"    tau = {module.tau}")
                    
                    # Test forward pass of neuron directly
                    print(f"\n  Testing ManifoldLIFNeuron.forward directly...")
                    batch_size_test = 10
                    x_test = torch.randn(batch_size_test, 128)  # 128 = hidden_dim
                    h_prev = torch.zeros(batch_size_test, 128)
                    c_prev = torch.zeros(batch_size_test, 128)
                    
                    try:
                        out, h_new, c_new = module(x_test, h_prev, c_prev)
                        print(f"    ✓ Output shape: {out.shape}")
                        print(f"    ✓ Output mean: {out.mean().item():.6f}")
                        print(f"    ✓ Output max: {out.max().item():.6f}")
                        print(f"    ✓ Output min: {out.min().item():.6f}")
                        
                        # Check if output is all zeros (no spikes)
                        if out.abs().sum() == 0:
                            print(f"    🚨 WARNING: Output is all zeros (no spikes)!")
                        else:
                            print(f"    ✓ Output has non-zero values (spikes produced)")
                            
                    except Exception as e:
                        print(f"    ❌ Error in neuron forward pass: {e}")
                        import traceback
                        traceback.print_exc()
        
    except Exception as e:
        print(f"  ❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_spikes_during_forward()
