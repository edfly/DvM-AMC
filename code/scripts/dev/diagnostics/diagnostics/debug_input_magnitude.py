"""
Debug: Check input magnitude to neurons in actual model.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.model import create_spikergnn
from spikergnn.data import load_dataset

def debug_input_magnitude():
    """Debug input magnitude to neurons."""
    
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
    print("Debug: Checking input magnitude to neurons...")
    print("=" * 80)
    
    # Load dataset
    train_data, test_data, n_way = load_dataset('ENZYMES', config.seed_base)
    
    # Create sampler
    from spikergnn.sampler import EpisodicSampler
    sampler = EpisodicSampler(
        train_data, test_data, n_way=5, n_shot=1,
        episodes_per_epoch=2, test_episodes=10, seed=config.seed_base
    )
    
    # Create model
    model = create_spikergnn(config, in_dim=3)
    model.train()
    
    # Run one episode
    print("\n1. Running one training episode...")
    episode = sampler.sample_episode(mode="train")
    support_batch, support_labels, query_batch, query_labels = episode
    
    support_x, support_edge_index, support_batch = support_batch
    query_x, query_edge_index, query_batch = query_batch
    
    # Forward pass
    try:
        logits = model(
            support_x, support_edge_index, support_batch,
            support_labels, query_x, query_edge_index, query_batch,
            n_way=5
        )
        
        print(f"  ✓ Forward pass succeeded")
        print(f"  ✓ Logits shape: {logits.shape}")
        
        # Check spike count
        spike_count = model.count_spikes()
        print(f"  ✓ Spike count: {spike_count}")
        
        # Check input magnitude to MSPE
        print(f"\n2. Checking input to MSPE...")
        with torch.no_grad():
            # Get input to MSPE
            x = support_x
            edge_index = support_edge_index
            batch = support_batch
            
            # Project input
            z_hyp, z_euc = model.mspe.input_proj(x).chunk(2, dim=-1)
            print(f"    After input_proj: std={model.mspe.input_proj(x).std().item():.4f}")
            print(f"    z_hyp std: {z_hyp.std().item():.4f}")
            print(f"    z_euc std: {z_euc.std().item():.4f}")
            
            # Normalize
            z_hyp = F.normalize(z_hyp, dim=-1) * math.tan(model.mspe.manifold.curvature_sqrt)
            z_euc = torch.tanh(z_euc) * model.mspe.euc_scale
            z_merged = model.mspe.manifold.merge(z_hyp, z_euc)
            
            print(f"    After normalization: z_merged std={z_merged.std().item():.4f}")
            
            # Create input sequence
            x_seq = z_merged.unsqueeze(1).expand(-1, model.mspe.T, -1).clone()
            print(f"    x_seq std: {x_seq.std().item():.4f}")
            print(f"    x_seq mean: {x_seq.mean().item():.4f}")
            
            # Check if input is too small
            if x_seq.std().item() < 0.5:
                print(f"\n  🚨 WARNING: Input magnitude is SMALL!")
                print(f"     This might explain why no spikes are produced.")
                print(f"     Suggestion: Increase input magnitude or lower threshold.")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)

if __name__ == "__main__":
    import math
    import torch.nn.functional as F
    debug_input_magnitude()
