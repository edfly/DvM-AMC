"""
Minimal test: Check ManifoldLIFNeuron output directly.
"""
import sys
sys.path.insert(0, '.')

import torch
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.manifolds import ProductManifold

# Create manifold
manifold = ProductManifold(d_h=128, d_e=32, T=20)

# Create neuron
neuron = ManifoldLIFNeuron(
    manifold=manifold,
    beta_leak=0.1,
    gamma_input=0.5,
    alpha_decay=0.9,
    u_threshold=0.5,
    delta_reset=0.2,
    surrogate_beta=10.0,
    surrogate_type='sigmoid',
    ablation='full',  # Test full variant
)

print("=" * 80)
print("Minimal test: ManifoldLIFNeuron")
print("=" * 80)

# Test forward pass
print("\nTesting forward pass...")
batch_size = 10
d_total = manifold.d_h + manifold.d_e

# Create input sequence
x_seq = torch.randn(batch_size, manifold.T, d_total)

# Initial state
p_init = manifold.origin(batch_size)

# Forward pass
try:
    spikes, p_final = neuron(x_seq, p_init)
    
    print(f"  ✓ Forward pass succeeded")
    print(f"  ✓ Spikes shape: {spikes.shape}")
    print(f"  ✓ Spikes mean: {spikes.mean().item():.6f}")
    print(f"  ✓ Spikes std: {spikes.std().item():.6f}")
    print(f"  ✓ Spikes max: {spikes.max().item():.6f}")
    print(f"  ✓ Spikes min: {spikes.min().item():.6f}")
    
    # Check if spikes are binary (0 or 1)
    unique_vals = torch.unique(spikes.detach())
    print(f"  ✓ Unique values in spikes: {len(unique_vals)}")
    
    if len(unique_vals) <= 2:
        print(f"    → Spikes are binary (good!)")
    else:
        print(f"    → Spikes are continuous (using surrogate gradient)")
        print(f"    → This is normal for training (differentiable)")
        
        # Count "effective spikes" (values > 0.5)
        effective_spikes = (spikes > 0.5).sum().item()
        total_elements = spikes.numel()
        spike_rate = effective_spikes / total_elements
        print(f"    → Effective spike rate: {spike_rate:.4f} ({effective_spikes}/{total_elements})")
    
    # Check p_final
    print(f"\n  ✓ p_final type: {type(p_final)}")
    if isinstance(p_final, tuple):
        print(f"    p_final is tuple of length {len(p_final)}")
        for i, p in enumerate(p_final):
            if isinstance(p, torch.Tensor):
                print(f"    p_final[{i}] shape: {p.shape}")
                print(f"    p_final[{i}] mean: {p.mean().item():.6f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
