"""
Debug: Check ManifoldLIFNeuron threshold and membrane potential.
"""
import sys
sys.path.insert(0, '.')

import torch
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.manifolds import ProductManifold

# Create manifold and neuron
manifold = ProductManifold(d_h=128, d_e=32)
neuron = ManifoldLIFNeuron(
    manifold=manifold,
    beta_leak=0.1,
    gamma_input=0.5,
    alpha_decay=0.9,
    u_threshold=0.5,
    delta_reset=0.2,
    surrogate_beta=10.0,
    surrogate_type='sigmoid',
    ablation='full',
)

print("=" * 80)
print("Debug: ManifoldLIFNeuron threshold and membrane potential")
print("=" * 80)

# Check parameters
print(f"\n1. Neuron parameters:")
print(f"  u_threshold (fixed): {neuron.u_th}")
print(f"  u_th_learnable: {neuron.u_th_learnable.item():.6f}")
print(f"  beta_leak: {neuron.beta}")
print(f"  gamma_input: {neuron.gamma}")
print(f"  surrogate_beta: {neuron.surrogate.beta}")

# Create dummy input
print(f"\n2. Testing forward pass with different input magnitudes...")
d_total = manifold.d_h + manifold.d_e
T = 20  # Default T value

test_cases = [
    ("Small input (std=0.1)", torch.randn(10, T, d_total) * 0.1),
    ("Medium input (std=0.5)", torch.randn(10, T, d_total) * 0.5),
    ("Large input (std=1.0)", torch.randn(10, T, d_total) * 1.0),
    ("Very large input (std=2.0)", torch.randn(10, T, d_total) * 2.0),
]

for name, x_seq in test_cases:
    # Initial state
    batch_size = x_seq.size(0)
    p_init = manifold.origin(batch_size)
    
    # Forward pass
    neuron.reset_membrane()
    spikes, p_final = neuron(x_seq, p_init)
    
    # Check results
    spike_count = neuron._spike_count
    spike_rate = neuron._spike_rate
    
    print(f"\n  {name}:")
    print(f"    Input std: {x_seq.std().item():.4f}")
    print(f"    Spike count: {spike_count}")
    print(f"    Spike rate: {spike_rate:.4f}")
    
    if spike_count == 0:
        print(f"    ⚠️  No spikes produced!")
        print(f"    Suggestion: Lower u_th or increase input magnitude")
    else:
        print(f"    ✓ Spikes produced!")

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)

print("\n💡 Diagnosis:")
print("  If all test cases produce 0 spikes, then:")
print("  1. u_th might be too high")
print("  2. Or the neuron implementation has a bug")
print("\n  Solution: Try lowering u_th_learnable or check the forward() implementation.")
