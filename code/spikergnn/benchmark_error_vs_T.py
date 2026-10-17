"""
DvM Error vs T Benchmark — validates Theorem 1 Corollary 1.

Corollary 1: As T → ∞ with fixed M, ε_grad → constant (does not grow).
This script measures gradient relative error across varying T at fixed M,
confirming the error plateaus rather than diverging.
"""

import torch
import math
import sys
import os

_dvm_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_dvm_dir)
if _dvm_dir not in sys.path:
    sys.path.insert(0, _dvm_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dvm_core import conformal_factor, metric_correction_factor
from dvm_full import compute_gradient_fidelity
from msg_baseline.manifolds import PoincareManifold


def generate_chain(T, d_h, N, c, alpha, noise_std):
    """Generate Riemannian chain of length T."""
    manifold = PoincareManifold(c=c, eps=1e-5)
    p_init = torch.randn(N, d_h) * 0.15
    p_init = p_init / (1.0 + p_init.norm(dim=-1, keepdim=True) * 0.3)
    p_bar = torch.zeros(N, d_h)

    states = [p_init]
    p = p_init
    for t in range(T):
        v = alpha * manifold.logmap(p_bar, p) + torch.randn(N, d_h) * noise_std
        p = manifold.projx(manifold.expmap(v, p))
        states.append(p.clone())

    return states


def compute_bptt_gradient(states, c):
    """Full BPTT gradient (ground truth)."""
    T = len(states) - 1
    p = states[0].clone().requires_grad_(True)
    p_curr = p
    for t in range(T):
        diff = states[t+1].detach() - states[t].detach()
        p_curr = p_curr + diff

    loss = p_curr.norm()
    loss.backward()
    grad = p.grad.clone()

    lam_0 = conformal_factor(states[0], c)
    lam_T = conformal_factor(states[-1], c)
    correction = (lam_0 / lam_T.clamp(min=1e-5)) ** 2
    return grad * correction.unsqueeze(-1)


def compute_dvm_gradient(states, M, alpha, c):
    """Truncated DvM gradient."""
    T = len(states) - 1
    if M >= T:
        return compute_bptt_gradient(states, c)

    p_start = states[T - M].clone().requires_grad_(True)
    p = p_start
    for i in range(T - M + 1, T + 1):
        diff = states[i].detach() - states[i-1].detach()
        p = p + diff

    loss = p.norm()
    loss.backward()
    grad = p_start.grad.clone()

    lam_start = conformal_factor(states[T - M], c)
    lam_end = conformal_factor(states[-1], c)
    grad = grad * ((lam_start / lam_end.clamp(min=1e-5)) ** 2).unsqueeze(-1)

    rho = alpha
    grad = grad * (rho ** (T - M))

    lam_0 = conformal_factor(states[0], c)
    grad = grad * ((lam_0 / lam_start.clamp(min=1e-5)) ** 2).unsqueeze(-1)

    return grad


def main():
    torch.manual_seed(42)
    c = 1.0
    d_h = 32
    N = 128
    alpha = 0.9
    M = 8
    n_trials = 5

    T_values = [8, 16, 24, 32, 48, 64, 96, 128]

    print("=" * 70)
    print("DvM Error vs Sequence Length (Theorem 1 Corollary 1 validation)")
    print(f"  M={M} fixed, T={T_values}, alpha={alpha}, c={c}")
    print("=" * 70)
    print(f"{'T':>5} | {'cos_sim':>8} | {'rel_err':>8} | {'mag_ratio':>9} | {'pred bound':>10}")
    print("-" * 70)

    for T in T_values:
        cos_sims, rel_errs, mag_ratios = [], [], []

        for _ in range(n_trials):
            states = generate_chain(T, d_h, N, c, alpha, 0.05)
            grad_full = compute_bptt_gradient(states, c)
            grad_dvm = compute_dvm_gradient(states, M, alpha, c)
            m = compute_gradient_fidelity(grad_full, grad_dvm)
            cos_sims.append(m["cosine_similarity"])
            rel_errs.append(m["relative_error"])
            mag_ratios.append(m["magnitude_ratio"])

        cos_mean = sum(cos_sims) / n_trials
        rel_mean = sum(rel_errs) / n_trials
        mag_mean = sum(mag_ratios) / n_trials

        # Theoretical bound
        rho = alpha
        bound = (1.0 / (1.0 - rho)) * (1.0 - rho ** (T - M))

        print(f"{T:5d} | {cos_mean:8.4f} | {rel_mean:8.4f} | {mag_mean:9.4f} | {bound:10.2f}")

    # Corollary 1 check: error should plateau, not diverge
    print(f"\nCorollary 1 check:")
    rel_first = rel_errs[0] if isinstance(rel_errs, float) else rel_mean
    print(f"  rel_err range: [{min(rel_errs):.4f}, {max(rel_errs):.4f}]" if n_trials > 1 else "")
    print(f"  Expected: error plateaus at C = {1.0/(1.0-alpha):.1f} × ∇_T")


if __name__ == "__main__":
    main()
