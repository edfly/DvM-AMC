"""Diagnose why POC rel_err pinned to 1.0: print raw norms."""
import torch
from poc_independent_verification import build_chain, loss_fn, lam, mcf, proj, DEV

torch.manual_seed(0)
d, N, T, M, alpha, rho = 32, 64, 16, 8, 0.9, 0.9
C = 1.0
W = torch.randn(d, d, device=DEV) * 0.3
u = torch.randn(T, N, d, device=DEV) * 0.05
head = torch.randn(d, 1, device=DEV)
p0_init = torch.randn(N, d, device=DEV) * 0.05

p0 = p0_init.clone().requires_grad_(True)
st = build_chain(p0, W, u, T, alpha)
print("chain norms:", [round(x.norm().item(), 6) for x in st])
loss_fn(st[-1], head).backward()
g_full = p0.grad.clone()
print(f"g_full.norm = {g_full.norm().item():.6e}")
print(f"g_full has nan: {torch.isnan(g_full).any().item()}")

# gradient magnitude vs depth: where does it vanish?
for cut in (1, 4, 8, 12, 16):
    p = p0_init.clone().requires_grad_(True)
    st2 = build_chain(p, W, u, cut, alpha)
    loss_fn(st2[-1], head).backward()
    print(f"  depth={cut:2d}  grad_norm={p.grad.norm().item():.6e}")

# per-step Jacobian contraction check
p = p0_init.clone().requires_grad_(True)
a = build_chain(p, W, u, 1, alpha)[-1]
J = torch.zeros(N * d, N * d, device=DEV)
flat = a.reshape(-1)
# spectral norm of dp_{t+1}/dp_t at init
p1 = p0_init.clone().requires_grad_(True)
q = proj(p1 + 0.1 * (alpha * p1 + torch.tanh(p1 @ W) + u[0]))
Jn = torch.autograd.functional.jacobian(lambda z: proj(z + 0.1 * (alpha * z + torch.tanh(z @ W) + u[0])), p0_init)
sv = torch.linalg.svdvals(Jn.reshape(N * d, N * d))
print(f"\nper-step Jacobian top singular values: {sv[:5].tolist()}")
print(f"=> effective per-step contraction ~ {sv[0].item():.4f}")
print(f"=> after {T - M} skipped steps: {sv[0].item() ** (T - M):.6e}  vs rho^(T-M)={rho ** (T - M):.6e}")
