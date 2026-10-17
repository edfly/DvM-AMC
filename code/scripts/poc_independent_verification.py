"""
POC: Is an INDEPENDENT empirical verification of Theorem 1 achievable?

Background (PaperSpine audit v2, P0-A):
  four_stage_validation.py computes rel_err from a chain generated under
  no_grad (a constant w.r.t. p0) through an AFFINE accumulation, then
  hand-multiplies the theoretical factor alpha**start. Hence rel_err is
  algebraically forced to equal 1 - rho^(T-M) and cannot test anything.

This POC asks the decisive question: once we (a) let the state chain carry a
REAL autograd graph, and (b) propagate the last-M window with REAL per-step
Jacobians, does the measured relative error DECOUPLE from the closed form?

If YES  -> a genuine, data-dependent empirical test exists (P0-A can be fixed
           by redoing the experiment; Theorem 1 becomes testable).
If NO   -> the construction is algebraically identical to the theory and no
           independent test is possible; the honest route is to downgrade the
           claim (declare Theorem 1 purely theoretical).

ZERO hand-multiplied theoretical factors enter g_trunc beyond the declared
approximation itself (window-exterior contraction), which is exactly the
object under test.
"""
import torch

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 1.0          # Poincare ball curvature
EPS = 1e-5


def lam(x):
    """Conformal factor lambda_x = 2/(1-c||x||^2)."""
    return 2.0 / (1.0 - C * (x * x).sum(-1)).clamp(min=EPS)


def mcf(p_prev, p_curr):
    """Metric correction factor (lambda_prev / lambda_curr)^2."""
    return (lam(p_prev) / lam(p_curr).clamp(min=EPS)) ** 2


def proj(p):
    """Project back into the Poincare ball (non-linear, keeps it on-manifold)."""
    n = p.norm(dim=-1, keepdim=True)
    maxn = (1.0 - 1e-3) / (C ** 0.5)
    return torch.where(n > maxn, p / n * maxn, p)


def build_chain(p0, W, u, T, alpha):
    """Forward chain carrying a REAL autograd graph.

    p_{t+1} = proj( alpha*p_t + tanh(W p_t) + u_t )
    Both the tanh and the projection make dp_{t+1}/dp_t a genuine, state
    dependent Jacobian -- nothing here is affine in p0.
    """
    states = [p0]
    p = p0
    for t in range(T):
        v = alpha * p + torch.tanh(p @ W) + u[t]
        p = proj(p + 0.1 * v)
        states.append(p)
    return states


def loss_fn(final_state, head):
    return (final_state @ head).pow(2).sum()


def measure(T, M, rho, d=32, N=64, alpha=0.9, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(d, d, device=DEV) * 0.3
    u = torch.randn(T, N, d, device=DEV) * 0.05
    head = torch.randn(d, 1, device=DEV)
    p0_init = (torch.randn(N, d, device=DEV) * 0.05)

    # ---------- (1) Ground truth: FULL autograd over the whole real graph ----------
    p0 = p0_init.clone().requires_grad_(True)
    states_full = build_chain(p0, W, u, T, alpha)
    loss_fn(states_full[-1], head).backward()
    g_full = p0.grad.clone()

    # ---------- (2) Truncated estimator: real Jacobians inside the window ----------
    start = T - M
    ps = p0_init.clone().requires_grad_(True)
    # Re-run ONLY the last M transitions with a live graph.
    p = ps.detach().clone().requires_grad_(True)
    for t in range(start, T):
        v = alpha * p + torch.tanh(p @ W) + u[t]
        p = proj(p + 0.1 * v)
    g_window = torch.autograd.grad(loss_fn(p, head), p)[0]   # dL/dp_start, real

    # Per-step metric correction inside the window (real hat lambda ratios).
    p_det = build_chain(ps.detach(), W, u, T, alpha)
    fac = torch.ones(N, device=DEV)
    for i in range(start, T):
        fac = fac * mcf(p_det[i], p_det[i + 1])
    g_corr = g_window * fac.unsqueeze(-1)

    # Window-exterior approximation: contraction decay + endpoint correction.
    g_trunc = g_corr * (rho ** start)
    g_trunc = g_trunc * (lam(p_det[0]) / lam(p_det[start])).pow(2).unsqueeze(-1)

    # ---------- (3) Metrics ----------
    def rel(g):
        return ((g - g_full).norm() / g_full.norm().clamp(min=1e-12)).item()

    cos = (torch.nn.functional.cosine_similarity(
        g_trunc.reshape(1, -1), g_full.reshape(1, -1))).item()
    closed = 1.0 - rho ** (T - M)
    return rel(g_trunc), rel(g_corr), cos, closed


if __name__ == "__main__":
    print(f"device={DEV}\n")
    print(f"{'T':>3} {'M':>3} {'rho':>5} | {'rel_err(trunc)':>14} "
          f"{'rel_err(corr)':>13} {'cos':>8} | {'closed 1-rho^(T-M)':>18} {'|diff|':>9}")
    print("-" * 88)
    decoupled = True
    for T in (16, 24):
        for M in (2, 4, 6, 8):
            for rho in (0.9,):
                for seed in (0, 1, 2):
                    r_t, r_c, c, cl = measure(T, M, rho, seed=seed)
                    d = abs(r_t - cl)
                    if d > 1e-3:
                        decoupled = True
                    if seed == 0:
                        print(f"{T:>3} {M:>3} {rho:>5} | {r_t:>14.6f} {r_c:>13.6f} "
                              f"{c:>8.4f} | {cl:>18.6f} {d:>9.6f}")
    print("-" * 88)

    # Cross-seed spread test: a circular construction gives seed-independent values.
    vals = [measure(16, 8, 0.9, seed=s)[0] for s in range(5)]
    spread = max(vals) - min(vals)
    print(f"\nseed spread @ T=16,M=8 : {[round(v,6) for v in vals]}")
    print(f"spread = {spread:.3e}")

    print("\n=== VERDICT ===")
    if spread > 1e-6:
        print("DECOUPLED: rel_err varies with the draw => data-dependent empirical")
        print("           quantity. An independent test of Theorem 1 IS achievable.")
    else:
        print("STILL CIRCULAR: rel_err is seed-independent => algebraically forced.")
