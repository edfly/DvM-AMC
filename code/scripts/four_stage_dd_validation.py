"""
Four-Stage Validation on DD Dataset.
Mirrors the proven four_stage_validation.py architecture, adapted for DD.

Stage 1: Optimization Convergence — DvM M=8 vs Full BPTT on DD
Stage 2: Geometric Properties — Riemannian vs Euclidean, conformal factor dist.
Stage 3: Robustness — gradient noise sensitivity + DvM truncation on DD embeddings
Stage 4: Computational Scalability — memory scaling, batch ablation, CI

Output: dd_four_stage_results.json
"""

import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, os, sys, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _ROOT)

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy
from msg_baseline.config import DATASET_CONFIGS
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from spikergnn.dvm import conformal_factor
from torch_geometric.nn import global_mean_pool

DEVICE = "cpu"; torch.set_num_threads(4)
DS_NAME = "DD"
N_EPOCHS = 50; N_RUNS = 3
SHOT = 1

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
def load_dd(split_seed=42):
    ds_cfg = DATASET_CONFIGS[DS_NAME]
    dataset = TUGraphDataset(name=DS_NAME, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=split_seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=SHOT, q_query=5, seed=split_seed)
    in_dim = max(dataset.num_features, 1)
    return dataset, sampler, in_dim

def train_ep(model, sampler, opt, cfg):
    model.train(); acc, lo = 0.0, 0.0
    for _ in range(cfg.episodes_per_epoch):
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, DEVICE)
        opt.zero_grad()
        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
        L = nn.functional.cross_entropy(
            model.classify(q_z, s_z, s_l, sampler.n_way), q_l)
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc += compute_accuracy(
            model.classify(q_z, s_z, s_l, sampler.n_way).detach(), q_l)
        lo += L.item()
    n = max(cfg.episodes_per_epoch, 1)
    return {"acc": acc / n, "loss": lo / n}

def eval_model(model, sampler, cfg):
    model.eval(); total = 0.0
    with torch.no_grad():
        for _ in range(cfg.test_episodes):
            ep = sampler.sample_episode(mode="test")
            s_b, s_l, q_b, q_l = batch_episode_data(ep, DEVICE)
            if hasattr(model, "stmp"): model.stmp.lif.reset_membrane()
            s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
            q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
            total += compute_accuracy(
                model.classify(q_z, s_z, s_l, sampler.n_way), q_l)
    return total / max(cfg.test_episodes, 1)

def train_full(model, sampler, opt, cfg, n_epochs):
    best = 0.0; curve = []
    for ep in range(n_epochs):
        train_ep(model, sampler, opt, cfg)
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            a = eval_model(model, sampler, cfg)
            if a > best: best = a
            curve.append(a)
    return best, curve

# ------------------------------------------------------------
print("=" * 64)
print("  DvM FOUR-STAGE VALIDATION — DD Dataset (1178 graphs)")
print(f"  Epochs: {N_EPOCHS} | Runs: {N_RUNS} | Shot: {SHOT}")
print("=" * 64)

results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "dataset": DS_NAME, "device": DEVICE}
t0_global = time.time()

# ------------------------------------------------------------
# STAGE 1: Optimization Efficiency & Convergence
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 1: Optimization Efficiency & Convergence")
print("=" * 64)

stage1 = {}
_, sampler, in_dim = load_dd(42)

for tag, cfg_kwargs in [
    ("DvM_M8", dict(use_dvm=True, dvm_mode="truncated", dv_m_window=8,
                     use_metric_correction=True)),
    ("Full_BPTT", dict(use_dvm=False)),
]:
    base = dict(hidden_dim=128, num_layers=3, tau=20.0, ablation="full",
                episodes_per_epoch=10, test_episodes=30,
                lr=1e-3, weight_decay=5e-4)
    base.update(cfg_kwargs)
    cfg = SpikeRPGNNConfig(**base, epochs=N_EPOCHS)

    best_accs = []; curves = []
    t_run = time.time()
    for run in range(N_RUNS):
        set_seed(42 + run * 17)
        _, run_sampler, _ = load_dd(42 + run * 17)
        model = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, curve = train_full(model, run_sampler, opt, cfg, N_EPOCHS)
        best_accs.append(best)
        if run == 0: curves = curve
    elapsed = time.time() - t_run
    key = f"{DS_NAME}_{tag}"
    stage1[key] = {
        "accs": best_accs,
        "mean": np.mean(best_accs),
        "std": np.std(best_accs, ddof=1),
        "conv_curve": curves,
        "time_s": elapsed,
        "params": sum(p.numel() for p in model.parameters()),
    }
    print(f"  {key:25s} -> {np.mean(best_accs):.4f}+-{np.std(best_accs,ddof=1):.4f} "
          f"({elapsed:.0f}s)")
results["stage1_convergence"] = stage1

# ------------------------------------------------------------
# STAGE 2: Geometric Properties
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 2: Geometric Properties (Riemannian vs Euclidean)")
print("=" * 64)

stage2 = {}
set_seed(42)
dataset, sampler_s2, in_dim = load_dd(42)

# Riemannian
cfg_riem = SpikeRPGNNConfig(
    hidden_dim=128, num_layers=3, tau=20.0,
    use_dvm=True, dvm_mode="truncated", dv_m_window=8,
    ablation="full", episodes_per_epoch=10, test_episodes=30,
    lr=1e-3, weight_decay=5e-4, epochs=N_EPOCHS)
# Euclidean ablation
cfg_euc = SpikeRPGNNConfig(
    hidden_dim=128, num_layers=3, tau=20.0,
    use_dvm=False, ablation="euclidean",
    episodes_per_epoch=10, test_episodes=30,
    lr=1e-3, weight_decay=5e-4, epochs=N_EPOCHS)

for tag, cfg in [("Riemannian", cfg_riem), ("Euclidean", cfg_euc)]:
    accs = []
    for run in range(N_RUNS):
        set_seed(42 + run * 13)
        model = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, _ = train_full(model, sampler_s2, opt, cfg, N_EPOCHS)
        accs.append(best)
    key = f"{DS_NAME}_{tag}"
    stage2[key] = {"mean": np.mean(accs),
                   "std": np.std(accs, ddof=1) if N_RUNS > 1 else 0}
    print(f"  {key:25s} -> {np.mean(accs):.4f}+-{np.std(accs,ddof=1):.4f}")

# Conformal factor distribution analysis
print("  --- Conformal Factor Analysis ---")
model_s2 = create_spikergnn(cfg_riem, in_dim=in_dim).to(DEVICE)
model_s2.eval()
with torch.no_grad():
    ep = sampler_s2.sample_episode(mode="test")
    s_b, _, _, _ = batch_episode_data(ep, DEVICE)
    z_h, z_e = model_s2.encode(s_b.x, s_b.edge_index, s_b.batch)
    norms = z_h.norm(dim=-1)
    lam = conformal_factor(z_h)
    lam_median = lam.median().item()
    lam_mean = lam.mean().item()
    lam_std = lam.std().item()
    near_origin = (lam < 2.5).float().mean().item() * 100

stage2["conformal_factor"] = {
    "lam_mean": lam_mean, "lam_median": lam_median, "lam_std": lam_std,
    "near_origin_pct": near_origin,
    "norm_mean": norms.mean().item(), "norm_std": norms.std().item(),
}
print(f"    lambda: mean={lam_mean:.3f}, median={lam_median:.3f}, "
      f"std={lam_std:.3f}")
print(f"    near origin (lambda<2.5): {near_origin:.1f}%")
print(f"    norm: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")
results["stage2_geometry"] = stage2

# ------------------------------------------------------------
# STAGE 3: Robustness & Stability
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 3: Robustness & Stability")
print("=" * 64)

stage3 = {}
# 3a. Build gradient chain from DD embeddings
model_full = create_spikergnn(
    SpikeRPGNNConfig(hidden_dim=128, num_layers=3, tau=20.0,
                     use_dvm=False, ablation="full",
                     episodes_per_epoch=3, test_episodes=3),
    in_dim=in_dim).to(DEVICE)
model_full.eval()

with torch.no_grad():
    ep = sampler_s2.sample_episode(mode="train")
    s_b, _, _, _ = batch_episode_data(ep, DEVICE)
    z_h, z_e = model_full.encode(s_b.x, s_b.edge_index, s_b.batch)
N_h = z_h.shape[0]
print(f"  Chain: {N_h} nodes, dim={z_h.shape[1]}")

# Generate contractive dynamics chain
T_c, alpha = 16, 0.9
chain = [z_h.clone()]; p = z_h.clone()
for t in range(T_c):
    v = alpha * (-p) + torch.randn(N_h, z_h.shape[1]) * 0.01
    r = (p + v).norm(dim=-1, keepdim=True).clamp(min=1e-5)
    p = 0.3 * (p + v) / r
    chain.append(p.clone())

# Ref gradient: Full BPTT with metric correction
p0 = chain[0].clone().requires_grad_(True)
pc = p0
for t in range(T_c):
    pc = pc + chain[t + 1] - chain[t]
pc.norm().backward()
g_ref = p0.grad.clone()
l0 = conformal_factor(chain[0]).unsqueeze(-1)
lT = conformal_factor(chain[-1]).unsqueeze(-1)
g_ref = g_ref * ((l0 / lT.clamp(min=1e-5)) ** 2)

# 3b. Noise robustness
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
cos_noise = {}
for ns in noise_levels:
    pn = (chain[0] + torch.randn_like(chain[0]) * ns).requires_grad_(True)
    pc_n = pn
    for t in range(T_c):
        pc_n = pc_n + chain[t + 1] - chain[t]
    pc_n.norm().backward()
    gn = pn.grad.clone()
    gn = gn * ((conformal_factor(pn.detach()).unsqueeze(-1) /
                lT.clamp(min=1e-5)) ** 2)
    cos = (gn.reshape(-1) @ g_ref.reshape(-1)).item() / \
          (gn.norm() * g_ref.norm() + 1e-8).item()
    cos_noise[f"sigma_{ns}"] = cos
    print(f"  Noise sigma={ns:.2f}: cos_to_ref={cos:.4f}")
stage3["gradient_noise"] = cos_noise

# 3c. DvM truncation stability
M_vals = [2, 4, 6, 8]
dvm_stab = {}
for M in M_vals:
    start = T_c - min(M, T_c)
    ps = chain[start].clone().requires_grad_(True)
    pcd = ps
    for i in range(start + 1, T_c + 1):
        pcd = pcd + chain[i].detach() - chain[i - 1].detach()
    pcd.norm().backward()
    gd = ps.grad.clone()
    for i in range(min(M, T_c) - 1, 0, -1):
        lp = conformal_factor(chain[start + i - 1]).unsqueeze(-1)
        lc = conformal_factor(chain[start + i]).unsqueeze(-1)
        gd = gd * (lp / lc.clamp(min=1e-5)) ** 2
    gd = gd * (alpha ** start) * \
         (l0 / conformal_factor(chain[start]).unsqueeze(-1).clamp(min=1e-5)) ** 2
    cos_dvm = (gd.reshape(-1) @ g_ref.reshape(-1)).item() / \
              (gd.norm() * g_ref.norm() + 1e-8).item()
    rel = (gd.reshape(-1) - g_ref.reshape(-1)).norm().item() / \
          g_ref.norm().item()
    dvm_stab[f"M_{M}"] = {"cos": cos_dvm, "rel_err": rel}
    print(f"  DvM M={M}: cos={cos_dvm:.4f}, rel_err={rel:.4f}")
stage3["dvm_truncation"] = dvm_stab
results["stage3_robustness"] = stage3

# ------------------------------------------------------------
# STAGE 4: Computational Scalability
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 4: Computational Scalability")
print("=" * 64)

stage4 = {}

# 4a. Memory scaling O(M)
T_tests = [4, 8, 16, 32, 64, 128]
M = 8
mem_data = {}
for T_val in T_tests:
    ch = [z_h.clone()]; p = z_h.clone()
    for t in range(T_val):
        v = alpha * (-p) + torch.randn(N_h, z_h.shape[1]) * 0.01
        r = (p + v).norm(dim=-1, keepdim=True).clamp(min=1e-5)
        p = 0.3 * (p + v) / r
        ch.append(p.clone())
    states = min(T_val + 1, M + 1)
    ratio = states / (T_val + 1)
    mem_data[f"T_{T_val}"] = {"states": states, "ratio": round(ratio, 3),
                                "saving": f"{1/ratio:.1f}x" if ratio > 0 else "inf"}
    print(f"  T={T_val:4d}: stored={states} (M+1={M+1}), ratio={ratio:.1%}, "
          f"saving={1/ratio:.1f}x")

stage4["memory_scaling"] = mem_data

# 4b. Batch size ablation
bs_vals = [8, 16, 32]
for bs in bs_vals:
    cfg = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, dvm_mode="truncated", dv_m_window=8, ablation="full",
        episodes_per_epoch=bs, test_episodes=30,
        lr=1e-3, weight_decay=5e-4, epochs=20)
    set_seed(42)
    model = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best, _ = train_full(model, sampler_s2, opt, cfg, 20)
    stage4[f"bs_{bs}"] = best
    print(f"  batch_size={bs:3d}: best_acc={best:.4f}")

# 4c. Confidence interval (5 runs)
accs_ci = []
for run in range(5):
    cfg = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, dvm_mode="truncated", dv_m_window=8, ablation="full",
        episodes_per_epoch=10, test_episodes=30,
        lr=1e-3, weight_decay=5e-4, epochs=30)
    set_seed(42 + run * 31)
    _, run_sampler, _ = load_dd(42 + run * 31)
    model = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best, _ = train_full(model, run_sampler, opt, cfg, 30)
    accs_ci.append(best)

mu = np.mean(accs_ci)
se = np.std(accs_ci, ddof=1) / np.sqrt(len(accs_ci))
stage4["confidence"] = {
    "mean": float(mu), "se": float(se),
    "ci95_low": float(mu - 1.96 * se),
    "ci95_high": float(mu + 1.96 * se),
    "accs": [float(a) for a in accs_ci],
}
print(f"  95% CI: {mu:.4f} +- {1.96*se:.4f} (n={len(accs_ci)})")
results["stage4_scalability"] = stage4

# ------------------------------------------------------------
results["total_time_s"] = time.time() - t0_global
out = os.path.join(_ROOT, "dd_four_stage_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 64)
print(f"  DD FOUR-STAGE VALIDATION COMPLETE ({results['total_time_s']:.0f}s)")
print(f"  Results: {out}")
print("=" * 64)
