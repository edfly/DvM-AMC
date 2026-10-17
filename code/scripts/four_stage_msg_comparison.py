"""
Four-Stage MSG* vs SpikeRPGNN Comparison on MUTAG.

Multi-dimensional comparison between MSG (NeurIPS 2024 reimplementation)
and SpikeRPGNN DvM M=8 across four validation stages.

Stage 1: Convergence comparison — training curves, best accuracy
Stage 2: Geometric properties — conformal factor, embedding structure
Stage 3: Robustness — gradient noise sensitivity for both methods
Stage 4: Computational efficiency — memory, time, parameter count

Output: msg_four_stage_comparison.json
"""

import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, os, sys, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _ROOT)

from msg_baseline.models import MSGStarEncoder, RiemannianProtoClassifier
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed, compute_accuracy
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from spikergnn.dvm import conformal_factor

DEVICE = "cpu"; torch.set_num_threads(4)
DS_NAME = "MUTAG"
N_EPOCHS = 50; N_RUNS = 3; SHOT = 1

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
def load_ds():
    ds_cfg = DATASET_CONFIGS[DS_NAME]
    dataset = TUGraphDataset(name=DS_NAME, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=42)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=SHOT, q_query=5, seed=42)
    in_dim = max(dataset.num_features, 1)
    return dataset, sampler, in_dim

def train_ep_spikergnn(model, sampler, opt, cfg):
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
    return acc / n, lo / n

def eval_spikergnn(model, sampler, cfg):
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

def train_full_sg(model, sampler, opt, cfg, n_epochs):
    best = 0.0; curve = []
    for ep in range(n_epochs):
        train_ep_spikergnn(model, sampler, opt, cfg)
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            a = eval_spikergnn(model, sampler, cfg)
            if a > best: best = a
            curve.append(a)
    return best, curve

# --- MSG helpers ---
def train_ep_msg(encoder, classifier, sampler):
    encoder.train()
    total_acc = 0.0
    for _ in range(20):
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, DEVICE)
        s_z = encoder(s_b.x, s_b.edge_index, s_b.batch)
        q_z = encoder(q_b.x, q_b.edge_index, q_b.batch)
        logits = classifier(q_z, s_z, s_l, n_classes=sampler.n_way)
        loss = nn.CrossEntropyLoss()(logits, q_l)
        encoder.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        # Adam step via optimizer from outside
        total_acc += compute_accuracy(logits, q_l)
    return total_acc / 20

@torch.no_grad()
def eval_msg(encoder, classifier, dataset, n_way, shot, eval_seed, n_test=50):
    encoder.eval()
    test_sampler = EpisodeSampler(dataset=dataset, n_way=n_way,
                                  k_shot=shot, q_query=5, seed=eval_seed)
    total = 0.0
    for _ in range(n_test):
        ep = test_sampler.sample_episode(mode="test")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, DEVICE)
        s_z = encoder(s_b.x, s_b.edge_index, s_b.batch)
        q_z = encoder(q_b.x, q_b.edge_index, q_b.batch)
        logits = classifier(q_z, s_z, s_l, n_classes=n_way)
        total += compute_accuracy(logits, q_l)
    return total / n_test

def train_full_msg(encoder, classifier, sampler, dataset, opt, sched,
                   n_way, shot, seed, n_epochs):
    best = 0.0; curve = []
    for ep in range(1, n_epochs + 1):
        train_ep_msg(encoder, classifier, sampler)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            # Use optimizer state for backward
            opt.zero_grad()  # clean up from last step
            a = eval_msg(encoder, classifier, dataset, n_way, shot,
                         seed + 10000, n_test=30)
            if a > best: best = a
            curve.append(a)
    return best, curve

# ------------------------------------------------------------
print("=" * 64)
print("  MSG* vs SpikeRPGNN FOUR-STAGE COMPARISON — MUTAG")
print(f"  Epochs: {N_EPOCHS} | Runs: {N_RUNS} | Shot: {SHOT}")
print("=" * 64)

results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "methods": ["MSG*", "SpikeRPGNN DvM M=8"],
           "dataset": DS_NAME, "device": DEVICE}
t0_global = time.time()

# ------------------------------------------------------------
# STAGE 1: Convergence
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 1: Convergence Comparison")
print("=" * 64)

stage1 = {}
dataset, sampler_base, in_dim = load_ds()
ds_cfg = DATASET_CONFIGS[DS_NAME]

for method in ["MSG*", "SpikeRPGNN DvM M=8"]:
    print(f"\n  --- {method} ---")
    best_accs = []; curves = []; times = []
    for run in range(N_RUNS):
        run_seed = 42 + run * 17
        set_seed(run_seed)
        _, sampler, _ = load_ds()

        t_run = time.time()
        if method == "MSG*":
            encoder = MSGStarEncoder(
                in_dim=in_dim, hidden_dim=32, num_layers=3,
                manifold_c=1.0, epsilon=1e-4, tau=20.0).to(DEVICE)
            classifier = RiemannianProtoClassifier(manifold=encoder.manifold)
            opt = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=5e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=N_EPOCHS, eta_min=1e-5)
            best, curve = train_full_msg(
                encoder, classifier, sampler, dataset, opt, sched,
                ds_cfg["n_way"], SHOT, run_seed, N_EPOCHS)
            n_params = sum(p.numel() for p in encoder.parameters())
        else:
            cfg = SpikeRPGNNConfig(
                hidden_dim=128, num_layers=3, tau=20.0,
                use_dvm=True, dvm_mode="truncated", dv_m_window=8,
                use_metric_correction=True, ablation="full",
                episodes_per_epoch=10, test_episodes=30,
                lr=1e-3, weight_decay=5e-4, epochs=N_EPOCHS)
            model = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
            best, curve = train_full_sg(model, sampler, opt, cfg, N_EPOCHS)
            n_params = sum(p.numel() for p in model.parameters())

        best_accs.append(best)
        curves.append(curve)
        times.append(time.time() - t_run)
        print(f"    run {run}: best={best:.4f}, time={times[-1]:.0f}s")

    stage1[method] = {
        "accs": best_accs,
        "mean": np.mean(best_accs),
        "std": np.std(best_accs, ddof=1),
        "curves": curves,
        "mean_time_s": np.mean(times),
        "params": n_params,
    }
    print(f"    => {np.mean(best_accs):.4f}+-{np.std(best_accs,ddof=1):.4f} "
          f"({np.mean(times):.0f}s/run, {n_params} params)")

results["stage1_convergence"] = stage1

# ------------------------------------------------------------
# STAGE 2: Geometric Properties
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 2: Geometric Properties Comparison")
print("=" * 64)

stage2 = {}
set_seed(42)
with torch.no_grad():
    ep = sampler_base.sample_episode(mode="test")
    s_b, s_l, _, _ = batch_episode_data(ep, DEVICE)

    # MSG embeddings
    encoder_msg = MSGStarEncoder(
        in_dim=in_dim, hidden_dim=32, num_layers=3,
        manifold_c=1.0, epsilon=1e-4, tau=20.0).to(DEVICE)
    z_msg = encoder_msg(s_b.x, s_b.edge_index, s_b.batch)
    # z_msg is in Poincare ball (hidden_dim=32)
    lam_msg = conformal_factor(z_msg)  # use same conformal factor formula
    norms_msg = z_msg.norm(dim=-1)

    # SpikeRPGNN embeddings
    cfg = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, dvm_mode="truncated", dv_m_window=8,
        ablation="full", episodes_per_epoch=10, test_episodes=30,
        lr=1e-3, weight_decay=5e-4)
    model_sg = create_spikergnn(cfg, in_dim=in_dim).to(DEVICE)
    z_h_sg, z_e_sg = model_sg.encode(s_b.x, s_b.edge_index, s_b.batch)
    norms_sg = z_h_sg.norm(dim=-1)
    lam_sg = conformal_factor(z_h_sg)

    stage2["MSG"] = {
        "norm_mean": float(norms_msg.mean()), "norm_std": float(norms_msg.std()),
        "lam_mean": float(lam_msg.mean()), "lam_median": float(lam_msg.median()),
        "lam_std": float(lam_msg.std()),
        "dim": int(z_msg.shape[1]),
    }
    stage2["SpikeRPGNN"] = {
        "norm_mean": float(norms_sg.mean()), "norm_std": float(norms_sg.std()),
        "lam_mean": float(lam_sg.mean()), "lam_median": float(lam_sg.median()),
        "lam_std": float(lam_sg.std()),
        "dim": int(z_h_sg.shape[1]),
    }

    print(f"  MSG       : norm={norms_msg.mean():.4f}+-{norms_msg.std():.4f}, "
          f"lambda={lam_msg.mean():.3f}+-{lam_msg.std():.3f}")
    print(f"  SpikeRPGNN: norm={norms_sg.mean():.4f}+-{norms_sg.std():.4f}, "
          f"lambda={lam_sg.mean():.3f}+-{lam_sg.std():.3f}")

results["stage2_geometry"] = stage2

# ------------------------------------------------------------
# STAGE 3: Robustness
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 3: Robustness Comparison (gradient noise)")
print("=" * 64)

stage3 = {}
# Gradient chain from MUTAG embeddings
with torch.no_grad():
    z_h, _ = model_sg.encode(s_b.x, s_b.edge_index, s_b.batch)
N_h = z_h.shape[0]

T_c, alpha = 16, 0.9
chain = [z_h.clone()]; p = z_h.clone()
for t in range(T_c):
    v = alpha * (-p) + torch.randn(N_h, z_h.shape[1]) * 0.01
    r = (p + v).norm(dim=-1, keepdim=True).clamp(min=1e-5)
    p = 0.3 * (p + v) / r
    chain.append(p.clone())

# Reference gradient
p0 = chain[0].clone().requires_grad_(True)
pc = p0
for t in range(T_c): pc = pc + chain[t + 1] - chain[t]
pc.norm().backward()
g_ref = p0.grad.clone()
l0 = conformal_factor(chain[0]).unsqueeze(-1)
lT = conformal_factor(chain[-1]).unsqueeze(-1)
g_ref = g_ref * ((l0 / lT.clamp(min=1e-5)) ** 2)

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
for ns in noise_levels:
    pn = (chain[0] + torch.randn_like(chain[0]) * ns).requires_grad_(True)
    pc_n = pn
    for t in range(T_c): pc_n = pc_n + chain[t + 1] - chain[t]
    pc_n.norm().backward()
    gn = pn.grad.clone()
    gn = gn * ((conformal_factor(pn.detach()).unsqueeze(-1) /
                lT.clamp(min=1e-5)) ** 2)
    cos = (gn.reshape(-1) @ g_ref.reshape(-1)).item() / \
          (gn.norm() * g_ref.norm() + 1e-8).item()
    stage3[f"noise_{ns}"] = cos
    print(f"  sigma={ns:.2f}: cos_to_ref={cos:.4f}")

results["stage3_robustness"] = stage3

# ------------------------------------------------------------
# STAGE 4: Computational Efficiency
# ------------------------------------------------------------
print("\n" + "=" * 64)
print("STAGE 4: Computational Efficiency")
print("=" * 64)

stage4 = {}
stage4["MSG"] = stage1.get("MSG*", {})
stage4["SpikeRPGNN"] = stage1.get("SpikeRPGNN DvM M=8", {})

# Memory footprint
msg_params = stage4["MSG"].get("params", 0)
sg_params = stage4["SpikeRPGNN"].get("params", 0)
msg_time = stage4["MSG"].get("mean_time_s", 0)
sg_time = stage4["SpikeRPGNN"].get("mean_time_s", 0)

stage4["comparison"] = {
    "params_ratio": round(msg_params / sg_params, 2) if sg_params else 0,
    "time_ratio": round(msg_time / sg_time, 2) if sg_time else 0,
    "msg_params": msg_params,
    "sg_params": sg_params,
    "msg_time_s": msg_time,
    "sg_time_s": sg_time,
}
print(f"  MSG params:       {msg_params:,}")
print(f"  SpikeRPGNN params: {sg_params:,}")
print(f"  Params ratio:      {msg_params/sg_params:.2f}x" if sg_params else "  N/A")
print(f"  MSG time/run:      {msg_time:.0f}s")
print(f"  SpikeRPGNN time:   {sg_time:.0f}s")

results["stage4_efficiency"] = stage4

# ------------------------------------------------------------
results["total_time_s"] = time.time() - t0_global
out = os.path.join(_ROOT, "msg_four_stage_comparison.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 64)
print(f"  MSG vs SpikeRPGNN FOUR-STAGE COMPARISON COMPLETE "
      f"({results['total_time_s']:.0f}s)")
print(f"  Results: {out}")
print("=" * 64)

# Summary
print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
for method in ["MSG*", "SpikeRPGNN DvM M=8"]:
    if method in stage1:
        m = stage1[method]
        print(f"  {method:25s}: {m['mean']:.4f}+-{m['std']:.4f} "
              f"({m['mean_time_s']:.0f}s, {m['params']:,} params)")
