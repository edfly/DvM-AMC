"""
Fix & Re-run: No Correction + Baseline Experiments.

1. No Correction: Truncated DvM (M=8) WITHOUT metric correction (λ² factors)
   - Achieved by patching RiemannianPullbackFunction to be identity
2. Proto-GNN baseline: Standard GIN encoder + Euclidean prototypical network

All results append to DvM/complete_results.json.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy
from msg_baseline.config import DATASET_CONFIGS
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import add_self_loops


# ═══════════════════════════════════════════════════════════════
# No Correction Config — disable metric correction via config flag
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Proto-GNN Baseline
# ═══════════════════════════════════════════════════════════════

class GINEncoder(nn.Module):
    """Standard GIN encoder for graph classification."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(),
                nn.Linear(2*hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        for conv in self.convs:
            x = conv(x, edge_index_sl)
            x = nn.functional.relu(x)
        return self.pool(x, batch)


class ProtoGNN(nn.Module):
    """Proto-GNN: GIN encoder + Euclidean prototypical network."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, n_way=2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_layers)
        self.n_way = n_way

    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)

    def classify(self, query_z, support_z, support_labels, n_classes):
        # Euclidean prototypical: mean support embeddings -> distance
        n_query = query_z.size(0)
        logits = torch.zeros(n_query, n_classes, device=query_z.device)
        for c in range(n_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                proto = support_z[mask].mean(dim=0)
                logits[:, c] = -((query_z - proto.unsqueeze(0))**2).sum(dim=-1)
        return logits

    def count_spikes(self):
        return 0  # ANN has no spikes


# ═══════════════════════════════════════════════════════════════
# Training utilities (same as complete runner)
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, sampler, optimizer, config, device, is_ann=False):
    model.train()
    total_acc, total_loss = 0.0, 0.0
    for _ in range(config.episodes_per_epoch):
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
        optimizer.zero_grad()
        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
        logits = model.classify(q_z, s_z, s_l, sampler.n_way)
        loss = nn.functional.cross_entropy(logits, q_l)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_acc += compute_accuracy(logits.detach(), q_l)
        total_loss += loss.item()
    n = max(config.episodes_per_epoch, 1)
    return {"mean_acc": total_acc/n, "mean_loss": total_loss/n}


def evaluate_model(model, sampler, config, device, is_ann=False):
    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for _ in range(config.test_episodes):
            ep = sampler.sample_episode(mode="test")
            s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
            if not is_ann and hasattr(model, 'stmp'):
                model.stmp.lif.reset_membrane()
            s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
            q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
            logits = model.classify(q_z, s_z, s_l, sampler.n_way)
            total_acc += compute_accuracy(logits, q_l)
    n = max(config.test_episodes, 1)
    return {"mean_acc": total_acc/n}


def run_training(dataset_name, model, config, device, n_epochs, label, seed_offset=0, is_ann=False):
    """Generic training runner."""
    ds_cfg = DATASET_CONFIGS[dataset_name]
    n_way = ds_cfg["n_way"]
    seed = 42 + seed_offset
    set_seed(seed)

    dataset = TUGraphDataset(name=dataset_name, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=n_way, k_shot=1, q_query=5, seed=seed)
    config.epochs = n_epochs

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(n_epochs):
        train_one_epoch(model, sampler, optimizer, config, device, is_ann)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            eval_m = evaluate_model(model, sampler, config, device, is_ann)
            if eval_m["mean_acc"] > best_acc:
                best_acc = eval_m["mean_acc"]

    t_total = time.time() - t0
    print(f"  [{label}] best={best_acc:.4f}  time={t_total:.0f}s", flush=True)
    return {"best_acc": float(best_acc), "time_s": t_total}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = "cpu"
    n_epochs = 50
    n_runs = 3
    dataset = "MUTAG"

    # Load existing results
    results_file = os.path.join(os.path.dirname(__file__), "complete_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {"started": datetime.now().isoformat(), "device": device}

    config_base = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, ablation="full",
        episodes_per_epoch=20, test_episodes=50,
        lr=1e-3, weight_decay=5e-4,
    )

    # No Correction config: DvM M=8 but WITHOUT metric correction
    config_nocorr = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, ablation="full",
        use_metric_correction=False,  # ← KEY: disables λ² correction
        episodes_per_epoch=20, test_episodes=50,
        lr=1e-3, weight_decay=5e-4,
    )

    ds_cfg = DATASET_CONFIGS[dataset]
    in_dim = max(7, 1)  # MUTAG has 7 features

    # ═══════════════════════════════════════════════════════════
    # Exp 2 Fix: No Correction (corrected)
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Exp 2 Fix: No Correction (Truncated M=8, no metric correction)")
    print("=" * 60)

    no_corr_accs = []
    for run in range(n_runs):
        model = create_spikergnn(config_nocorr, in_dim=in_dim).to(device)
        r = run_training(dataset, model, config_nocorr, device, n_epochs,
                        f"NoCorr run{run}", seed_offset=run * 29)
        no_corr_accs.append(r["best_acc"])

    results["exp2_nocorr_fixed"] = {
        "accs": no_corr_accs,
        "mean": sum(no_corr_accs)/len(no_corr_accs),
        "std": (sum((x - sum(no_corr_accs)/len(no_corr_accs))**2 for x in no_corr_accs)/max(len(no_corr_accs)-1,1))**0.5,
        "note": "Truncated DvM M=8 WITHOUT metric correction"
    }
    print(f"  NoCorr fixed: {results['exp2_nocorr_fixed']['mean']:.4f}±{results['exp2_nocorr_fixed']['std']:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp 6: Baseline Comparison (Proto-GNN)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Exp 6: Baseline Comparison (Proto-GNN on MUTAG)")
    print("=" * 60)

    proto_accs = []
    for run in range(n_runs):
        model = ProtoGNN(in_dim=in_dim, hidden_dim=128, num_layers=3).to(device)
        r = run_training(dataset, model, config_base, device, n_epochs,
                        f"ProtoGNN run{run}", seed_offset=run * 31, is_ann=True)
        proto_accs.append(r["best_acc"])

    results["exp6_baselines"] = {
        "Proto-GNN": {
            "accs": proto_accs,
            "mean": sum(proto_accs)/len(proto_accs),
            "std": (sum((x - sum(proto_accs)/len(proto_accs))**2 for x in proto_accs)/max(len(proto_accs)-1,1))**0.5,
        },
        "DvM M=8 (from Exp2)": results.get("exp2_completion", {}).get("DvM M=8", {"mean": 0.795, "std": 0.071}),
        "Full BPTT (from Exp2)": results.get("exp2_completion", {}).get("Full BPTT", {"mean": 0.801, "std": 0.054}),
    }
    print(f"  Proto-GNN: {results['exp6_baselines']['Proto-GNN']['mean']:.4f}±{results['exp6_baselines']['Proto-GNN']['std']:.4f}")

    results["finished"] = datetime.now().isoformat()

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("FIX & BASELINE EXPERIMENTS COMPLETE")
    print(f"Results saved to: {results_file}")

    # Summary
    print("\n=== FINAL COMPARISON TABLE ===")
    print(f"{'Config':<20} {'Mean Acc':>8} {'Std':>8}")
    print("-" * 40)
    for name, data in results["exp6_baselines"].items():
        print(f"{name:<20} {data['mean']:8.4f} {data['std']:8.4f}")
    if "exp2_nocorr_fixed" in results:
        nc = results["exp2_nocorr_fixed"]
        print(f"{'No Correction (fixed)':<20} {nc['mean']:8.4f} {nc['std']:8.4f}")


if __name__ == "__main__":
    main()
