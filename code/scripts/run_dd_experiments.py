"""
DD Dataset Complete Experiment Matrix (and multi-dataset re-runs).

Runs the full 5-config experiment matrix on DD with 5 runs x 100 epochs,
plus re-runs MUTAG/PROTEINS/ENZYMES with 5 runs for CI computation.

Configurations:
  1. Full BPTT         — standard autograd, O(T) memory
  2. DvM M=8 truncated — O(M) memory, metric-corrected truncated BPTT
  3. DvM M=4 truncated — more aggressive truncation
  4. No Correction M=8  — truncated without metric correction (ablation)
  5. Proto-GNN          — ANN baseline (GIN + mean pool)

Output: dd_results.json, mutag_5runs.json, proteins_5runs.json, enzymes_5runs.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys, os, json, time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy
from msg_baseline.config import DATASET_CONFIGS
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import add_self_loops


# ------------------------------------------------------------
# Proto-GNN (ANN baseline)
# ------------------------------------------------------------
class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        for conv in self.convs:
            x = nn.functional.relu(conv(x, edge_index_sl))
        return global_mean_pool(x, batch)


class ProtoGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_layers)

    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)

    def classify(self, query_z, support_z, support_labels, n_classes):
        n_query = query_z.size(0)
        logits = torch.zeros(n_query, n_classes, device=query_z.device)
        for c in range(n_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                proto = support_z[mask].mean(dim=0)
                logits[:, c] = -((query_z - proto.unsqueeze(0)) ** 2).sum(dim=-1)
        return logits

    def count_spikes(self):
        return 0


# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------
def train_ep(model, sampler, optimizer, config, device, is_ann=False):
    model.train()
    accs, losses = 0.0, 0.0
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
        accs += compute_accuracy(logits.detach(), q_l)
        losses += loss.item()
    n = max(config.episodes_per_epoch, 1)
    return {"acc": accs / n, "loss": losses / n}


def eval_model(model, sampler, config, device, is_ann=False):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(config.test_episodes):
            ep = sampler.sample_episode(mode="test")
            s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
            if not is_ann and hasattr(model, "stmp"):
                model.stmp.lif.reset_membrane()
            s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
            q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
            logits = model.classify(q_z, s_z, s_l, sampler.n_way)
            total += compute_accuracy(logits, q_l)
    return total / max(config.test_episodes, 1)


def run_one(dataset_name, model_or_config, device, n_epochs, label,
            seed_offset, is_ann=False, shot=1):
    set_seed(42 + seed_offset)
    ds_cfg = DATASET_CONFIGS[dataset_name]
    dataset = TUGraphDataset(
        name=dataset_name, tu_name=ds_cfg["tu_name"],
        train_ratio=0.8, seed=42 + seed_offset)
    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"],
        k_shot=shot, q_query=5, seed=42 + seed_offset)

    if is_ann:
        model = model_or_config
    else:
        model_or_config.epochs = n_epochs
        in_dim = max(dataset.num_features, 1)
        model = create_spikergnn(model_or_config, in_dim=in_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    best = 0.0
    t0 = time.time()
    for ep in range(n_epochs):
        train_ep(model, sampler, optimizer,
                 model_or_config if not is_ann else SpikeRPGNNConfig(
                     episodes_per_epoch=20),
                 device, is_ann)
        scheduler.step()
        if (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            a = eval_model(model, sampler,
                           model_or_config if not is_ann else SpikeRPGNNConfig(
                               test_episodes=50),
                           device, is_ann)
            if a > best:
                best = a
    t = time.time() - t0
    print(f"  [{label}] best={best:.4f}  time={t:.0f}s", flush=True)
    return {"best_acc": float(best), "time_s": t}


# ------------------------------------------------------------
# Experiment configurations
# ------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "Full BPTT",
        "config": dict(hidden_dim=128, num_layers=3, tau=20.0,
                       use_dvm=False, ablation="full",
                       episodes_per_epoch=20, test_episodes=50,
                       lr=1e-3, weight_decay=5e-4),
    },
    {
        "name": "DvM M=8 truncated",
        "config": dict(hidden_dim=128, num_layers=3, tau=20.0,
                       use_dvm=True, dvm_mode="truncated",
                       use_metric_correction=True, ablation="full",
                       dv_m_window=8,
                       episodes_per_epoch=20, test_episodes=50,
                       lr=1e-3, weight_decay=5e-4),
    },
    {
        "name": "DvM M=4 truncated",
        "config": dict(hidden_dim=128, num_layers=3, tau=20.0,
                       use_dvm=True, dvm_mode="truncated",
                       use_metric_correction=True, ablation="full",
                       dv_m_window=4,
                       episodes_per_epoch=20, test_episodes=50,
                       lr=1e-3, weight_decay=5e-4),
    },
    {
        "name": "No Correction M=8",
        "config": dict(hidden_dim=128, num_layers=3, tau=20.0,
                       use_dvm=True, dvm_mode="truncated",
                       use_metric_correction=False, ablation="full",
                       dv_m_window=8,
                       episodes_per_epoch=20, test_episodes=50,
                       lr=1e-3, weight_decay=5e-4),
    },
    {
        "name": "Proto-GNN",
        "is_ann": True,
    },
]


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def run_dataset(dataset_name, n_runs=5, n_epochs=100, shots=(1, 5),
                device="cpu"):
    """Run full experiment matrix on a single dataset."""
    results = {
        "started": datetime.now().isoformat(),
        "device": device,
        "dataset": dataset_name,
        "n_epochs": n_epochs,
        "n_runs": n_runs,
        "shots": list(shots),
    }

    for shot in shots:
        shot_key = f"{shot}-shot"
        results[shot_key] = {}
        print(f"\n{'=' * 60}")
        print(f"  {dataset_name} | {shot}-shot | {n_runs} runs × {n_epochs} epochs")
        print(f"{'=' * 60}")

        for exp in EXPERIMENTS:
            name = exp["name"]
            is_ann = exp.get("is_ann", False)
            label_prefix = f"{dataset_name}/{shot}-shot/{name}"
            print(f"\n--- {label_prefix} ---", flush=True)

            accs = []
            for run in range(n_runs):
                seed_offset = run * 17 + shot * 31
                if is_ann:
                    ds_cfg = DATASET_CONFIGS[dataset_name]
                    tmp_ds = TUGraphDataset(
                        name=dataset_name, tu_name=ds_cfg["tu_name"],
                        train_ratio=0.8, seed=42 + seed_offset)
                    in_dim_val = max(tmp_ds.num_features, 1)
                    model = ProtoGNN(in_dim=in_dim_val).to(device)
                    r = run_one(dataset_name, model, device, n_epochs,
                                f"{label_prefix} run{run}", seed_offset,
                                is_ann=True, shot=shot)
                else:
                    cfg = SpikeRPGNNConfig(**exp["config"])
                    r = run_one(dataset_name, cfg, device, n_epochs,
                                f"{label_prefix} run{run}", seed_offset,
                                shot=shot)
                accs.append(r["best_acc"])

            mean = sum(accs) / len(accs)
            std = (sum((x - mean) ** 2 for x in accs) /
                   (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
            results[shot_key][name] = {
                "accs": accs, "mean": mean, "std": std}
            print(f"  => {mean:.4f} ± {std:.4f}", flush=True)

    results["finished"] = datetime.now().isoformat()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DD (and TU) dataset experiment matrix")
    parser.add_argument("--datasets", nargs="+",
                        default=["DD"],
                        help="Datasets to run (default: DD)")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of runs per config (default: 5)")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Epochs per run (default: 100)")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5],
                        help="Shot values (default: 1 5)")
    parser.add_argument("--device", default="cpu",
                        help="Device (default: cpu)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (auto-generated if None)")
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\n{'#' * 60}")
        print(f"#  Running: {dataset}")
        print(f"{'#' * 60}")
        results = run_dataset(
            dataset,
            n_runs=args.n_runs,
            n_epochs=args.n_epochs,
            shots=args.shots,
            device=args.device,
        )

        out_file = args.output or os.path.join(
            os.path.dirname(__file__),
            f"{dataset.lower()}_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {out_file}")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  Summary: {dataset}")
        print(f"{'=' * 60}")
        for shot_key in sorted(results.get("shots", [])):
            sk = f"{shot_key}-shot"
            print(f"\n  {sk}:")
            for exp in EXPERIMENTS:
                entry = results.get(sk, {}).get(exp["name"], {})
                m = entry.get("mean", 0)
                s = entry.get("std", 0)
                print(f"    {exp['name']:<22} {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
