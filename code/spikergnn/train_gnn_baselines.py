"""
Standard (non-spiking) GNN baselines for few-shot graph classification.

Mirrors the SpikeRPGNN / MSG experimental protocol (ProtoNet-style episodic
training on TU datasets) but replaces the spiking / hyperbolic encoder with a
plain Euclidean GNN encoder (SGC / GIN / GCN / GraphSAGE) + Euclidean
Prototypical Network (mean prototype, squared Euclidean distance).

Why this script exists
----------------------
The manuscript's 28-config matrix (4 datasets x 7 methods) includes SGC/GIN/GCN/
SAGE as non-spiking GNN baselines. They do NOT use DvM / metric correction, so
the P0-② gradient fix does not change their numbers -- but for an honest,
fully-reproducible matrix they must have a real runnable entry point rather than
being "carried over" from prior results. This script provides that entry point,
reusing msg_baseline's data / sampling / evaluation infrastructure.

Usage
-----
  python train_gnn_baselines.py --method gin  --datasets MUTAG --shots 1 5 --n-runs 9 --device cuda
  python train_gnn_baselines.py --method gcn --datasets MUTAG PROTEINS DD ENZYMES --shots 1 5 --n-runs 9

  # CPU smoke test (before the full GPU run): import + one episode per encoder,
  # epochs=2, n-runs=1, no GPU. Verifies the whole pipeline runs end-to-end.
  python train_gnn_baselines.py --quick-test
  python train_gnn_baselines.py --quick-test --method gin --datasets MUTAG --epochs 2
"""
import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    SAGEConv,
    SGConv,
    global_mean_pool,
)

# msg_baseline lives as a sub-package of programfiles/ (no setup.py); add parent.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msg_baseline.config import ExperimentConfig, DATASET_CONFIGS, get_config
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed, compute_accuracy


# ── Euclidean GNN encoders ──────────────────────────────────────────


class GNNEncoder(nn.Module):
    """Plain Euclidean GNN encoder for few-shot graph classification.

    X -> Linear(d_in -> d_h) -> [GNN layer]xL -> global mean pool -> R^{d_h}
    Output is a Euclidean embedding consumed by an Euclidean ProtoNet.
    """

    def __init__(self, method: str, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.method = method
        proj_in = max(int(in_dim), 1)
        if method == "sgc":
            # Classic SGC: K-hop propagation + single linear readout, no nonlinearity.
            self.conv = SGConv(proj_in, hidden_dim, K=max(num_layers, 1), cached=False)
        else:
            self.input_proj = nn.Linear(proj_in, hidden_dim)
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                if method == "gcn":
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif method == "sage":
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                elif method == "gin":
                    mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    self.convs.append(GINConv(mlp, train_eps=True))
                else:
                    raise ValueError(f"Unknown GNN method: {method}")

    def forward(self, x, edge_index, batch):
        if x is None or x.shape[-1] == 0:
            num_nodes = int(batch.numel())
            x = torch.ones(num_nodes, 1, dtype=torch.float, device=edge_index.device)
        if self.method == "sgc":
            h = self.conv(x, edge_index)
            return global_mean_pool(h, batch)
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return global_mean_pool(h, batch)


# ── Euclidean Prototypical Network ──────────────────────────────────


def euclidean_protonet_logits(query_z, supp_z, supp_labels, n_classes):
    """logits[q, c] = -||z_q - proto_c||^2  (smaller distance -> larger logit)."""
    protos = []
    for c in range(n_classes):
        mask = supp_labels == c
        protos.append(supp_z[mask].mean(dim=0))
    protos = torch.stack(protos, dim=0)  # (n_way, d)
    dists = torch.cdist(query_z, protos)  # (n_query, n_way) L2
    return -dists


# ── Train / eval (identical episodic protocol to msg_baseline) ──────


def train_one_epoch(encoder, sampler, optimizer, config, n_way):
    encoder.train()
    total_acc = 0.0
    for _ in range(config.episodes_per_epoch):
        episode = sampler.sample_episode(mode="train")
        supp_b, supp_lbls, qry_b, qry_lbls = batch_episode_data(
            episode, device=config.device
        )
        sz = encoder(supp_b.x, supp_b.edge_index, supp_b.batch)
        qz = encoder(qry_b.x, qry_b.edge_index, qry_b.batch)
        logits = euclidean_protonet_logits(qz, sz, supp_lbls, n_way)
        loss = F.cross_entropy(logits, qry_lbls)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        total_acc += compute_accuracy(logits, qry_lbls)
    return total_acc / config.episodes_per_epoch


@torch.no_grad()
def evaluate(encoder, dataset, config, shot, seed_offset=0):
    encoder.eval()
    ds_cfg = DATASET_CONFIGS[dataset.name]
    test_sampler = EpisodeSampler(
        dataset=dataset,
        n_way=ds_cfg["n_way"],
        k_shot=shot,
        q_query=config.q_query,
        seed=config.seed_base + seed_offset + 10000,
    )
    n_way = test_sampler.n_way
    total = 0.0
    for _ in range(config.test_episodes):
        ep = test_sampler.sample_episode(mode="test")
        sb, sl, qb, ql = batch_episode_data(ep, device=config.device)
        sz = encoder(sb.x, sb.edge_index, sb.batch)
        qz = encoder(qb.x, qb.edge_index, qb.batch)
        logits = euclidean_protonet_logits(qz, sz, sl, n_way)
        total += compute_accuracy(logits, ql)
    return total / config.test_episodes


def run_single_experiment(dataset_name, shot, run_id, config, method):
    ds_cfg = DATASET_CONFIGS[dataset_name]
    run_seed = config.seed_base + run_id * 100 + shot * 10
    set_seed(run_seed)

    dataset = TUGraphDataset(
        name=dataset_name,
        tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio,
        seed=config.seed_base,
    )
    sampler = EpisodeSampler(
        dataset=dataset,
        n_way=ds_cfg["n_way"],
        k_shot=shot,
        q_query=config.q_query,
        seed=run_seed,
    )
    encoder = GNNEncoder(
        method, dataset.num_features, config.hidden_dim, config.num_layers
    ).to(config.device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    best = 0.0
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_acc = train_one_epoch(encoder, sampler, optimizer, config, ds_cfg["n_way"])
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(encoder, dataset, config, shot)
            best = max(best, test_acc)
            print(
                f"  [{dataset_name} {shot}-shot run{run_id}] Epoch {epoch:3d} | "
                f"Train {train_acc:.1%} | Test {test_acc:.1%} | "
                f"Best {best:.1%} | {time.time()-t0:.0f}s",
                flush=True,
            )
    final = evaluate(encoder, dataset, config, shot)
    return final


# ── CLI ─────────────────────────────────────────────────────────────


def quick_test_main(args):
    """CPU smoke test: build every encoder, run ONE train episode + ONE eval
    episode per (method, dataset, shot). Verifies (a) the import chain at the
    top of this file resolves, and (b) the full fwd/backward/eval pipeline runs
    end-to-end on a real TU graph -- without needing a GPU.

    Designed to be run locally *before* the full GPU matrix, so a broken import
    or autograd error is caught early. Raises AssertionError (non-zero exit) if
    anything produces a non-finite accuracy.
    """
    config = get_config()
    config.device = "cpu"            # smoke test: CPU only, never touch the GPU
    config.n_runs = 1                # a single run is enough to prove it runs
    config.epochs = args.epochs if args.epochs is not None else 2
    config.episodes_per_epoch = 1    # one train episode per epoch
    config.test_episodes = 1         # one eval episode

    methods = [args.method] if args.method else ["sgc", "gin", "gcn", "sage"]
    datasets = [d for d in args.datasets if d in DATASET_CONFIGS] or ["MUTAG"]
    shots = args.shots or [1]

    print("=" * 64)
    print("QUICK-TEST (smoke) — CPU only, minimal episodes, no GPU")
    print(f"  methods  = {methods}")
    print(f"  datasets = {datasets}")
    print(f"  shots    = {shots}")
    print(
        f"  epochs={config.epochs} | train_eps/epoch={config.episodes_per_epoch} | "
        f"eval_eps={config.test_episodes}"
    )
    print("=" * 64, flush=True)

    results = {}
    t0 = time.time()
    for method in methods:
        results.setdefault(method, {})
        for ds in datasets:
            if ds not in DATASET_CONFIGS:
                print(f"  skip unknown dataset {ds}")
                continue
            ds_shots = [
                s for s in shots if s in DATASET_CONFIGS[ds].get("shots", shots)
            ]
            results[method].setdefault(ds, {})
            for shot in ds_shots:
                acc = run_single_experiment(ds, shot, 0, config, method)
                assert torch.isfinite(
                    torch.tensor(acc)
                ), f"non-finite acc for {method}/{ds}/{shot}-shot"
                results[method][ds][str(shot)] = float(acc)
                print(
                    f"  [smoke] {method:5s} {ds:8s} {shot}-shot -> {acc:.4f}",
                    flush=True,
                )

    out = {
        "mode": "quick-test",
        "device": "cpu",
        "epochs": config.epochs,
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "gnn_quicktest.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        "\nQUICK-TEST PASSED -- all encoders import and one episode runs "
        "end-to-end on CPU."
    )
    print(f"Saved -> {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Standard GNN baselines (SGC/GIN/GCN/SAGE) few-shot graph classification"
    )
    ap.add_argument(
        "--method",
        default=None,
        choices=["sgc", "gin", "gcn", "sage"],
        help="GNN encoder to train. Required for a full run; omitted under --quick-test "
        "(which loops over all four).",
    )
    ap.add_argument(
        "--datasets", nargs="+", default=["MUTAG", "PROTEINS", "DD", "ENZYMES"]
    )
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--n-runs", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: this script's directory)",
    )
    ap.add_argument(
        "--quick-test",
        action="store_true",
        help=(
            "CPU smoke test: build every encoder, run ONE train+eval episode per "
            "(method,dataset,shot) with epochs=2, n-runs=1. Verifies imports + "
            "end-to-end run without a GPU. Run this locally before the full GPU matrix."
        ),
    )
    args = ap.parse_args()

    if args.quick_test:
        quick_test_main(args)
        return

    if args.method is None:
        ap.error("--method is required for a full run (or use --quick-test to smoke-test all).")

    config = get_config()
    if args.epochs is not None:
        config.epochs = args.epochs
    config.device = args.device
    config.n_runs = args.n_runs

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    results = {}
    t_total = time.time()
    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            print(f"skip unknown dataset {ds}")
            continue
        ds_shots = [
            s for s in args.shots if s in DATASET_CONFIGS[ds].get("shots", args.shots)
        ]
        results[ds] = {}
        for shot in ds_shots:
            accs = []
            for run_id in range(args.n_runs):
                a = run_single_experiment(ds, shot, run_id, config, args.method)
                accs.append(float(a))
                print(f"  {ds} {shot}-shot run{run_id}: {a:.4f}", flush=True)
            mean = sum(accs) / len(accs)
            std = (sum((x - mean) ** 2 for x in accs) / max(len(accs) - 1, 1)) ** 0.5
            results[ds][str(shot)] = {"accs": accs, "mean": mean, "std": std}
            print(f"  >> {ds} {shot}-shot: {mean:.4f} ± {std:.4f}", flush=True)

    out = {
        "method": args.method,
        "device": args.device,
        "n_runs": args.n_runs,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - t_total,
        "results": results,
    }
    tag = "_".join(args.datasets)
    out_path = os.path.join(out_dir, f"gnn_{args.method}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
