"""
MSG Baseline Fair Comparison: MSG* vs SpikeRPGNN DvM M=8.

Runs both methods on all 4 TU datasets (MUTAG, PROTEINS, ENZYMES, DD)
with identical episode sampling, seeds, and evaluation protocol.

Methods:
  - MSG*     — reimplementation of Sun et al. (NeurIPS 2024)
  - SpikeRPGNN DvM M=8 — our method with truncated DvM

Control: same seeds, same n_way × k_shot, same evaluation protocol.
Output: msg_comparison_results.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys, os, json, time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from msg_baseline.models import MSGStarEncoder, RiemannianProtoClassifier
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import ExperimentConfig as MSGConfig, DATASET_CONFIGS, get_config
from msg_baseline.utils import set_seed, compute_accuracy
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


# ------------------------------------------------------------
# MSG* training (from msg_baseline)
# ------------------------------------------------------------
def msg_train_epoch(encoder, classifier, sampler, optimizer):
    encoder.train()
    total_acc = 0.0
    for _ in range(20):  # episodes_per_epoch
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device="cpu")
        optimizer.zero_grad()
        s_z = encoder(s_b.x, s_b.edge_index, s_b.batch)
        q_z = encoder(q_b.x, q_b.edge_index, q_b.batch)
        logits = classifier(q_z, s_z, s_l, n_classes=sampler.n_way)
        loss = nn.CrossEntropyLoss()(logits, q_l)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()
        total_acc += compute_accuracy(logits, q_l)
    return total_acc / 20


@torch.no_grad()
def msg_evaluate(encoder, classifier, dataset, sampler_cfg, n_test=50):
    """Evaluate MSG on meta-test episodes."""
    encoder.eval()
    test_sampler = EpisodeSampler(
        dataset=dataset, n_way=sampler_cfg["n_way"],
        k_shot=sampler_cfg["shot"], q_query=5,
        seed=sampler_cfg["eval_seed"])
    total_acc = 0.0
    for _ in range(n_test):
        ep = test_sampler.sample_episode(mode="test")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device="cpu")
        s_z = encoder(s_b.x, s_b.edge_index, s_b.batch)
        q_z = encoder(q_b.x, q_b.edge_index, q_b.batch)
        logits = classifier(q_z, s_z, s_l, n_classes=test_sampler.n_way)
        total_acc += compute_accuracy(logits, q_l)
    return total_acc / n_test


def run_msg(dataset_name, shot, run_id, n_epochs=100, device="cpu"):
    """Run one MSG* experiment."""
    ds_cfg = DATASET_CONFIGS[dataset_name]
    run_seed = 42 + run_id * 100 + shot * 10
    set_seed(run_seed)

    dataset = TUGraphDataset(
        name=dataset_name, tu_name=ds_cfg["tu_name"],
        train_ratio=0.8, seed=42)

    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"],
        k_shot=shot, q_query=5, seed=run_seed)

    encoder = MSGStarEncoder(
        in_dim=dataset.num_features, hidden_dim=32, num_layers=3,
        manifold_c=1.0, epsilon=1e-4, tau=20.0).to(device)
    classifier = RiemannianProtoClassifier(manifold=encoder.manifold)

    optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    best = 0.0
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        msg_train_epoch(encoder, classifier, sampler, optimizer)
        scheduler.step()
        if ep % 10 == 0 or ep == 1:
            a = msg_evaluate(encoder, classifier, dataset,
                             {"n_way": ds_cfg["n_way"], "shot": shot,
                              "eval_seed": run_seed + 10000})
            if a > best:
                best = a
    t = time.time() - t0
    return {"best_acc": float(best), "time_s": t}


# ------------------------------------------------------------
# SpikeRPGNN training (from run_final_matrix)
# ------------------------------------------------------------
def spikergnn_train_ep(model, sampler, optimizer):
    model.train()
    accs, losses = 0.0, 0.0
    for _ in range(20):
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device="cpu")
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
    return accs / 20


@torch.no_grad()
def spikergnn_evaluate(model, sampler, n_test=50):
    model.eval()
    total = 0.0
    for _ in range(n_test):
        ep = sampler.sample_episode(mode="test")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device="cpu")
        if hasattr(model, "stmp"):
            model.stmp.lif.reset_membrane()
        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
        logits = model.classify(q_z, s_z, s_l, sampler.n_way)
        total += compute_accuracy(logits, q_l)
    return total / n_test


def run_spikergnn(dataset_name, shot, run_id, n_epochs=100, device="cpu"):
    """Run one SpikeRPGNN DvM M=8 experiment."""
    ds_cfg = DATASET_CONFIGS[dataset_name]
    run_seed = 42 + run_id * 100 + shot * 10
    set_seed(run_seed)

    dataset = TUGraphDataset(
        name=dataset_name, tu_name=ds_cfg["tu_name"],
        train_ratio=0.8, seed=42)

    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"],
        k_shot=shot, q_query=5, seed=run_seed)

    cfg = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, dvm_mode="truncated",
        use_metric_correction=True, ablation="full",
        dv_m_window=8,
        episodes_per_epoch=20, test_episodes=50,
        lr=1e-3, weight_decay=5e-4,
        epochs=n_epochs,
    )
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(cfg, in_dim=in_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    best = 0.0
    t0 = time.time()
    for ep in range(n_epochs):
        spikergnn_train_ep(model, sampler, optimizer)
        scheduler.step()
        if (ep + 1) % 10 == 0:
            test_sampler = EpisodeSampler(
                dataset=dataset, n_way=ds_cfg["n_way"],
                k_shot=shot, q_query=5,
                seed=run_seed + 10000)
            a = spikergnn_evaluate(model, test_sampler)
            if a > best:
                best = a
    t = time.time() - t0
    return {"best_acc": float(best), "time_s": t}


# ------------------------------------------------------------
# Main comparison runner
# ------------------------------------------------------------
DATASETS = ["MUTAG", "PROTEINS", "ENZYMES", "DD"]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MSG* vs SpikeRPGNN fair comparison")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to compare")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Runs per method×dataset (default: 5)")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Epochs per run (default: 100)")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5],
                        help="Shot values (default: 1 5)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = {
        "started": datetime.now().isoformat(),
        "device": args.device,
        "n_runs": args.n_runs,
        "n_epochs": args.n_epochs,
        "datasets": args.datasets,
        "methods": ["MSG*", "SpikeRPGNN DvM M=8"],
    }

    total_experiments = (len(args.datasets) * len(args.shots) *
                         2 * args.n_runs)
    print(f"Total experiments: {total_experiments}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Shots: {args.shots}")
    print(f"  Runs: {args.n_runs}")
    print(f"  Methods: MSG*, SpikeRPGNN DvM M=8")
    print("=" * 60)

    for ds in args.datasets:
        results[ds] = {}
        for shot in args.shots:
            sk = f"{shot}-shot"
            results[ds][sk] = {}
            for method_name, run_fn in [
                ("MSG*", run_msg),
                ("SpikeRPGNN DvM M=8", run_spikergnn),
            ]:
                print(f"\n--- {ds} {sk} {method_name} ---", flush=True)
                accs = []
                times = []
                for r in range(args.n_runs):
                    out = run_fn(ds, shot, r, args.n_epochs, args.device)
                    accs.append(out["best_acc"])
                    times.append(out["time_s"])
                    print(f"    run{r}: acc={out['best_acc']:.4f} "
                          f"time={out['time_s']:.0f}s", flush=True)

                mean = sum(accs) / len(accs)
                std = ((sum((x - mean) ** 2 for x in accs) /
                        (len(accs) - 1)) ** 0.5
                       if len(accs) > 1 else 0.0)
                results[ds][sk][method_name] = {
                    "accs": accs, "mean": mean, "std": std,
                    "time_s": sum(times),
                }
                print(f"    => {mean:.4f} ± {std:.4f}", flush=True)

    results["finished"] = datetime.now().isoformat()

    out_file = args.output or os.path.join(
        os.path.dirname(__file__), "msg_comparison_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Summary: MSG* vs SpikeRPGNN DvM M=8")
    print(f"{'=' * 70}")
    for ds in args.datasets:
        for shot in args.shots:
            sk = f"{shot}-shot"
            msg_data = results[ds][sk].get("MSG*", {})
            sg_data = results[ds][sk].get("SpikeRPGNN DvM M=8", {})
            delta = (sg_data.get("mean", 0) - msg_data.get("mean", 0)) * 100
            print(f"  {ds:<12} {sk:<8} | "
                  f"MSG*: {msg_data.get('mean',0):.4f} | "
                  f"DvM:  {sg_data.get('mean',0):.4f} | "
                  f"Δ={delta:+.1f}%")


if __name__ == "__main__":
    main()
