#!/usr/bin/env python3
"""
Regression guard for the B-route experiment layout.

Goal: confirm that the P5 encoder change (node_pre_dim, default 0) did NOT
break the working pipeline on the feature-rich main benchmarks MUTAG (18-dim,
2-way) and PROTEINS (feature-rich, 2-way) — the datasets where SpikeRPGNN is
supposed to actually learn (vs. the data-limited ENZYMES 5-way that P6 proved
intrinsically hard).

Runs the 6 ablation variants on each dataset for a short 15 epochs and records
train-loss curve, 50-episode eval accuracy, and spike health (first/last epoch).
"""
import sys, os, json, time, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch
from spikergnn.ablation import get_all_ablation_configs
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data

EPOCHS = 15
EVAL_EPISODES = 50
DEVICE = "cpu"


def evaluate(model, sampler, n_way, device, n_eval=EVAL_EPISODES):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            ep = sampler.sample_episode("test")
            sb, sl, qb, ql = batch_episode_data(ep, device)
            zs = model.encode(sb.x, sb.edge_index, sb.batch)
            zq = model.encode(qb.x, qb.edge_index, qb.batch)
            out = model.classify(zq, zs, sl, n_way)
            pred = out.argmax(1)
            correct += (pred == ql).sum().item()
            total += ql.numel()
    return correct / max(total, 1)


def run_variant(ds_name, n_way, cfg, device):
    DS = TUGraphDataset(name=ds_name, tu_name=ds_name, train_ratio=0.8, seed=42)
    in_dim = max(DS.num_features, 1)
    model = SpikeRPGNN(cfg, in_dim=in_dim)
    sampler = EpisodeSampler(DS, n_way=n_way, k_shot=1, q_query=5, seed=42)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses = []
    spikes = []
    for ep in range(EPOCHS):
        tm = train_one_epoch(model, sampler, opt, cfg, device)
        losses.append(tm["mean_loss"])
        spikes.append(tm["spike_count"])
        if (ep + 1) % 5 == 0 or ep == 0:
            acc = evaluate(model, sampler, n_way, device, n_eval=20)
            print(
                f"  [{ds_name}|{cfg.ablation}] ep{ep+1} "
                f"loss={tm['mean_loss']:.3f} val={acc:.3f} spikes={tm['spike_count']}",
                flush=True,
            )
    final = evaluate(model, sampler, n_way, device, n_eval=EVAL_EPISODES)
    return {
        "dataset": ds_name,
        "variant": cfg.ablation,
        "n_way": n_way,
        "train_loss": losses,
        "final_eval": final,
        "random": 1.0 / n_way,
        "spike_first": int(spikes[0]),
        "spike_last": int(spikes[-1]),
        "params": sum(p.numel() for p in model.parameters()),
    }


def main():
    datasets = {"MUTAG": 2, "PROTEINS": 2}
    results = {}
    for ds, nway in datasets.items():
        base = SpikeRPGNNConfig(
            hidden_dim=128, num_layers=3, use_dvm=True, ablation="full", epochs=EPOCHS
        )
        configs = get_all_ablation_configs(base)
        for variant, cfg in configs.items():
            t0 = time.time()
            r = run_variant(ds, nway, cfg, DEVICE)
            r["time_s"] = time.time() - t0
            results[f"{ds}|{variant}"] = r
            print(
                f">>> DONE {ds}|{variant}: eval={r['final_eval']:.3f} "
                f"(rand={r['random']:.3f}) spikes_last={r['spike_last']}",
                flush=True,
            )
    json.dump(results, open("spikergnn_results/diag_regression.json", "w"), indent=2, default=str)
    print("SAVED spikergnn_results/diag_regression.json")


if __name__ == "__main__":
    main()
