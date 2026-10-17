"""clamp-lambda full rerun: bounded metric correction sweep + locked matrix.

Reuses the EXACT training protocol of run_final_matrix.py (train_ep / eval_model /
run_one) so the numbers are directly comparable to final_results_{dataset}.json
(uncapped AMC M=8 = 74.0% on MUTAG). Adds metric_clamp to the AMC truncated config
to exercise the capped conformal factor (dvm.py conformal_factor(max_norm)).

Usage:
  # fast sweep to locate r_max* (n_runs=3)
  python exp_clamp_rerun.py --dataset MUTAG --clamps 0.70,0.80,0.85,0.90,0.95 \
      --n-runs 3 --n-epochs 50 --device cuda --out results_v3/clamp_sweep_mutag.json

  # locked full rerun (n_runs=9, matches final_results protocol)
  python exp_clamp_rerun.py --dataset MUTAG --clamps 0.90 --n-runs 9 --n-epochs 50 \
      --device cuda --out results_v3/clamp_locked_mutag.json
"""
import sys, os, json, time, argparse
from datetime import datetime

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.run_final_matrix import run_one   # identical protocol as final_results_*.json

# AMC truncated M=8 base config (mirrors run_final_matrix EXPERIMENTS[1])
AMC_BASE = dict(
    hidden_dim=128, num_layers=3, tau=20.0,
    use_dvm=True, dvm_mode="truncated",
    use_metric_correction=True, ablation="full",
    dv_m_window=8,
    episodes_per_epoch=20, test_episodes=50,
    lr=1e-3, weight_decay=5e-4,
)


def run_clamp(dataset, clamp, n_runs, n_epochs, device):
    accs = []
    for run in range(n_runs):
        cfg = SpikeRPGNNConfig(**AMC_BASE, metric_clamp=clamp)
        r = run_one(dataset, cfg, device, n_epochs,
                    f"clamp={clamp} run{run}", run * 17)
        accs.append(r["best_acc"])
    mean = sum(accs) / len(accs)
    std = (sum((x - mean) ** 2 for x in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
    return {"clamp": clamp, "accs": accs, "mean": mean, "std": std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--clamps", default="0.70,0.80,0.85,0.90,0.95",
                    help="comma-separated metric_clamp (trusted sub-ball radius) values")
    ap.add_argument("--n-runs", type=int, default=9)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_v3/clamp_sweep.json")
    args = ap.parse_args()

    clamps = [float(x) for x in args.clamps.split(",") if x.strip()]
    print(f"[exp_clamp_rerun] dataset={args.dataset} clamps={clamps} "
          f"n_runs={args.n_runs} n_epochs={args.n_epochs} device={args.device}")

    results = {"started": datetime.now().isoformat(), "dataset": args.dataset,
               "n_runs": args.n_runs, "n_epochs": args.n_epochs, "device": args.device,
               "clamps": clamps, "rows": []}

    for clamp in clamps:
        t0 = time.time()
        row = run_clamp(args.dataset, clamp, args.n_runs, args.n_epochs, args.device)
        row["seconds"] = time.time() - t0
        results["rows"].append(row)
        print(f"  clamp={clamp:.3f}  acc={row['mean']:.4f}±{row['std']:.4f}  "
              f"({row['seconds']:.0f}s)", flush=True)

    results["finished"] = datetime.now().isoformat()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {args.out}")


if __name__ == "__main__":
    main()
