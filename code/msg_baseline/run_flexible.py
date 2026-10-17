"""
Flexible runner for MSG* baseline experiments with timeout-aware execution.

Usage:
    python run_flexible.py --datasets DD --shots 5 --start-run 0 --n-runs 5 [--episodes-per-epoch 10]

This script overrides episodes_per_epoch to fit within Bash tool's 10-min timeout.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msg_baseline.config import ExperimentConfig, DATASET_CONFIGS, get_config
from msg_baseline.train_episodic import run_single_experiment
from msg_baseline.utils import ResultTracker, set_seed
import argparse


def main():
    parser = argparse.ArgumentParser(description="Flexible MSG* Runner (timeout-aware)")
    parser.add_argument("--datasets", nargs="+", default=["DD"], help="Datasets")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5], help="Shot settings")
    parser.add_argument("--start-run", type=int, default=0, help="Start run index")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--episodes-per-epoch", type=int, default=None,
                        help="Override episodes per epoch (for timeout control)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs")
    args = parser.parse_args()

    config = get_config()
    if args.episodes_per_epoch is not None:
        config.episodes_per_epoch = args.episodes_per_epoch
    if args.epochs is not None:
        config.epochs = args.epochs
    config.n_runs = args.n_runs

    tracker = ResultTracker()

    print("=" * 65)
    print(f"  MSG* Flexible Runner")
    print(f"  Datasets: {args.datasets} | Shots: {args.shots}")
    print(f"  Runs: {args.start_run}-{config.n_runs-1} | Epochs: {config.epochs}")
    print(f"  Episodes/epoch: {config.episodes_per_epoch}")
    print("=" * 65)

    t_total = time.time()

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            continue
        ds_cfg = DATASET_CONFIGS[dataset_name]

        for shot in args.shots:
            if shot not in ds_cfg.get("shots", args.shots):
                continue

            print(f"\n{'#'*65}")
            print(f"  {dataset_name} @ {shot}-shot")
            print(f"{'#'*65}")

            for run_id in range(args.start_run, config.n_runs):
                t0 = time.time()
                result = run_single_experiment(
                    dataset_name=dataset_name,
                    shot=shot,
                    run_id=run_id,
                    config=config,
                )
                elapsed = time.time() - t0

                tracker.log_run(
                    dataset=dataset_name,
                    shot=shot,
                    run_id=run_id,
                    accuracy=result["accuracy"] * 100,
                    epoch_accuracies=result["epoch_accuracies"],
                )
                print(f"  [Run {run_id}] logged in {elapsed:.0f}s")

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(tracker.format_table())
    save_path = tracker.save()
    print(f"\nTotal wall time: {time.time()-t_total:.0f}s")
    print(f"Results saved to {save_path}")
    return tracker


if __name__ == "__main__":
    main()
