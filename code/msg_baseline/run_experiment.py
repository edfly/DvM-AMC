"""
Main entry point for MSG* baseline experiments.

Usage:
    python -m msg_baseline.run_experiment              # Full 5-run experiment
    python -m msg_baseline.run_experiment --quick       # Quick smoke test
    python -m msg_baseline.run_experiment --datasets MUTAG PROTEINS  # Select datasets
"""

import argparse
import sys
import os

# Add parent directory to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msg_baseline.config import (
    ExperimentConfig, DATASET_CONFIGS, ALL_DATASETS, get_config
)
from msg_baseline.train_episodic import run_single_experiment
from msg_baseline.utils import ResultTracker


def main():
    parser = argparse.ArgumentParser(description="MSG* Baseline Experiments")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to run (default: all)"
    )
    parser.add_argument(
        "--shots", nargs="+", type=int, default=[1, 5],
        help="Shot settings to run (default: [1, 5])"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--start-run", type=int, default=0,
        help="Start from this run index (0-based, default: 0)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Override number of runs (default: from config, usually 5)"
    )
    args = parser.parse_args()

    config = get_config(quick_test=args.quick)
    if args.n_runs is not None:
        config.n_runs = args.n_runs
    tracker = ResultTracker()

    print("=" * 65)
    print("  MSG* Baseline Experiments — SpikeRPGNN Comparison Study")
    print(f"  Datasets: {args.datasets}")
    print(f"  Shots: {args.shots}")
    print(f"  Runs per config: {config.n_runs}")
    print(f"  Epochs: {config.epochs}")
    if config.quick_test:
        print("  ** QUICK TEST MODE **")
    print("=" * 65)

    total_configs = len(args.datasets) * len(args.shots)
    config_idx = 0

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"[WARN] Unknown dataset {dataset_name}, skipping.")
            continue

        ds_cfg = DATASET_CONFIGS[dataset_name]
        valid_shots = [s for s in args.shots if s in ds_cfg.get("shots", args.shots)]

        for shot in valid_shots:
            config_idx += 1
            print(f"\n{'#'*65}")
            print(f"  Config [{config_idx}/{total_configs}]: {dataset_name} @ {shot}-shot")
            print(f"{'#'*65}")

            for run_id in range(args.start_run, config.n_runs):
                result = run_single_experiment(
                    dataset_name=dataset_name,
                    shot=shot,
                    run_id=run_id,
                    config=config,
                )

                tracker.log_run(
                    dataset=dataset_name,
                    shot=shot,
                    run_id=run_id,
                    accuracy=result["accuracy"] * 100,  # store as percentage
                    epoch_accuracies=result["epoch_accuracies"],
                )

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 65)
    print(tracker.format_table())

    save_path = tracker.save()
    print("\nDone!")

    return tracker


if __name__ == "__main__":
    main()
