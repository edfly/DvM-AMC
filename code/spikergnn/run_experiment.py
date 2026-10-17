"""
Main experiment runner for SpikeRPGNN.

CLI entry point: python -m spikergnn.run_experiment

Usage:
  python -m spikergnn.run_experiment [OPTIONS]

Options:
  --dataset NAME     Dataset name (default: MUTAG)
  --shot N           Shot number (default: 1)
  --n-runs N         Number of runs (default: 5)
  --quick            Quick test mode (3 epochs, 1 run)
  --ablation NAME    Ablation variant (default: full)
  --all-datasets     Run on all 5 datasets
  --full-study       Run complete study (all datasets x all shots x all ablations)
  --device DEVICE    Device (default: cpu)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.config import DATASET_CONFIGS, ALL_DATASETS
from msg_baseline.utils import set_seed

from spikergnn.config import SpikeRPGNNConfig, get_spikergnn_config
from spikergnn.train import run_single_experiment
from spikergnn.ablation import run_ablation_study, ABLATION_VARIANTS
from spikergnn.energy import estimate_energy, count_model_params
from spikergnn.statistics import run_statistical_tests
from spikergnn.tables import (
    format_accuracy_table,
    format_ablation_table,
    format_energy_table,
    format_stat_table,
)


def run_standard(
    dataset_name: str,
    shot: int,
    n_runs: int,
    config: SpikeRPGNNConfig,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run standard experiment on one dataset and shot setting.

    Args:
        dataset_name: TUDataset name.
        shot: Number of support examples per class.
        n_runs: Number of independent runs.
        config: SpikeRPGNNConfig instance.
        device: Device string.

    Returns:
        Dict with aggregated results.
    """
    accs = []
    spike_counts = []
    all_results = []

    for run_id in range(n_runs):
        print(f"\n{'='*60}")
        print(f"  {dataset_name} | {shot}-shot | Run {run_id+1}/{n_runs}")
        print(f"{'='*60}")

        result = run_single_experiment(
            dataset_name=dataset_name,
            shot=shot,
            run_id=run_id,
            config=config,
            device=device,
        )
        accs.append(result["final_acc"])
        spike_counts.append(result["total_spikes"])
        all_results.append(result)

        print(f"  Accuracy: {result['final_acc']*100:.2f}%")
        print(f"  Spikes: {result['total_spikes']}")

    # Energy estimation
    from spikergnn.model import create_spikergnn
    in_dim = max(DATASET_CONFIGS[dataset_name].get("n_features", 7), 1)
    # Approximate param count
    model = create_spikergnn(config, in_dim=7)
    model_params = count_model_params(model)
    avg_spikes = int(np.mean(spike_counts))
    energy = estimate_energy(avg_spikes, model_params, config.T)

    summary = {
        "dataset": dataset_name,
        "shot": shot,
        "n_runs": n_runs,
        "accs": accs,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "spike_counts": spike_counts,
        "energy": energy,
        "all_results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {dataset_name} | {shot}-shot")
    print(f"  Mean Acc: {summary['mean_acc']*100:.2f} ± {summary['std_acc']*100:.2f}%")
    print(f"  Energy Reduction: {energy['reduction_pct']:.1f}%")
    print(f"{'='*60}")

    return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpikeRPGNN experiment runner"
    )
    parser.add_argument(
        "--dataset", type=str, default="MUTAG",
        choices=ALL_DATASETS,
        help="Dataset name (default: MUTAG)"
    )
    parser.add_argument(
        "--shot", type=int, default=1,
        choices=[1, 5],
        help="Number of support examples per class (default: 1)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of independent runs (default: 5)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode (3 epochs, 1 run)"
    )
    parser.add_argument(
        "--ablation", type=str, default="full",
        choices=list(ABLATION_VARIANTS),
        help="Ablation variant (default: full)"
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Run on all 5 datasets"
    )
    parser.add_argument(
        "--full-study", action="store_true",
        help="Run complete ablation study"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (default: cpu)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./spikergnn_results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Create config
    config = get_spikergnn_config(
        quick_test=args.quick,
        ablation=args.ablation,
    )
    if not args.quick:
        config.n_runs = args.n_runs

    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()

    if args.full_study:
        # Complete ablation study
        print("="*60)
        print("  FULL ABLATION STUDY")
        print("="*60)
        datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
        shots = [1, 5]
        results = run_ablation_study(
            base_config=config,
            datasets=datasets,
            shots=shots,
            n_runs=config.n_runs,
            device=args.device,
        )

        # Statistical tests
        stats = run_statistical_tests(results)

        # Generate tables
        acc_table = format_accuracy_table(results)
        abl_table = format_ablation_table(results)

        # Save results
        save_path = os.path.join(args.save_dir, "full_study_results.json")
        with open(save_path, "w") as f:
            # Convert numpy types for JSON
            serializable = {}
            for variant, vdata in results.items():
                serializable[variant] = {}
                for key, kdata in vdata.items():
                    if isinstance(kdata, dict):
                        serializable[variant][key] = {
                            k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in kdata.items()
                            if k != "accs"
                        }
                        serializable[variant][key]["accs"] = [
                            float(a) for a in kdata.get("accs", [])
                        ]
            json.dump(serializable, f, indent=2)

        # Save tables
        with open(os.path.join(args.save_dir, "table_accuracy.tex"), "w") as f:
            f.write(acc_table)
        with open(os.path.join(args.save_dir, "table_ablation.tex"), "w") as f:
            f.write(abl_table)

    elif args.all_datasets:
        # Run on all datasets
        all_results = {}
        for ds_name in ALL_DATASETS:
            result = run_standard(
                dataset_name=ds_name,
                shot=args.shot,
                n_runs=config.n_runs,
                config=config,
                device=args.device,
            )
            all_results[ds_name] = result

        # Save results
        save_path = os.path.join(args.save_dir, "all_datasets_results.json")
        with open(save_path, "w") as f:
            json.dump({
                ds: {
                    "mean_acc": r["mean_acc"],
                    "std_acc": r["std_acc"],
                }
                for ds, r in all_results.items()
            }, f, indent=2)

    else:
        # Single experiment
        result = run_standard(
            dataset_name=args.dataset,
            shot=args.shot,
            n_runs=config.n_runs,
            config=config,
            device=args.device,
        )

        # Save results
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.shot}shot_{args.ablation}.json"
        )
        with open(save_path, "w") as f:
            json.dump({
                "mean_acc": result["mean_acc"],
                "std_acc": result["std_acc"],
                "accs": result["accs"],
                "energy": result["energy"],
            }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
