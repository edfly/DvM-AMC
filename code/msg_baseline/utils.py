"""
Utility functions for MSG* baseline experiments.

Covers: reproducible seeding, result formatting, logging, metrics.
"""

import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits and ground-truth labels."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


class ResultTracker:
    """
    Track experiment results across runs and configurations.
    """

    def __init__(self, save_dir: str = "./msg_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results: dict = {}
        self._load_existing()

    def _load_existing(self):
        """Load previously saved results to append new runs."""
        path = os.path.join(self.save_dir, "msgstar_results.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.results = json.load(f)
            except (json.JSONDecodeError, Exception):
                self.results = {}

    def log_run(
        self,
        dataset: str,
        shot: int,
        run_id: int,
        accuracy: float,
        epoch_accuracies: list = None,
    ):
        """Log results of a single run."""
        key = f"{dataset}_{shot}shot"
        if key not in self.results:
            self.results[key] = {
                "dataset": dataset,
                "shot": shot,
                "runs": [],
            }
        self.results[key]["runs"].append({
            "run_id": run_id,
            "accuracy": accuracy,
            "epoch_accuracies": epoch_accuracies or [],
        })

    def summarize(self) -> dict:
        """Compute mean ± std across runs for each config."""
        summary = {}
        for key, data in self.results.items():
            accs = [r["accuracy"] for r in data["runs"]]
            summary[key] = {
                **data,
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "accs": accs,
                "n_runs": len(accs),
            }
        return summary

    def format_table(self) -> str:
        """Format results as a LaTeX-compatible table row string."""
        lines = []
        lines.append(f"{'Dataset':<14} {'Shot':<6} {'Mean':>8} {'Std':>8} {'Runs':>6}")
        lines.append("-" * 46)

        summ = self.summarize()
        for key in sorted(summ.keys()):
            s = summ[key]
            lines.append(
                f"{s['dataset']:<14} {s['shot']}-shot  "
                f"{s['mean_acc']:>7.2f}% {s['std_acc']:>7.2f}% "
                f"{s['n_runs']:>6d}"
            )

        return "\n".join(lines)

    def save(self, filename: str = "msgstar_results.json"):
        """Save full results to JSON file."""
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            # Convert numpy types for JSON serialization
            serializable = {}
            for k, v in self.results.items():
                d = dict(v)
                d["runs"] = [
                    {**r, "accuracy": float(r["accuracy"])} for r in v["runs"]
                ]
                serializable[k] = d
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {path}")
        return path
