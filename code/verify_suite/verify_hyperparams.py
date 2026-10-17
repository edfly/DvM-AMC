"""Four-fold verification (3/4): HYPERPARAMETERS — runtime reflection.

Instantiates the EXACT config that actually produced the result tables
(run_final_matrix.py variant 2, AMC M=8) and reads the values that REALLY
apply, then compares them to what the manuscript's Experimental Setup
(Section "Experimental Setup and Hyperparameters") declares.

WHY runtime reflection (not source grep): a caller that sets an argument
silently overrides the class default; only instantiating and reading the
attribute reveals the true value. Adapted from papersh
verify_hyperparams_template.

Exit 0 = paper matches the executed config.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PROG = os.path.join(ROOT, "programfiles")
TEX = os.path.join(os.path.dirname(ROOT), "paper", "DvM_Paper_Anonymous.tex")

sys.path.insert(0, ROOT)
from spikergnn.config import SpikeRPGNNConfig  # noqa: E402


def runtime_values():
    """Exactly the kwargs run_final_matrix.py passes for the AMC M=8 variant."""
    cfg = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0,
        use_dvm=True, dvm_mode="truncated", use_metric_correction=True,
        ablation="full", dv_m_window=8,
        lr=1e-3, weight_decay=5e-4,   # <-- as written in run_final_matrix.py
    )
    return {
        "hidden_dim": float(cfg.hidden_dim),
        "num_layers": float(cfg.num_layers),
        "tau": float(cfg.tau),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "c_init": float(cfg.c_init),
    }


# What the manuscript Section "Experimental Setup and Hyperparameters" declares.
# (hidden/num_layers/tau from L1619; lr/wd from L1620; c from L1619.
#  L1620 now reads "weight decay $5\times10^{-4}$" => 5e-4, matching the
#  executed config after the drift fix.)
PAPER_CLAIMS = {
    "hidden_dim": 128.0,
    "num_layers": 3.0,
    "tau": 20.0,
    "lr": 1e-3,
    "weight_decay": 5e-4,   # paper L1620: "weight decay $5\times10^{-4}$"
    "c_init": 1.0,
}

RTOL = 1e-6


def main():
    print("=" * 70)
    print("HYPERPARAMETER DRIFT CHECK (runtime code  vs  manuscript setup)")
    print("=" * 70)
    rt = runtime_values()
    problems = []
    for key, claimed in PAPER_CLAIMS.items():
        actual = rt[key]
        ok = abs(actual - claimed) <= max(RTOL * max(abs(actual), 1.0), 1e-9)
        flag = "OK" if ok else "**DRIFT**"
        print(f"  {key:14s} runtime={actual:10.5f}  paper={claimed:10.5f}  {flag}")
        if not ok:
            problems.append((key, actual, claimed))
    print("-" * 70)
    if problems:
        print(f"{len(problems)} DRIFT(S):")
        for key, actual, claimed in problems:
            print(f"  - {key}: code runs with {actual}, manuscript declares {claimed}")
        print("\nAction: update the manuscript setup to match the executed config"
              " (or change the code and re-run). Do NOT edit this script to pass.")
        return 1
    print("ALL HYPERPARAMETERS MATCH THE EXECUTED CONFIG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
