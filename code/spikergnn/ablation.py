"""
Ablation experiment management.

Defines the 6 ablation variants and provides utilities for running
ablation studies across multiple datasets and shot settings.

Variants:
  - "full": Complete SpikeRPGNN
  - "no_spikes": ManifoldLIFNeuron → identity activation
  - "euclidean": ProductManifold → pure Euclidean
  - "no_frechet": FPC uses arithmetic mean for both components
  - "no_dvm": DvM → full BPTT
  - "no_meta": Episodic training → standard supervised
"""

from typing import Dict, List, Any

from spikergnn.config import SpikeRPGNNConfig


# All supported ablation variant names
ABLATION_VARIANTS = {
    "full",          # Complete SpikeRPGNN
    "no_spikes",     # Remove spiking, use identity activation
    "euclidean",     # Pure Euclidean (hyperbolic → linear)
    "no_frechet",    # Arithmetic mean instead of Fréchet mean
    "no_dvm",        # Full BPTT instead of DvM truncation
    "no_meta",       # Standard supervised instead of episodic
}

# Human-readable labels for each variant
ABLATION_LABELS = {
    "full": "Full SpikeRPGNN",
    "no_spikes": "w/o Spikes",
    "euclidean": "w/o Riemannian (Euclidean)",
    "no_frechet": "w/o Fréchet",
    "no_dvm": "w/o DvM (BPTT)",
    "no_meta": "w/o Meta-Learning",
}


def make_ablation_config(
    base_config: SpikeRPGNNConfig,
    ablation: str,
) -> SpikeRPGNNConfig:
    """Create a SpikeRPGNNConfig for a specific ablation variant.

    Args:
        base_config: Base configuration to modify.
        ablation: Ablation variant name.

    Returns:
        New SpikeRPGNNConfig with ablation-specific modifications.
    """
    assert ablation in ABLATION_VARIANTS, f"Unknown ablation: {ablation}"

    # Create a new config with the ablation field set
    config_dict = {
        k: getattr(base_config, k)
        for k in base_config.__dataclass_fields__
    }
    config_dict["ablation"] = ablation

    # Variant-specific overrides
    if ablation == "no_dvm":
        config_dict["use_dvm"] = False

    return SpikeRPGNNConfig(**config_dict)


def get_all_ablation_configs(
    base_config: SpikeRPGNNConfig,
) -> Dict[str, SpikeRPGNNConfig]:
    """Generate configurations for all ablation variants.

    Args:
        base_config: Base configuration to derive variants from.

    Returns:
        Dict mapping variant name to SpikeRPGNNConfig.
    """
    configs = {}
    for variant in ABLATION_VARIANTS:
        configs[variant] = make_ablation_config(base_config, variant)
    return configs


# Bug 2 fix: alias for backward compatibility
get_ablation_configs = get_all_ablation_configs


def run_ablation_study(
    base_config: SpikeRPGNNConfig,
    datasets: List[str],
    shots: List[int],
    n_runs: int = 5,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Run complete ablation study across all variants.

    For each variant x dataset x shot combination, runs n_runs
    independent experiments and collects results.

    Args:
        base_config: Base SpikeRPGNNConfig.
        datasets: List of dataset names.
        shots: List of shot values (e.g., [1, 5]).
        n_runs: Number of independent runs per configuration.
        device: Device to run on.

    Returns:
        Nested dict: results[variant][dataset_shot] = {accs, mean, std, ...}
    """
    from spikergnn.train import run_single_experiment
    import numpy as np

    ablation_configs = get_all_ablation_configs(base_config)
    results = {}

    for variant, config in ablation_configs.items():
        results[variant] = {}

        for dataset_name in datasets:
            for shot in shots:
                key = f"{dataset_name}_{shot}shot"
                accs = []
                spike_counts = []

                for run_id in range(n_runs):
                    exp_result = run_single_experiment(
                        dataset_name=dataset_name,
                        shot=shot,
                        run_id=run_id,
                        config=config,
                        device=device,
                    )
                    accs.append(exp_result["final_acc"])
                    spike_counts.append(exp_result["total_spikes"])

                results[variant][key] = {
                    "dataset": dataset_name,
                    "shot": shot,
                    "variant": variant,
                    "accs": accs,
                    "mean_acc": float(np.mean(accs)),
                    "std_acc": float(np.std(accs)),
                    "spike_counts": spike_counts,
                    "mean_spikes": float(np.mean(spike_counts)),
                }

    return results
