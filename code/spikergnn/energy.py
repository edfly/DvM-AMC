"""
Energy estimation for SpikeRPGNN.

Computes synaptic operation counts and energy consumption:
  - SNN energy: spike_ops * 0.9 pJ
  - ANN energy: ann_ops * 4.6 pJ
  - Reduction: (1 - E_snn / E_ann) * 100%

References: SpikeRPGNN paper §6.3.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from spikergnn.model import SpikeRPGNN
from spikergnn.config import SpikeRPGNNConfig


# Energy constants from paper §6.3
SNN_OP_ENERGY_PJ = 0.9   # pJ per synaptic operation (SNN)
ANN_OP_ENERGY_PJ = 4.6   # pJ per operation (ANN)


def count_synaptic_ops(
    spike_count: int,
    model_params: int,
    T: int = 8,
) -> Dict[str, int]:
    """Count synaptic operations for SNN and equivalent ANN.

    Args:
        spike_count: Total number of non-zero spikes from forward pass.
        model_params: Total number of model parameters.
        T: Number of time steps.

    Returns:
        Dict with 'snn_ops' (spike-driven) and 'ann_ops' (dense).
    """
    # SNN ops: number of active (non-zero) spike transmissions
    snn_ops = spike_count

    # ANN ops: all weights fire every step (dense computation)
    ann_ops = model_params * T

    return {
        "snn_ops": snn_ops,
        "ann_ops": ann_ops,
    }


def count_model_params(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_energy(
    spike_count: int,
    model_params: int,
    T: int = 8,
) -> Dict[str, float]:
    """Estimate energy consumption for SNN vs ANN.

    Args:
        spike_count: Total non-zero spikes from forward pass.
        model_params: Total trainable parameters.
        T: Number of time steps.

    Returns:
        Dict with energy estimates:
          - 'e_snn_mJ': SNN energy in millijoules
          - 'e_ann_mJ': ANN energy in millijoules
          - 'reduction_pct': Energy reduction percentage
          - 'snn_ops': Synaptic operation count
          - 'ann_ops': ANN operation count
    """
    ops = count_synaptic_ops(spike_count, model_params, T)

    # Convert pJ to mJ: 1 pJ = 1e-9 mJ
    e_snn_pj = ops["snn_ops"] * SNN_OP_ENERGY_PJ
    e_ann_pj = ops["ann_ops"] * ANN_OP_ENERGY_PJ

    e_snn_mj = e_snn_pj * 1e-9
    e_ann_mj = e_ann_pj * 1e-9

    if e_ann_mj > 0:
        reduction_pct = (1 - e_snn_mj / e_ann_mj) * 100
    else:
        reduction_pct = 0.0

    return {
        "e_snn_mJ": e_snn_mj,
        "e_ann_mJ": e_ann_mj,
        "reduction_pct": reduction_pct,
        "snn_ops": ops["snn_ops"],
        "ann_ops": ops["ann_ops"],
    }


def estimate_energy_for_model(
    model: SpikeRPGNN,
    T: int = 8,
) -> Dict[str, float]:
    """Estimate energy consumption for a SpikeRPGNN model.

    Uses the spike count from the last forward pass and the model's
    parameter count.

    Args:
        model: SpikeRPGNN model instance.
        T: Number of time steps.

    Returns:
        Energy estimation dict.
    """
    spike_count = model.count_spikes()
    model_params = count_model_params(model)
    return estimate_energy(spike_count, model_params, T)
