"""
SpikeRPGNN: Spiking Riemannian Prototypical Graph Neural Network.

A framework for few-shot graph classification combining:
  - Manifold-valued LIF neurons (state evolution on product manifold)
  - Riemannian graph convolution (geodesic message passing)
  - Spiking temporal message passing (event-driven sparse communication)
  - Fréchet prototypical construction (manifold-aware class prototypes)
  - Geodesic distance classification (learnable gate fusion)
  - Differentiation via Manifold (truncated BPTT with tangent-bundle gradients)
"""

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.manifolds import ProductManifold
from spikergnn.neurons import ManifoldLIFNeuron, SigmoidSurrogate
from spikergnn.mspe import MSPEncoder
from spikergnn.rgc import RiemannianGraphConv
from spikergnn.stmp import SpikingTemporalMP
from spikergnn.fpc import FrechetProtoConstructor
from spikergnn.gdc import GeodesicDistClassifier
from spikergnn.dvm import DvMFunction
from spikergnn.model import SpikeRPGNN, create_spikergnn
from spikergnn.train import EpisodicTrainer
from spikergnn.ablation import get_ablation_configs
from spikergnn.tables import generate_results_table

__all__ = [
    "SpikeRPGNNConfig",
    "ProductManifold",
    "ManifoldLIFNeuron",
    "SigmoidSurrogate",
    "MSPEncoder",
    "RiemannianGraphConv",
    "SpikingTemporalMP",
    "FrechetProtoConstructor",
    "GeodesicDistClassifier",
    "DvMFunction",
    "SpikeRPGNN",
    "create_spikergnn",
    "EpisodicTrainer",
    "get_ablation_configs",
    "generate_results_table",
]
