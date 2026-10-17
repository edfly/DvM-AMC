"""Check that the use_metric_correction switch actually changes the computation.

Two independent probes:
  1. instrument PerStepMetricCorrection.apply -- is it called at all?
  2. read back the diagnostic lambda^2 prefactor -- is it far from 1?
"""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "programfiles"))
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
import spikergnn.dvm as dvm_mod

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- probe 1: wrap the correction op with a call counter ----
calls = {"n": 0}
_orig = dvm_mod.PerStepMetricCorrection.apply


@staticmethod
def counting_apply(*a, **k):
    calls["n"] += 1
    return _orig(*a, **k)


dvm_mod.PerStepMetricCorrection.apply = counting_apply

set_seed(42)
ds = TUGraphDataset(name="MUTAG", tu_name="MUTAG", train_ratio=0.8, seed=42)
sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=42)
s_b, s_l, q_b, q_l = batch_episode_data(sampler.sample_episode(mode="train"), DEV)

for mc in (True, False):
    calls["n"] = 0
    cfg = SpikeRPGNNConfig(hidden_dim=128, num_layers=2, T=16, tau=20.0,
                           ablation="full", use_metric_correction=mc,
                           use_dvm=True, dvm_mode="truncated", dv_m_window=8,
                           lr=1e-3, epochs=1)
    set_seed(42)
    m = create_spikergnn(cfg, in_dim=max(ds.num_features, 1)).to(DEV)
    m.train()
    logits = m(s_b.x, s_b.edge_index, s_b.batch,
               q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
    torch.nn.functional.cross_entropy(logits, q_l).backward()
    print(f"use_metric_correction={mc!s:5s} -> PerStepMetricCorrection called {calls['n']:>4d} times")

# ---- probe 2: how far is the lambda^2 prefactor from 1? ----
set_seed(42)
cfg = SpikeRPGNNConfig(hidden_dim=128, num_layers=2, T=16, tau=20.0,
                       ablation="full", use_metric_correction=True,
                       use_dvm=True, dvm_mode="full_bptt_mc", lr=1e-3, epochs=1)
m = create_spikergnn(cfg, in_dim=max(ds.num_features, 1)).to(DEV)
m.train()
with torch.no_grad():
    z = m.encode(s_b.x, s_b.edge_index, s_b.batch)
z_h = z[0]
from dvm_core import conformal_factor
lam = conformal_factor(z_h)
print(f"\nhyperbolic embedding norms : mean={z_h.norm(dim=-1).mean():.4f} "
      f"max={z_h.norm(dim=-1).max():.4f}")
print(f"conformal factor lambda    : mean={lam.mean():.4f} min={lam.min():.4f} max={lam.max():.4f}")
print(f"=> lambda^2 ratio spread   : {(lam.max()/lam.min())**2:.4f}")
print("   (a correction near 1.0 means the metric correction can barely move the gradient)")
