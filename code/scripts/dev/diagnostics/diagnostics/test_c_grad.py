"""Test if gradient flows to _c_log using torch.autograd.grad."""
import sys, torch, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

set_seed(42)
ds = TUGraphDataset(name='MUTAG', tu_name='MUTAG', train_ratio=0.8, seed=42)
sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=42)
cfg = SpikeRPGNNConfig(learnable_c=True, c_init=1.0)
model = create_spikergnn(cfg, in_dim=7)

ep = sampler.sample_episode(mode='train')
s_b, s_l, q_b, q_l = batch_episode_data(ep)
s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
logits = model.classify(q_z, s_z, s_l, 2)
loss = torch.nn.functional.cross_entropy(logits, q_l)
print(f"loss={loss.item():.4f}, loss.requires_grad={loss.requires_grad}")

# Try autograd.grad
try:
    g = torch.autograd.grad(loss, model.manifold._c_log, retain_graph=True)
    print(f"grad w.r.t. _c_log: {g[0].item():.6f}")
except Exception as e:
    print(f"ERROR: {e}")

# Also check if s_z depends on c
print(f"\ns_z[0] requires_grad: {s_z[0].requires_grad}")
print(f"s_z[0] grad_fn: {s_z[0].grad_fn}")
