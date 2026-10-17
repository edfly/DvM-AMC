"""
Minimal debug: one training step, check if gradient flows to _c_log.
"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
import torch.optim as optim

ds_name, shot, seed = "MUTAG", 1, 42
ds_cfg = DATASET_CONFIGS[ds_name]
n_way = ds_cfg["n_way"]
set_seed(seed)

cfg = SpikeRPGNNConfig(epochs=1, episodes_per_epoch=4, test_episodes=10,
                         learnable_c=True, c_init=1.0)
dataset = TUGraphDataset(name=ds_name, tu_name=ds_cfg["tu_name"], train_ratio=0.8, seed=seed)
sampler = EpisodeSampler(dataset=dataset, n_way=n_way, k_shot=shot, q_query=5, seed=seed)
model = create_spikergnn(cfg, in_dim=max(dataset.num_features, 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer with c_params
u_th_params = [p for n, p in model.named_parameters() if 'u_th_learnable' in n]
c_params    = [p for n, p in model.named_parameters() if '_c_log' in n]
other_params = [p for n, p in model.named_parameters() if 'u_th_learnable' not in n and '_c_log' not in n]
print(f"c_params found: {len(c_params)}, _c_log requires_grad={c_params[0].requires_grad if c_params else 'N/A'}")
optimizer = optim.Adam([
    {'params': other_params, 'lr': 1e-3, 'weight_decay': 5e-4},
    {'params': u_th_params, 'lr': 1e-4, 'weight_decay': 0},
    {'params': c_params, 'lr': 1e-3, 'weight_decay': 0},
])

# One training step
episode = sampler.sample_episode(mode='train')
s_b, s_l, q_b, q_l = batch_episode_data(episode)
s_b, q_b = s_b.to(device), q_b.to(device)
s_l, q_l = s_l.to(device), q_l.to(device)

optimizer.zero_grad()
s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
logits = model.classify(q_z, s_z, s_l, n_way)
loss = nn.functional.cross_entropy(logits, q_l)
print(f"loss={loss.item():.4f}, c_before={model.manifold.c.item():.6f}, _c_log={model.manifold._c_log.item():.6f}")
loss.backward()

# Check gradients
print(f"\n=== Gradient check ===")
print(f"  _c_log.grad = {model.manifold._c_log.grad}")
if model.manifold._c_log.grad is not None:
    print(f"  |_c_log.grad| = {model.manifold._c_log.grad.norm().item():.6f}")
else:
    print("  WARNING: _c_log.grad is None!")

# Check if other params have gradients
has_grad = [(n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None]
print(f"  Params with grad: {len(has_grad)}/{sum(1 for _ in model.parameters())}")
if has_grad:
    print(f"  Sample grads: {has_grad[:3]}")

optimizer.step()
print(f"\n=== After optimizer step ===")
print(f"  _c_log = {model.manifold._c_log.item():.6f}")
print(f"  c = {model.manifold.c.item():.6f}")
