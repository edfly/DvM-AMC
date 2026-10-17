"""
Batch-Size Ablation — Extended to 30 Epochs.
Measures convergence speed vs batch size on a synthetic embedding chain.
"""
import torch, torch.nn as nn, numpy as np, json, os, sys, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
sys.path.insert(0, _PARENT); sys.path.insert(0, _ROOT)
from msg_baseline.utils import set_seed

DEVICE = "cpu"; torch.set_num_threads(4)
N_EPOCHS = 30; N_RUNS = 5
BATCH_SIZES = [128, 256, 512, 1024]
D = 128

class ChainModel(nn.Module):
    """T-step contractive chain for embedding simulation."""
    def __init__(self, T=16):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(T)])
        self.cls = nn.Linear(D, 2)
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x)) * 0.95
        return self.cls(x)

def run(bs, run_id):
    set_seed(100 + run_id * bs)
    model = ChainModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Fixed data
    x_data = torch.randn(2000, D)
    y_data = torch.randint(0, 2, (2000,))
    records = []
    for ep in range(N_EPOCHS):
        total_loss = 0.0; total_acc = 0.0; n = 0
        for start in range(0, 2000, bs):
            end = min(start + bs, 2000)
            bx, by = x_data[start:end], y_data[start:end]
            opt.zero_grad()
            logits = model(bx)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(1) == by).float().mean().item()
            n += 1
        if ep % 2 == 0 or ep == N_EPOCHS - 1:
            records.append({"run": run_id, "epoch": ep+1, "bs": bs,
                           "loss": total_loss/max(n,1), "acc": total_acc/max(n,1)})
    return records

def main():
    print("=" * 60)
    print("BATCH-SIZE ABLATION — 30 Epochs, 5 Runs")
    print("=" * 60)
    all_r = []
    for bs in BATCH_SIZES:
        t0 = time.time()
        print(f"\nBatch size = {bs}...")
        for rid in range(N_RUNS):
            all_r.extend(run(bs, rid))
        final = [r for r in all_r if r["bs"]==bs and r["epoch"]==30]
        accs = [r["acc"] for r in final]
        print(f"  Epoch 30: {np.mean(accs):.4f} +/- {np.std(accs):.4f}, {time.time()-t0:.0f}s")
    outpath = os.path.join(_ROOT, "batch_ablation_30ep.json")
    with open(outpath, "w") as f:
        json.dump(all_r, f, indent=2)
    print(f"\nSaved {len(all_r)} records to {outpath}")

if __name__ == "__main__":
    main()
