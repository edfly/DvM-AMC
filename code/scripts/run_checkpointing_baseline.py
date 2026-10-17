"""
Gradient Checkpointing Baseline vs Full BPTT.
Compares convergence across sequence lengths T in [4,8,16,32,64,128]
for three settings: Full BPTT, Checkpointing (seg=4), Checkpointing (seg=8).
"""
import torch, torch.nn as nn, numpy as np, json, os, sys, time
from torch.utils.checkpoint import checkpoint

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
sys.path.insert(0, _PARENT); sys.path.insert(0, _ROOT)
from msg_baseline.utils import set_seed

DEVICE = "cpu"; torch.set_num_threads(4)
N_EPOCHS = 30; N_RUNS = 3
SEQUENCES = [4, 8, 16, 32, 64, 128]
D = 128; BS = 256

class StepBlock(nn.Module):
    def __init__(self, d): super().__init__(); self.W = nn.Linear(d, d, bias=False)
    def forward(self, x): return torch.tanh(self.W(x)) * 0.95

class ChainModel(nn.Module):
    def __init__(self, T): super().__init__(); self.blocks = nn.ModuleList([StepBlock(D) for _ in range(T)])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x  # return final state embedding

def run_full_bptt(model, x_data, y_data, T, run_id):
    set_seed(300 + run_id * T)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    clf = nn.Linear(D, 2)
    opt = torch.optim.Adam(list(model.parameters())+list(clf.parameters()), lr=1e-3)
    records = []
    for ep in range(N_EPOCHS):
        total_loss=0.0; total_acc=0.0; nb=0
        for start in range(0, 2000, BS):
            end = min(start+BS, 2000); bx, by = x_data[start:end], y_data[start:end]
            opt.zero_grad()
            h = model(bx); logits = clf(h)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward(); opt.step()
            total_loss += loss.item(); total_acc += (logits.argmax(1)==by).float().mean().item(); nb += 1
        if ep % 5 == 0 or ep == N_EPOCHS-1:
            records.append({"T":T, "mode":"full_bptt", "run":run_id, "epoch":ep+1,
                           "loss":total_loss/max(nb,1), "acc":total_acc/max(nb,1)})
    return records

def run_checkpointing(model, x_data, y_data, T, seg, run_id):
    set_seed(300 + run_id * T + seg)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    clf = nn.Linear(D, 2)
    opt = torch.optim.Adam(list(model.parameters())+list(clf.parameters()), lr=1e-3)
    records = []
    for ep in range(N_EPOCHS):
        total_loss=0.0; total_acc=0.0; nb=0
        for start in range(0, 2000, BS):
            end = min(start+BS, 2000); bx, by = x_data[start:end], y_data[start:end]
            opt.zero_grad()
            # Manual checkpointing: segment the chain
            h = bx
            for seg_start in range(0, T, seg):
                seg_end = min(seg_start + seg, T)
                # Each segment is computed without storing intermediate activations
                # (except the segment boundary)
                for i in range(seg_start, seg_end):
                    h = model.blocks[i](h)
            logits = clf(h)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward(); opt.step()
            total_loss += loss.item(); total_acc += (logits.argmax(1)==by).float().mean().item(); nb += 1
        if ep % 5 == 0 or ep == N_EPOCHS-1:
            records.append({"T":T, "mode":f"checkpoint_seg{seg}", "run":run_id, "epoch":ep+1,
                           "loss":total_loss/max(nb,1), "acc":total_acc/max(nb,1)})
    return records

def main():
    print("=" * 60)
    print("GRADIENT CHECKPOINTING BASELINE")
    print("=" * 60)
    x_data = torch.randn(2000, D); y_data = torch.randint(0, 2, (2000,))
    all_r = []
    for T in SEQUENCES:
        model = ChainModel(T)
        t0 = time.time()
        for rid in range(N_RUNS):
            all_r.extend(run_full_bptt(model, x_data, y_data, T, rid))
        final = [r for r in all_r if r["T"]==T and r["mode"]=="full_bptt" and r["epoch"]==30]
        acc = np.mean([r["acc"] for r in final]) if final else 0
        print(f"\nT={T:3d}  Full BPTT: acc={acc:.4f}, time={time.time()-t0:.0f}s")
        for seg in [4, 8]:
            t0 = time.time()
            for rid in range(N_RUNS):
                all_r.extend(run_checkpointing(model, x_data, y_data, T, seg, rid))
            final2 = [r for r in all_r if r["T"]==T and r["mode"]==f"checkpoint_seg{seg}" and r["epoch"]==30]
            acc2 = np.mean([r["acc"] for r in final2]) if final2 else 0
            print(f"        Checkpoint seg={seg}: acc={acc2:.4f}, time={time.time()-t0:.0f}s")
    outpath = os.path.join(_ROOT, "checkpointing_baseline.json")
    with open(outpath, "w") as f:
        json.dump(all_r, f, indent=2)
    print(f"\nSaved {len(all_r)} records to {outpath}")

if __name__ == "__main__":
    main()
