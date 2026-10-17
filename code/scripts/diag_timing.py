import time, os, sys
t0=time.time()
def stamp(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

stamp("import torch")
import torch
stamp(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.train import run_single_experiment
stamp("imports done")

for (T, reg, ep) in [(32,"truncated",2),(64,"full",2)]:
    if reg=="truncated":
        cfg=SpikeRPGNNConfig(ablation="full",use_dvm=True,use_metric_correction=True,
                             T=T,dvm_mode="truncated",dv_m_window=8,epochs=ep)
    else:
        cfg=SpikeRPGNNConfig(ablation="full",use_dvm=True,use_metric_correction=True,
                             T=T,dvm_mode="full_bptt_mc",dv_m_window=T,epochs=ep)
    stamp(f"=== MUTAG T={T} {reg} dv_m_window={cfg.dv_m_window} epochs={ep} : building cfg OK")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    t1=time.time()
    res=run_single_experiment(dataset_name="MUTAG",shot=1,run_id=0,config=cfg,device="cuda")
    dt=time.time()-t1
    peak=torch.cuda.max_memory_allocated()/1e6
    stamp(f"    done: acc={res['final_acc']*100:.2f}% peak={peak:.1f}MB in {dt:.1f}s "
          f"({dt/ep:.1f}s/epoch)")
    torch.cuda.empty_cache()
stamp("ALL DIAG DONE")
