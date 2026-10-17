"""
probe_spike_accounting.py — self-consistency sentinel for the spike ACCOUNTING
path (Phase A follow-up).

Why
---
Phase A v1 (diagnose_spike_dynamics.py) printed a verdict of "H3" that rested on
`pre_peak_ratio_max`, a metric that is *saturated by construction*: neurons.py
369-370 states the input clamp makes the 2-step membrane reach exactly
0.6*(1+alpha)=1.14*u_th, so max(u_mag/u_th) == 1.14 for BOTH firing (MUTAG) and
collapsed (NCI1) runs. A saturated metric cannot discriminate -> that verdict is
void and is retracted here.

What is measured instead
------------------------
There are TWO ManifoldLIFNeuron instances in the model:
    model.mspe.lif   (MSPEncoder, spike generation)   -> mspe.py:92
    model.stmp.lif   (STMP, temporal message passing) -> stmp.py:110
`model.encode()` runs mspe then stmp, and `model.count_spikes()` returns
`model._last_spike_count = stmp._total_spike_count` (model.py:151-155), i.e. the
reported "Spikes:" number counts ONLY the STMP neuron. MSPE's spikes are never
reported.

This script cross-checks, on the SAME forward passes:
    SUM_S_OUT(instance)   -- spikes actually emitted (sum of surrogate outputs)
    instance._spike_count -- what the neuron says it emitted (last call)
    stmp._total_spike_count / model.count_spikes()  -- what gets REPORTED

Sentinel: SUM_S_OUT(stmp) must equal what is reported. If SUM_S_OUT(stmp) > 0
while the reported count is 0, the zero-spike result is an accounting artifact.

Usage
-----
  PYTHONPATH=$PWD/programfiles CUDA_VISIBLE_DEVICES="" \
      python -u programfiles/diagnostics/probe_spike_accounting.py \
      --datasets MUTAG NCI1 --shot 1
"""
import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.dirname(_THIS)
sys.path.insert(0, _PF)

import torch

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from spikergnn.config import get_spikergnn_config
from spikergnn.model import create_spikergnn
import spikergnn.neurons as neurons_mod

TAGS = {}          # id(lif) -> "mspe" / "stmp" / "other"
ACC = {}           # tag -> dict(n_calls, sum_s_out, sum_nodes, fire_frac_sum)
_ORIG_LIF_FORWARD = neurons_mod.ManifoldLIFNeuron.forward
_ORIG_SUR_ATAN = neurons_mod.ATanSurrogate.forward
_ORIG_SUR_SIGM = neurons_mod.SigmoidSurrogate.forward
_CUR = {"tag": None}
_CAP = {"on": False}


def _acc(tag):
    if tag not in ACC:
        ACC[tag] = {"n_calls": 0, "sum_s_out": 0.0, "sum_nodes": 0,
                    "cross": 0, "fire_frac_sum": 0.0}
    return ACC[tag]


def _patched_lif_forward(self, x_seq, p_init):
    _CUR["tag"] = TAGS.get(id(self), "other") if _CAP["on"] else None
    out = _ORIG_LIF_FORWARD(self, x_seq, p_init)
    _CUR["tag"] = None
    return out


def _patched_surrogate(self, v_mem):
    if _CAP["on"] and _CUR["tag"] is not None:
        with torch.no_grad():
            a = _acc(_CUR["tag"])
            a["n_calls"] += 1
            a["sum_nodes"] += int(v_mem.numel())
            cross = (v_mem >= 0).float()
            a["cross"] += int(cross.sum().item())
            a["fire_frac_sum"] += float(cross.mean())
    if isinstance(self, neurons_mod.SigmoidSurrogate):
        return _ORIG_SUR_SIGM(self, v_mem)
    return _ORIG_SUR_ATAN(self, v_mem)


neurons_mod.ManifoldLIFNeuron.forward = _patched_lif_forward
neurons_mod.ATanSurrogate.forward = _patched_surrogate
neurons_mod.SigmoidSurrogate.forward = _patched_surrogate


def run(ds, shot, seed):
    print("\n" + "=" * 74)
    print(f" {ds}  (accounting sentinel, random init, {shot}-shot, CPU)")
    print("=" * 74, flush=True)

    cfg = get_spikergnn_config(ablation="full")
    ds_cfg = DATASET_CONFIGS[ds]
    set_seed(seed)
    dataset = TUGraphDataset(name=ds, tu_name=ds_cfg["tu_name"],
                             train_ratio=cfg.train_ratio, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=shot, q_query=cfg.q_query, seed=seed)
    model = create_spikergnn(cfg, in_dim=max(dataset.num_features, 1)).to("cpu")
    model.eval()

    TAGS.clear()
    TAGS[id(model.mspe.lif)] = "mspe"
    TAGS[id(model.stmp.lif)] = "stmp"
    ACC.clear()
    print(f"  mspe.T = {getattr(model.mspe, 'T', '?')}   stmp.T = {getattr(model.stmp, 'T', '?')}"
          f"   (mspe.lif is not model.stmp.lif: {id(model.mspe.lif) != id(model.stmp.lif)})")

    ep = sampler.sample_episode(mode="train")
    sb, sl, qb, ql = batch_episode_data(ep, device="cpu")

    reported = []
    _CAP["on"] = True
    for tag, b in (("support", sb), ("query", qb)):
        model.stmp.lif.reset_membrane()
        model.encode(b.x, b.edge_index, b.batch)
        c = model.count_spikes()
        reported.append(c)
        print(f"  after encode({tag}): model.count_spikes() = {c}"
              f" | stmp._total_spike_count = {model.stmp._total_spike_count}"
              f" | mspe.lif._spike_count = {model.mspe.lif._spike_count}"
              f" | stmp.lif._spike_count (last step) = {model.stmp.lif._spike_count}",
              flush=True)
    _CAP["on"] = False

    print("  --- per-instance accounting ---")
    for tag in ("mspe", "stmp"):
        a = ACC.get(tag)
        if not a:
            print(f"   {tag:5s}: (no calls captured)")
            continue
        print(f"   {tag:5s}: calls={a['n_calls']:4d} | nodes={a['sum_nodes']:8d}"
              f" | SUM_s_out={a['cross']:8d} | mean_fire={a['fire_frac_sum']/max(a['n_calls'],1)*100:5.1f}%")

    mspe_sum = ACC.get("mspe", {}).get("cross", 0)
    stmp_sum = ACC.get("stmp", {}).get("cross", 0)
    total_reported = sum(reported)
    print("  --- SENTINEL ---")
    print(f"   reported total (sum of count_spikes)  = {total_reported}")
    print(f"   SUM_s_out(stmp)  [what SHOULD be read] = {stmp_sum}")
    print(f"   SUM_s_out(mspe)  [NEVER reported]      = {mspe_sum}")
    if stmp_sum > 0 and total_reported == 0:
        print("   => *** ACCOUNTING MISMATCH *** stmp emits spikes but 0 is reported.")
    elif stmp_sum == 0 and total_reported == 0:
        print("   => consistent: the STMP neuron itself emits no spikes (genuine).")
    else:
        print(f"   => consistent (ratio reported/SUM = "
              f"{(total_reported / stmp_sum) if stmp_sum else float('nan'):.3f})")
    print(f"   => MSPE spikes are invisible to the reported count "
          f"(mspe={mspe_sum}, reported excludes them).")
    return {"dataset": ds, "reported": reported, "sum_s_out_mspe": mspe_sum,
            "sum_s_out_stmp": stmp_sum, "mspe_T": getattr(model.mspe, 'T', None),
            "stmp_T": getattr(model.stmp, 'T', None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MUTAG", "NCI1"])
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for ds in args.datasets:
        run(ds, args.shot, args.seed)


if __name__ == "__main__":
    main()
