"""
diagnose_spike_dynamics.py — Phase A v2 (per-instance, sentinel-checked).

RETRACTION NOTE (v1 -> v2)
--------------------------
v1's verdict rule keyed on `pre_peak_ratio_max = max(u_mag)/u_th`, which is
SATURATED BY CONSTRUCTION: neurons.py:369-370 states the input clamp makes the
2-step membrane reach exactly 0.6*(1+alpha) = 1.14*u_th, so the max equals 1.14
for BOTH the firing (MUTAG) and the collapsed (NCI1) case. A saturated metric
cannot discriminate, so v1's "H3" verdict is VOID. It is replaced by:

  * per-instance accounting. There are TWO ManifoldLIFNeuron instances:
        model.mspe.lif  (MSPEncoder) -> mspe.py:92
        model.stmp.lif  (STMP)       -> stmp.py:110
    Both get T=8; mspe does 1 call x 8 steps, stmp does 8 calls x 1 step, so
    they contribute 8:8 surrogate calls per encode. The reported "Spikes:" /
    energy number is `stmp._total_spike_count` ONLY (model.py:151-155) --
    MSPE's spikes are never reported.
  * the discriminating quantity is the per-instance MEAN fire fraction and the
    per-instance SUM of surrogate outputs, cross-checked against what is
    reported (sentinel: SUM_s_out(stmp) must equal the reported count).

Gates
-----
  G1  reproduce the NCI1 zero-report;
  G2  locate which instance stopped firing (mspe vs stmp);
  G3  sentinel: SUM_s_out(stmp) vs reported -- mismatch => accounting artifact,
      match => the reported zero is faithful for STMP;
  G4  quantify the UNREPORTED MSPE spikes (energy-accounting bias).

Usage
-----
  PYTHONPATH=$PWD/programfiles CUDA_VISIBLE_DEVICES="" \
      python -u programfiles/diagnostics/diagnose_spike_dynamics.py \
      --datasets NCI1 MUTAG --epochs 12 --shot 1
"""
import argparse
import json
import math
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.dirname(_THIS)
sys.path.insert(0, _PF)

import torch
import torch.optim as optim

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from spikergnn.config import get_spikergnn_config
from spikergnn.model import create_spikergnn
from spikergnn.train import train_one_epoch, evaluate
import spikergnn.neurons as neurons_mod


# ---------------------------------------------------------------------------
# Instrumentation (monkeypatch only; no source edit)
# ---------------------------------------------------------------------------
TAGS = {}          # id(lif) -> "mspe" / "stmp" / "other"
_ORIG_LIF_FORWARD = neurons_mod.ManifoldLIFNeuron.forward
_ORIG_SUR_ATAN = neurons_mod.ATanSurrogate.forward
_ORIG_SUR_SIGM = neurons_mod.SigmoidSurrogate.forward
_CUR = {"tag": None}
_CAP = {"on": False, "store": None}


def _blank():
    return {"n_calls": 0, "nodes": 0, "cross": 0, "fire_sum": 0.0,
            "ratio_sum": 0.0, "ratio_max": 0.0, "w_pre_sum": 0.0,
            "w_pre_max": 0.0, "u_th_last": float("nan"), "cross_calls": 0}


def _rec(tag):
    st = _CAP["store"]
    if st is None:
        return None
    return st.setdefault(tag, _blank())


def _patched_lif_forward(self, x_seq, p_init):
    _CUR["tag"] = TAGS.get(id(self), "other") if _CAP["on"] else None
    rec = _CUR["tag"] is not None
    if rec:
        with torch.no_grad():
            d = x_seq.shape[-1]
            it = x_seq.reshape(-1, d)
            w_pre = float(self.W(self.input_ln(it)).norm(dim=-1).mean() / math.sqrt(self.d_total))
            st = _rec(_CUR["tag"])
            st["w_pre_sum"] += w_pre
            st["w_pre_max"] = max(st["w_pre_max"], w_pre)
            st["u_th_last"] = float(self.u_th_learnable.detach().clamp(min=0.05))
    out = _ORIG_LIF_FORWARD(self, x_seq, p_init)
    _CUR["tag"] = None
    return out


def _patched_surrogate(self, v_mem):
    if _CAP["on"] and _CUR["tag"] is not None:
        with torch.no_grad():
            st = _rec(_CUR["tag"])
            u_th = st["u_th_last"] if st["u_th_last"] == st["u_th_last"] else 1.0
            cross = (v_mem >= 0).float()
            st["n_calls"] += 1
            st["nodes"] += int(v_mem.numel())
            st["cross"] += int(cross.sum().item())
            st["fire_sum"] += float(cross.mean())
            if cross.sum() > 0:
                st["cross_calls"] += 1
            st["ratio_sum"] += float((v_mem / u_th + 1.0).mean())
            st["ratio_max"] = max(st["ratio_max"], float(v_mem.max() / u_th + 1.0))
    if isinstance(self, neurons_mod.SigmoidSurrogate):
        return _ORIG_SUR_SIGM(self, v_mem)
    return _ORIG_SUR_ATAN(self, v_mem)


neurons_mod.ManifoldLIFNeuron.forward = _patched_lif_forward
neurons_mod.ATanSurrogate.forward = _patched_surrogate
neurons_mod.SigmoidSurrogate.forward = _patched_surrogate


def _summ(store):
    out = {}
    for tag, st in store.items():
        n = max(st["n_calls"], 1)
        out[tag] = {
            "n_calls": st["n_calls"],
            "nodes": st["nodes"],
            "sum_s_out": st["cross"],                     # TRUE spikes emitted
            "fire_pct": 100.0 * st["fire_sum"] / n,       # discriminating
            "cross_call_pct": 100.0 * st["cross_calls"] / n,
            "ratio_mean": st["ratio_sum"] / n,
            "ratio_max": st["ratio_max"],                 # SATURATED (see header)
            "w_pre_mean": st["w_pre_sum"] / n,
            "u_th": st["u_th_last"],
        }
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def build(ds, shot, epochs, seed):
    cfg = get_spikergnn_config(ablation="full")
    cfg.n_runs = 1
    cfg.epochs = epochs
    ds_cfg = DATASET_CONFIGS[ds]
    set_seed(seed)
    dataset = TUGraphDataset(name=ds, tu_name=ds_cfg["tu_name"],
                             train_ratio=cfg.train_ratio, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=shot, q_query=cfg.q_query, seed=seed)
    model = create_spikergnn(cfg, in_dim=max(dataset.num_features, 1)).to("cpu")
    return cfg, dataset, sampler, model


def run(ds, epochs, shot, seed, verbose_every):
    print("\n" + "=" * 78)
    print(f" DATASET = {ds}   (ablation=full, {shot}-shot, {epochs} epochs, CPU)  [Phase A v2]")
    print("=" * 78, flush=True)

    cfg, dataset, sampler, model = build(ds, shot, epochs, seed)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    TAGS.clear()
    TAGS[id(model.mspe.lif)] = "mspe"
    TAGS[id(model.stmp.lif)] = "stmp"
    print(f"  mspe.T={getattr(model.mspe, 'T', '?')}  stmp.T={getattr(model.stmp, 'T', '?')}")

    ep = sampler.sample_episode(mode="train")
    sb, sl, qb, ql = batch_episode_data(ep, device="cpu")

    traj = []
    for e in range(epochs):
        _CAP["on"], _CAP["store"] = True, {}
        tm = train_one_epoch(model, sampler, opt, cfg, "cpu")
        _CAP["on"] = False
        ts = _summ(_CAP["store"])

        model.eval()
        _CAP["on"], _CAP["store"] = True, {}
        with torch.no_grad():
            model.stmp.lif.reset_membrane()
            model.encode(sb.x, sb.edge_index, sb.batch)
        _CAP["on"] = False
        ps = _summ(_CAP["store"])
        model.train()

        em = evaluate(model, sampler, cfg, "cpu") if ((e + 1) % verbose_every == 0 or e == 0) else None
        rec = {"epoch": e + 1, "train_acc": tm["mean_acc"], "train_loss": tm["mean_loss"],
               "reported_train_spikes": tm["spike_count"],
               "test_acc": (em["mean_acc"] if em else None),
               "reported_test_spikes": (em["spike_count"] if em else None),
               "train": ts, "probe": ps}
        traj.append(rec)

        m, s = ts.get("mspe", {}), ts.get("stmp", {})
        test_s = f"{em['mean_acc']*100:5.1f}%" if em else "  --  "
        print(f"  ep{e+1:2d} | loss={tm['mean_loss']:7.4f} tr={tm['mean_acc']*100:5.1f}% te={test_s}"
              f" | REP={tm['spike_count']:8d} | mspe fire={m.get('fire_pct', float('nan')):5.1f}%"
              f" Ssum={m.get('sum_s_out', 0):7d} | stmp fire={s.get('fire_pct', float('nan')):5.1f}%"
              f" Ssum={s.get('sum_s_out', 0):7d} | stmp|p|={s.get('ratio_mean', float('nan')):.3f}"
              f" w/u_th={s.get('w_pre_mean', float('nan'))/max(s.get('u_th',1) or 1,1e-9):.3f}",
              flush=True)

    lines = _verdict(traj)
    out_dir = os.path.join(os.path.dirname(_PF), "rerun_outputs", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"spike_dynamics_v2_{ds}.json"), "w") as f:
        json.dump({"dataset": ds, "shot": shot, "epochs": epochs, "seed": seed,
                   "trajectory": traj, "verdict": lines}, f, indent=2)
    print("  --- verdict ---", flush=True)
    for ln in lines:
        print("   " + ln, flush=True)
    return traj, lines


def _verdict(traj):
    last = traj[-1]
    ts = last["train"]
    m, s = ts.get("mspe", {}), ts.get("stmp", {})
    rep = last["reported_train_spikes"]
    ssum = s.get("sum_s_out", 0)
    out = [f"final: reported={rep}  mspe fire={m.get('fire_pct', float('nan')):.1f}%"
           f" (Ssum={m.get('sum_s_out', 0)})  stmp fire={s.get('fire_pct', float('nan')):.1f}%"
           f" (Ssum={ssum})"]
    # G3 sentinel
    if ssum > 0 and rep == 0:
        out.append(f"=> G3 MISMATCH: stmp emitted {ssum} spikes but 0 is reported "
                   f"-> ACCOUNTING ARTIFACT.")
    elif ssum == 0 and rep == 0:
        out.append("=> G3 consistent: the STMP neuron itself emits no spikes at the "
                   "final epoch (the reported zero is faithful FOR STMP).")
    else:
        out.append(f"=> G3 consistent (reported/SUM_s_out(stmp) = "
                   f"{(rep / ssum) if ssum else float('nan'):.3f}).")
    # G2 location
    if (s.get("fire_pct", 0) or 0) < 1.0 and (m.get("fire_pct", 0) or 0) > 1.0:
        out.append(f"=> G2: collapse is LOCALISED TO STMP; MSPE still fires "
                   f"{m.get('fire_pct', 0):.1f}% -> the reported count (STMP-only) "
                   f"understates total network spiking.")
    elif (s.get("fire_pct", 0) or 0) < 1.0 and (m.get("fire_pct", 0) or 0) < 1.0:
        out.append("=> G2: BOTH instances stopped firing.")
    else:
        out.append(f"=> G2: no collapse (stmp fires {s.get('fire_pct', 0):.1f}%).")
    out.append("trend reported : " + ", ".join(str(r["reported_train_spikes"]) for r in traj))
    out.append("trend mspe fire: " + ", ".join(
        f"{r['train'].get('mspe', {}).get('fire_pct', float('nan')):.0f}%" for r in traj))
    out.append("trend stmp fire: " + ", ".join(
        f"{r['train'].get('stmp', {}).get('fire_pct', float('nan')):.0f}%" for r in traj))
    out.append("trend stmp w/u_th: " + ", ".join(
        f"{r['train'].get('stmp', {}).get('w_pre_mean', float('nan')) / max((r['train'].get('stmp', {}).get('u_th') or 1), 1e-9):.3f}" for r in traj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NCI1", "MUTAG"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose-every", type=int, default=5)
    args = ap.parse_args()

    all_out = {}
    for ds in args.datasets:
        _, lines = run(ds, args.epochs, args.shot, args.seed, args.verbose_every)
        all_out[ds] = lines
    print("\n===== PHASE-A v2 SUMMARY =====", flush=True)
    for ds, v in all_out.items():
        print(f"[{ds}]", flush=True)
        for ln in v:
            print("   " + ln, flush=True)


if __name__ == "__main__":
    main()
