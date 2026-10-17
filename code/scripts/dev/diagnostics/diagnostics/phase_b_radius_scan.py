"""phase_b_radius_scan.py -- Phase B: causal test of the H2 mechanism chain.

Phase A v2 verdict (see post/attribution_analysis_plan.md S3): the zero-spike
collapse on NCI1/DD is REAL training dynamics (H2), localised to the STMP LIF:
  embedding radius -> Poincare metric scale -> input drive w_in_pre
  -> membrane peak u_mag/u_th -> firing.
H1 (threshold drift) and H3 (counting artifact) were excluded.

Phase B tests the chain CAUSALLY with two intervention arms (monkeypatch only,
no source edits):

  B1  threshold sweep: set stmp.lif.u_th_learnable to v in {0.3, 0.5, 1.0, 2.0}
      (frozen). NOTE: config.u_threshold does NOT reach the neuron -- the
      learnable threshold is hard-initialised to 1.0 (neurons.py:218), so the
      sweep must set the parameter directly.
  B2  input-drive sweep: multiply the STMP LIF input x_seq by s in
      {0.5, 1.0, 2.0, 4.0, 8.0}. This bypasses the radius->drive link and
      tests drive sufficiency directly.

Pre-registered verdict (written here, not decided after the fact):
  * fire% recovers monotonically with s AND test acc rises >chance+5pts
        -> H2 causal, representations still informative;
  * fire% recovers but acc stays ~chance
        -> drive restored but the representation itself collapsed
           (consistent with the radius story, negative for accuracy);
  * fire% does not recover
        -> H2 incomplete, escalate.

Usage
-----
  PYTHONPATH=$PWD/programfiles CUDA_VISIBLE_DEVICES="" \
      python -u programfiles/diagnostics/phase_b_radius_scan.py \
      --datasets NCI1 MUTAG --epochs 15 --device cpu
"""

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.dirname(_THIS)
sys.path.insert(0, _PF)

import torch

from msg_baseline.data_utils import (
    TUGraphDataset,
    EpisodeSampler,
    batch_episode_data,
)
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from spikergnn.config import get_spikergnn_config
from spikergnn.model import create_spikergnn
from spikergnn.train import train_one_epoch, evaluate
import spikergnn.neurons as neurons_mod

# ---------------------------------------------------------------------------
# instance tagging + optional input scaling for the STMP LIF
# ---------------------------------------------------------------------------
TAGS = {}                     # id(module) -> "mspe"/"stmp" (lif + children)
SCALES = {}                   # id(lif) -> float input scale (default 1.0)
_FIRE = {"on": False, "spikes": 0.0, "nodes": 0, "calls": 0}
_ORIG_LIF_FORWARD = neurons_mod.ManifoldLIFNeuron.forward
_ORIG_SUR_ATAN = neurons_mod.ATanSurrogate.forward
_ORIG_SUR_SIGM = neurons_mod.SigmoidSurrogate.forward


def _patched_lif_forward(self, x_seq, p_init):
    s = SCALES.get(id(self), 1.0)
    if s != 1.0:
        x_seq = x_seq * s
    return _ORIG_LIF_FORWARD(self, x_seq, p_init)


def _patched_surrogate(self, v_mem):
    """Count TRUE spikes of the tagged neuron -- same statistic as Phase A v2.

    NB (bug fix, 2026-10-02): the previous version read
    ``model.count_spikes()`` right after ``model.encode(...)``. That helper
    returns ``self._last_spike_count`` of the LAST forward pass only, so it
    measured a single encoding step and reported ~50% firing on *every*
    dataset (including NCI1, whose STMP LIF genuinely goes silent). It had
    zero discriminating power. Counting the surrogate threshold crossings
    restores the Phase A v2 statistic (spikes / nodes), which is comparable.
    """
    if _FIRE["on"] and TAGS.get(id(self)) == "stmp":
        with torch.no_grad():
            _FIRE["spikes"] += float((v_mem >= 0).float().sum())
            _FIRE["nodes"] += int(v_mem.numel())
            _FIRE["calls"] += 1
    if isinstance(self, neurons_mod.SigmoidSurrogate):
        return _ORIG_SUR_SIGM(self, v_mem)
    return _ORIG_SUR_ATAN(self, v_mem)


neurons_mod.ManifoldLIFNeuron.forward = _patched_lif_forward
neurons_mod.ATanSurrogate.forward = _patched_surrogate
neurons_mod.SigmoidSurrogate.forward = _patched_surrogate


def fire_monitored(fn, *a, **kw):
    """Run fn with STMP spike counting enabled; return (out, fire_pct)."""
    _FIRE.update(on=True, spikes=0.0, nodes=0, calls=0)
    out = fn(*a, **kw)
    _FIRE["on"] = False
    return out, 100.0 * _FIRE["spikes"] / max(_FIRE["nodes"], 1)


def tag_model(model):
    TAGS.clear()
    TAGS[id(model.mspe.lif)] = "mspe"
    TAGS[id(model.stmp.lif)] = "stmp"
    # children (the surrogate envelope lives on the LIF) share the parent tag
    for root in (model.mspe.lif, model.stmp.lif):
        for m in root.modules():
            TAGS[id(m)] = TAGS[id(root)]


# ---------------------------------------------------------------------------
def run_arm(dataset, shot, epochs, device, u_th=None, scale=None, seed_base=42):
    """Train one short-schedule arm and return final firing/accuracy stats."""
    cfg = get_spikergnn_config()
    seed = seed_base  # fixed seed across arms: differences come from the arm
    set_seed(seed)

    ds_cfg = DATASET_CONFIGS[dataset]
    ds_name = dataset
    dataset = TUGraphDataset(
        name=dataset, tu_name=ds_cfg["tu_name"],
        train_ratio=cfg.train_ratio, seed=seed,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"], k_shot=shot,
        q_query=cfg.q_query, seed=seed,
    )
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(cfg, in_dim=in_dim).to(device)
    tag_model(model)
    SCALES.clear()
    if scale is not None:
        SCALES[id(model.stmp.lif)] = scale
    if u_th is not None:
        with torch.no_grad():
            model.stmp.lif.u_th_learnable.fill_(float(u_th))
        model.stmp.lif.u_th_learnable.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    fire_curve = []
    ev, ev_fire, acc_last = None, float("nan"), float("nan")
    for epoch in range(epochs):
        # train-side firing (same counting window as Phase A v2: the whole
        # training epoch), so the two phases are directly comparable
        _, tr_fire = fire_monitored(train_one_epoch, model, sampler,
                                    optimizer, cfg, device)
        # eval-side firing + accuracy on the same epoch
        ev, ev_fire = fire_monitored(evaluate, model, sampler, cfg, device)
        acc_last = ev["mean_acc"]
        fire_curve.append({"epoch": epoch + 1, "train_fire": tr_fire,
                           "eval_fire": ev_fire,
                           "eval_acc": acc_last * 100.0})
        print(f"    [{ds_name}] ep {epoch+1:2d} | train fire={tr_fire:5.1f}% "
              f"| eval fire={ev_fire:5.1f}% | acc={acc_last*100:5.2f}%",
              flush=True)

    model.eval()
    stmp_fire_last = ev_fire
    tail = [e["eval_fire"] for e in fire_curve[-3:]] or [stmp_fire_last]
    stmp_fire_tail3 = sum(tail) / len(tail)

    return {
        "u_th": u_th, "scale": scale,
        "stmp_fire_pct_eval": stmp_fire_last,
        "stmp_fire_pct_tail3": stmp_fire_tail3,
        "fire_curve": fire_curve,
        "test_acc": acc_last,
        "u_th_learnable_final": float(
            model.stmp.lif.u_th_learnable.detach()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NCI1", "MUTAG"])
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", nargs="+",
                    default=["base", "uth0.3", "uth0.5", "uth1.0", "uth2.0",
                             "s0.5", "s1.0", "s2.0", "s4.0", "s8.0"],
                    help="subset of arms to run (smoke: base s2.0)")
    ap.add_argument("--out-dir", default="rerun_outputs/diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    CHANCE = {"MUTAG": 50.0, "PROTEINS": 50.0, "NCI1": 50.0, "DD": 50.0}

    for ds in args.datasets:
        rows = []
        for arm in args.arms:
            if arm == "base":
                r = run_arm(ds, args.shot, args.epochs, args.device)
            elif arm.startswith("uth"):
                r = run_arm(ds, args.shot, args.epochs, args.device,
                            u_th=float(arm[3:]))
            elif arm.startswith("s"):
                r = run_arm(ds, args.shot, args.epochs, args.device,
                            scale=float(arm[1:]))
            else:
                raise ValueError(f"unknown arm {arm}")
            r["arm"] = arm
            rows.append(r)
        print(f"[{ds}] {arm:>8} | stmp fire={r['stmp_fire_pct_eval']:5.1f}% "
              f"(tail3 {r['stmp_fire_pct_tail3']:5.1f}%) "
              f"| acc={r['test_acc']*100:5.2f}% "
              f"| u_th_final={r['u_th_learnable_final']:.3f}", flush=True)

        # pre-registered verdict on the scale arm family
        s_arms = [r for r in rows if r["scale"] is not None]
        s_arms.sort(key=lambda r: r["scale"])
        # Oscillation-robust statistic: NCI1 STMP firing is NOT monotone in
        # epoch (observed 5.5% -> 22.0% train / 0.0% -> 50.0% eval between
        # ep6 and ep7), so a single end-of-schedule number lands on an
        # arbitrary phase of the oscillation and is not comparable across
        # arms.  Use the mean over the last 3 epochs (pre-registered here,
        # before any full arm had finished).
        fires = [r["stmp_fire_pct_tail3"] for r in s_arms]
        accs = [r["test_acc"] * 100 for r in s_arms]

        # ---- self-consistency sentinel -------------------------------------
        # If STMP firing is ALREADY saturated at scale=1.0 there is no collapse
        # to repair, and "fire% monotone in s" is trivially true (a constant
        # series satisfies `b >= a`), so the naive rule below would emit a
        # false "H2-causal" verdict on datasets that never collapse (e.g.
        # MUTAG holds ~50% STMP firing throughout).  Such datasets are declared
        # non-informative for H2 instead of being read as positive evidence.
        ref = next((r for r in s_arms if abs(r["scale"] - 1.0) < 1e-9), None)
        fire_ref = ref["stmp_fire_pct_tail3"] if ref is not None else float("nan")
        if not s_arms:
            verdict = ("no-scale-arms: run the s* family "
                       "(s0.5 s1.0 s2.0 s4.0 s8.0) to enable the verdict")
        elif fire_ref >= 40.0:
            verdict = (f"no-collapse-at-base: STMP firing already saturated "
                       f"({fire_ref:.1f}% at scale=1.0) -- dataset does not "
                       f"test H2; verdict suppressed")
        else:
            monotonic_fire = all(b >= a - 1e-9 for a, b in zip(fires, fires[1:]))
            recovered = (fires[-1] - fire_ref) > 5.0      # s8.0 vs s1.0
            acc_recovered = max(accs) > CHANCE.get(ds, 50.0) + 5.0
            if monotonic_fire and recovered and acc_recovered:
                verdict = "H2-causal: firing and accuracy both recover with drive"
            elif monotonic_fire and recovered:
                verdict = "drive-restored-but-representation-collapsed"
            else:
                verdict = "H2-incomplete: firing does not recover with drive"
        print(f"[{ds}] verdict: {verdict}", flush=True)

        out = {"dataset": ds, "shot": args.shot, "epochs": args.epochs,
               "device": args.device, "rows": rows, "verdict": verdict}
        path = os.path.join(args.out_dir, f"phase_b_{ds}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[{ds}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
