"""Probe MSPE firing rate and per-forward node counts on the EVAL protocol.

Purpose (P0-C): the matrix energy numbers used `model.count_spikes()` which
covers ONLY the STMP LIF. The MSPE LIF fires at a *pinned* design rate
(exactly 50% of node-steps, because inputs saturate the 0.6*u_th clamp and the
neuron fires every other step: 0.6 -> 1.14 -> reset -> 0.6 -> ...).  To
recompute the energy table with MSPE included we need, per dataset:

  1. empirical verification that the MSPE pinned rate holds on the eval
     protocol (measured here with an instrumented forward),
  2. the mean node count per support/query forward over test episodes
     (MSPE spikes per forward = fire_rate * T * N_nodes).

Runs on CPU with an UNTRAINED model: the MSPE count is a structural property
(inputs saturate the clamp), not a trained one -- verified separately over 12
epochs of real training on MUTAG/NCI1 (exactly 50.0% every epoch, see
rerun_outputs/diagnostics/spike_dynamics_v2_*.json).

Output: rerun_outputs/diagnostics/mspe_rate_<DS>.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikergnn import neurons as neurons_mod  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig  # noqa: E402
from spikergnn.model import create_spikergnn  # noqa: E402
from msg_baseline.config import DATASET_CONFIGS  # noqa: E402
from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset,
    EpisodeSampler,
    batch_episode_data,
)


# --------------------------------------------------------------------------
# instrumentation: tag which LIF instance (mspe / stmp) is calling the
# surrogate, then accumulate v_mem-crossing statistics.
# --------------------------------------------------------------------------
_CAP = {"on": False}
_CUR = {"tag": None, "model": None}
_REC: dict = {}

_ORIG_SUR_ATAN = neurons_mod.ATanSurrogate.forward
_ORIG_SUR_SIGM = neurons_mod.SigmoidSurrogate.forward
_ORIG_LIF_FORWARD = neurons_mod.ManifoldLIFNeuron.forward


def _rec(tag):
    if tag not in _REC:
        _REC[tag] = {"n_calls": 0, "nodes": 0, "cross": 0, "u_th_last": 1.0}
    return _REC[tag]


def _patched_lif_forward(self, x_seq, p_init):
    if _CAP["on"]:
        model = _CUR["model"]
        if model is not None and self is model.mspe.lif:
            _CUR["tag"] = "mspe"
        elif model is not None and self is model.stmp.lif:
            _CUR["tag"] = "stmp"
        else:
            _CUR["tag"] = None
        if _CUR["tag"]:
            _rec(_CUR["tag"])["u_th_last"] = float(
                self.u_th_learnable.detach().clamp(min=0.05)
            )
    out = _ORIG_LIF_FORWARD(self, x_seq, p_init)
    _CUR["tag"] = None
    return out


def _patched_surrogate(self, v_mem):
    if _CAP["on"] and _CUR["tag"]:
        with torch.no_grad():
            st = _rec(_CUR["tag"])
            u_th = st["u_th_last"]
            st["n_calls"] += 1
            st["nodes"] += int(v_mem.numel())
            st["cross"] += int((v_mem >= 0).sum().item())
    if isinstance(self, neurons_mod.SigmoidSurrogate):
        return _ORIG_SUR_SIGM(self, v_mem)
    return _ORIG_SUR_ATAN(self, v_mem)


neurons_mod.ManifoldLIFNeuron.forward = _patched_lif_forward
neurons_mod.ATanSurrogate.forward = _patched_surrogate
neurons_mod.SigmoidSurrogate.forward = _patched_surrogate


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["MUTAG", "PROTEINS", "NCI1", "DD"])
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=50,
                    help="eval episodes to sample (the matrix protocol used "
                         "50 per evaluation)")
    ap.add_argument("--seed", type=int, default=52,
                    help="matrix used seed_base+run_id*100+shot*10; any seed "
                         "samples the same test-split distribution")
    ap.add_argument("--out-dir", default="rerun_outputs/diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"
    T = 8
    cfg = SpikeRPGNNConfig()

    for ds in args.datasets:
        ds_cfg = DATASET_CONFIGS[ds]
        seed = args.seed

        dataset = TUGraphDataset(
            name=ds, tu_name=ds_cfg["tu_name"],
            train_ratio=cfg.train_ratio, seed=seed,
        )
        sampler = EpisodeSampler(
            dataset=dataset, n_way=ds_cfg["n_way"], k_shot=args.shot,
            q_query=cfg.q_query, seed=seed,
        )
        in_dim = max(dataset.num_features, 1)
        model = create_spikergnn(cfg, in_dim=in_dim).to(device)
        _CUR["model"] = model

        # ---- measured pass: run the eval protocol with instrumentation ----
        _REC.clear()
        _CAP["on"] = True
        n_supp_nodes, n_query_nodes = [], []
        with torch.no_grad():
            for _ in range(args.n_episodes):
                ep = sampler.sample_episode(mode="test")
                sb, _, qb, _ = batch_episode_data(ep, device=device)
                n_supp_nodes.append(int(sb.x.shape[0]))
                n_query_nodes.append(int(qb.x.shape[0]))
                model.stmp.lif.reset_membrane()
                _ = model.encode(sb.x, sb.edge_index, sb.batch)
                _ = model.encode(qb.x, qb.edge_index, qb.batch)
        _CAP["on"] = False

        mspe = _REC.get("mspe", {"nodes": 0, "cross": 0})
        stmp = _REC.get("stmp", {"nodes": 0, "cross": 0})
        mspe_rate = 100.0 * mspe["cross"] / max(mspe["nodes"], 1)
        stmp_rate = 100.0 * stmp["cross"] / max(stmp["nodes"], 1)

        # MSPE spikes for the matrix eval protocol (50 episodes, 2 forwards
        # each): pinned_rate * T * total nodes over all forwards.
        mean_supp = sum(n_supp_nodes) / len(n_supp_nodes)
        mean_query = sum(n_query_nodes) / len(n_query_nodes)
        mspe_ops_eval = (mspe_rate / 100.0) * T * (mean_supp + mean_query) * 50

        out = {
            "dataset": ds,
            "shot": args.shot,
            "n_episodes": args.n_episodes,
            "mspe_fire_pct": mspe_rate,
            "stmp_fire_pct_untrained": stmp_rate,
            "mean_support_nodes": mean_supp,
            "mean_query_nodes": mean_query,
            "mspe_ops_per_eval": mspe_ops_eval,
            "T": T,
            "note": "MSPE count is structural (clamp-saturated); verified "
                    "over 12 training epochs on MUTAG/NCI1 at exactly 50.0% "
                    "(spike_dynamics_v2_*.json) and at init here.",
        }
        path = os.path.join(args.out_dir, f"mspe_rate_{ds}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[{ds}] mspe fire={mspe_rate:.2f}%  stmp(untrained)={stmp_rate:.2f}%  "
              f"E[N_supp]={mean_supp:.1f}  E[N_query]={mean_query:.1f}  "
              f"mspe_ops/eval={mspe_ops_eval:.0f}  -> {path}")


if __name__ == "__main__":
    main()
