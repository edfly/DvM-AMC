"""
Positive control for Group F: is the metric-correction pathway actually live?

Group F measures Delta(cos) = cos(AMC) - cos(no correction) ~ 1e-4 at every
curvature. Before concluding "the correction is a near-identity operator", we
must rule out the alternative explanation "the correction never reaches the
weights, so of course it changes nothing".

Test: amplify the correction factor to (lambda_old^2 / lambda_new^2)^k for
k = 1, 4, 16 and watch Delta(cos). If the pathway is live, Delta(cos) must grow
with k. If it stays at ~1e-4 for k=16, the correction is NOT wired into the
weight gradient and the null result means something entirely different (a bug,
not a geometric fact).

Usage:  python poc_correction_amplify.py [dataset] [seed] [c]
"""
import sys
import os

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "programfiles"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler, batch_episode_data)
from msg_baseline.utils import set_seed  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
# NOTE: stmp.py imports `from spikergnn.dvm import PerStepMetricCorrection`.
# `spikergnn` is an alias package for programfiles/, so `spikergnn.dvm` and
# `dvm` are DIFFERENT module objects holding different class objects. Patching
# `dvm.PerStepMetricCorrection` silently does nothing to the running model --
# an earlier version of this script made exactly that mistake.
from spikergnn.dvm import PerStepMetricCorrection, conformal_factor  # noqa: E402

# Sanity: the object we patch must be the one stmp.py holds.
import spikergnn.stmp as _stmp  # noqa: E402
assert _stmp.PerStepMetricCorrection is PerStepMetricCorrection, \
    "patching the wrong PerStepMetricCorrection class object"

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def named_grads(model):
    return {n: p.grad.reshape(-1) for n, p in model.named_parameters()
            if p.grad is not None}


def pair_up(ga, gb):
    common = [n for n in ga if n in gb and ga[n].numel() == gb[n].numel()]
    return (torch.cat([ga[n] for n in common]),
            torch.cat([gb[n] for n in common]), len(common))


def cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def make_cfg(c, window=None, mc=True, T=16):
    kw = dict(hidden_dim=128, num_layers=2, T=T, tau=20.0, ablation="full",
              use_metric_correction=mc, lr=1e-3, epochs=1, use_dvm=True,
              c_init=c, learnable_c=False)
    kw["dvm_mode"] = "full_bptt_mc" if window is None else "truncated"
    if window is not None:
        kw["dv_m_window"] = window
    return SpikeRPGNNConfig(**kw)


def install_amplifier(k):
    """Replace the correction backward with factor**k (k=1 is the real thing)."""
    def backward(ctx, grad_output):
        p_old, p_new = ctx.saved_tensors
        c = ctx.c
        lam_old = conformal_factor(p_old, c)
        lam_new = conformal_factor(p_new, c)
        factor = (lam_old / lam_new.clamp(min=1e-5)) ** 2
        factor = factor ** k
        return torch.zeros_like(grad_output), grad_output * factor.unsqueeze(-1), None
    PerStepMetricCorrection.backward = staticmethod(backward)


def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "MUTAG"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    c = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    T, M = 16, 8

    set_seed(seed)
    ds = TUGraphDataset(name=ds_name, tu_name=ds_name, train_ratio=0.8, seed=seed)
    in_dim = max(ds.num_features, 1)
    sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=seed)
    ep = sampler.sample_episode(mode="train")
    s_b, s_l, q_b, q_l = batch_episode_data(ep, DEV)

    def loss_of(model):
        logits = model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        return F.cross_entropy(logits, q_l)

    print(f"dataset={ds_name} seed={seed} c={c} T={T} M={M} device={DEV}")
    print(f"{'k':>4} {'cos(amc)':>12} {'cos(nocorr)':>13} {'Delta':>12}")
    print("-" * 46)

    results = {}
    for k in (1, 4, 16):
        install_amplifier(k)
        # Same weights and same batch for every k, otherwise the comparison
        # between k values is meaningless.
        set_seed(seed)

        # reference: full graph, correction ALSO amplified (so both sides agree
        # on what "exact" means under the modified operator)
        ref = create_spikergnn(make_cfg(c, None, mc=True, T=T), in_dim=in_dim).to(DEV)
        ref.train()
        loss_of(ref).backward()
        g_ref = named_grads(ref)
        state = {a: b.detach().clone() for a, b in ref.state_dict().items()}
        del ref
        if DEV == "cuda":
            torch.cuda.empty_cache()

        vals = {}
        for tag, mc in (("amc", True), ("no_metric_correction", False)):
            m = create_spikergnn(make_cfg(c, M, mc=mc, T=T), in_dim=in_dim).to(DEV)
            m.load_state_dict(state, strict=False)
            m.train()
            loss_of(m).backward()
            g_obj, g_ref2, n = pair_up(named_grads(m), g_ref)
            vals[tag] = cos(g_obj, g_ref2)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

        d = vals["amc"] - vals["no_metric_correction"]
        results[k] = {"cos_amc": vals["amc"], "cos_nocorr": vals["no_metric_correction"],
                      "delta": d}
        print(f"{k:>4} {vals['amc']:>12.6f} {vals['no_metric_correction']:>13.6f} "
              f"{d:>+12.6f}")

    print("\nInterpretation:")
    grew = abs(results[16]["delta"]) > 20 * abs(results[1]["delta"])
    if grew:
        print("  PATHWAY IS LIVE: Delta grows sharply with k, so the correction")
        print("  does reach the weights. The k=1 null is therefore a REAL")
        print("  geometric statement: the correction factor is ~identity here.")
    else:
        print("  PATHWAY LOOKS DEAD: amplifying 16x barely changes Delta.")
        print("  The k=1 null would then be a BUG, not a geometric fact -- do")
        print("  NOT report 'the correction is a no-op' before fixing this.")


if __name__ == "__main__":
    main()
