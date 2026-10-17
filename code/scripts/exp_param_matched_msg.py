"""
Group C -- parameter-matched comparison against MSG*.

The paper's headline result (+27.3 points on MUTAG) compares SpikeRPGNN (232,850
parameters) with MSG* (3,424 parameters): a 68x capacity gap. The comparison
cannot distinguish "a better method" from "a bigger model". This script closes
that gap in BOTH directions and then re-runs the accuracy comparison.

Direction 1 (up):   raise MSG* hidden_dim until its parameter count lands within
                    +/-5% of SpikeRPGNN's.
Direction 2 (down): lower SpikeRPGNN hidden_dim until it lands within +/-5% of
                    the original MSG*'s.

Both directions are reported, because a one-sided match can be read as tuning
the baseline to fail. The paper then reports whichever pair is matched, and the
headline is rewritten from the matched numbers -- or withdrawn, if the advantage
disappears. That outcome is accepted in advance; it is a result, not a failure.

Usage
-----
    python exp_param_matched_msg.py --dry-run
    python exp_param_matched_msg.py --datasets MUTAG --epochs 50 --seeds 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "programfiles"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msg_baseline.models import MSGStarEncoder, RiemannianProtoClassifier  # noqa
from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler)
from msg_baseline.config import DATASET_CONFIGS  # noqa: E402
from msg_baseline.utils import set_seed  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

import run_msg_baseline_comparison as R  # noqa: E402  (reuse train/eval loops)

# The reused train/eval helpers hardcode `batch_episode_data(ep, device="cpu")`
# while placing the model on the requested device, so the original script only
# ever ran on CPU. We force every episode batch onto the real compute device
# through a wrapper. This changes only WHERE tensors live, not what is computed,
# and it is the only change we make to the reused protocol.
_orig_batch = R.batch_episode_data


def _batch_on_dev(ep, device=None, **kw):
    return _orig_batch(ep, device=DEV, **kw)


R.batch_episode_data = _batch_on_dev

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join(_HERE, "results_v3")
MATCH_TOL = 0.05           # +/-5% on parameter count


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_msg(in_dim, hidden_dim, num_layers=3):
    return MSGStarEncoder(in_dim=in_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, manifold_c=1.0,
                          epsilon=1e-4, tau=20.0).to(DEV)


def build_spike(in_dim, hidden_dim, num_layers=3, T=None):
    # SpikeRPGNNConfig asserts d_h + d_e == hidden_dim, so shrinking the model
    # requires scaling BOTH components. We keep the paper's 1:3 hyperbolic to
    # Euclidean ratio (32:96) so that the architecture, not just the width,
    # stays comparable.
    d_e = max(int(round(hidden_dim * 0.75)), 1)
    d_h = hidden_dim - d_e
    if d_h < 1:
        d_h, d_e = 1, hidden_dim - 1
    cfg = SpikeRPGNNConfig(
        hidden_dim=hidden_dim, d_h=d_h, d_e=d_e,
        num_layers=num_layers, tau=20.0,
        use_dvm=True, dvm_mode="truncated", use_metric_correction=True,
        ablation="full", dv_m_window=8,
        episodes_per_epoch=20, test_episodes=50,
        lr=1e-3, weight_decay=5e-4, epochs=1)
    return create_spikergnn(cfg, in_dim=in_dim).to(DEV)


def search_hidden(build_fn, in_dim, target, lo=8, hi=1024, other=3):
    """Binary-search hidden_dim so that n_params lands within MATCH_TOL.

    Parameter count is monotone in hidden_dim for both architectures, so a
    binary search is valid. We return the best candidate found even if the
    tolerance is not met (e.g. the grid cannot express the target), and record
    by how much it missed -- the reader must see that, not a silent success.
    """
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build_fn(in_dim, mid, other)
        n = n_params(m)
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
        rel = abs(n - target) / target
        if best is None or rel < best[2]:
            best = (mid, n, rel)
        if rel <= MATCH_TOL:
            return {"hidden_dim": mid, "params": n, "rel_gap": rel,
                    "matched": True}
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return {"hidden_dim": best[0], "params": best[1], "rel_gap": best[2],
            "matched": best[2] <= MATCH_TOL}


def train_one(dataset_name, method, hidden_dim, num_layers, seed, epochs, shot):
    """One run. Mirrors run_msg_baseline_comparison.py's protocol exactly."""
    ds_cfg = DATASET_CONFIGS[dataset_name]
    set_seed(seed)
    dataset = TUGraphDataset(name=dataset_name, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=42)
    in_dim = max(dataset.num_features, 1)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=shot, q_query=5, seed=seed)

    if method == "MSGStar":
        model = build_msg(in_dim, hidden_dim, num_layers)
        classifier = RiemannianProtoClassifier(manifold=model.manifold)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        best = 0.0
        for ep in range(1, epochs + 1):
            R.msg_train_epoch(model, classifier, sampler, opt)
            sch.step()
            if ep % 10 == 0 or ep == 1:
                a = R.msg_evaluate(model, classifier, dataset,
                                   {"n_way": ds_cfg["n_way"], "shot": shot,
                                    "eval_seed": seed + 10000})
                best = max(best, a)
        return {"best_acc": float(best), "params": n_params(model)}

    model = build_spike(in_dim, hidden_dim, num_layers)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best = 0.0
    for ep in range(1, epochs + 1):
        R.spikergnn_train_ep(model, sampler, opt)
        sch.step()
        if ep % 10 == 0 or ep == 1:
            ts = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                                k_shot=shot, q_query=5, seed=seed + 10000)
            a = R.spikergnn_evaluate(model, ts)
            best = max(best, a)
    return {"best_acc": float(best), "params": n_params(model)}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MUTAG,DD")
    ap.add_argument("--seeds", default="0,7,42")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT,
                                                  "param_matched_msg.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_dir = os.path.join(OUT_ROOT, "_done_C")
    os.makedirs(done_dir, exist_ok=True)

    # ---- baseline parameter counts (the 68x gap we are closing) ----
    counts = {}
    for ds in datasets:
        ds_cfg = DATASET_CONFIGS[ds]
        dataset = TUGraphDataset(name=ds, tu_name=ds_cfg["tu_name"],
                                 train_ratio=0.8, seed=42)
        in_dim = max(dataset.num_features, 1)
        m_msg = build_msg(in_dim, 32, 3)
        m_sp = build_spike(in_dim, 128, 3)
        counts[ds] = {"in_dim": in_dim,
                      "MSGStar_h32": n_params(m_msg),
                      "SpikeRPGNN_h128": n_params(m_sp),
                      "ratio": n_params(m_sp) / n_params(m_msg)}
        del m_msg, m_sp
        if DEV == "cuda":
            torch.cuda.empty_cache()
        print(f"{ds}: MSG*(h=32)={counts[ds]['MSGStar_h32']}  "
              f"SpikeRPGNN(h=128)={counts[ds]['SpikeRPGNN_h128']}  "
              f"ratio={counts[ds]['ratio']:.1f}x")

    if args.dry_run:
        print(f"\ndry-run: datasets={datasets} seeds={seeds} "
              f"epochs={args.epochs} shot={args.shot} device={DEV}")
        print("units =", len(datasets) * len(seeds) * 2)
        return

    # ---- solve the two matching problems ----
    matching = {}
    for ds in datasets:
        ds_cfg = DATASET_CONFIGS[ds]
        dataset = TUGraphDataset(name=ds, tu_name=ds_cfg["tu_name"],
                                 train_ratio=0.8, seed=42)
        in_dim = max(dataset.num_features, 1)
        tgt_up = counts[ds]["SpikeRPGNN_h128"]
        tgt_dn = counts[ds]["MSGStar_h32"]
        up = search_hidden(build_msg, in_dim, tgt_up)
        dn = search_hidden(build_spike, in_dim, tgt_dn)
        matching[ds] = {"up_msg_to_spike": up, "down_spike_to_msg": dn}
        print(f"{ds}: MSG* h={up['hidden_dim']} -> {up['params']} "
              f"(gap {up['rel_gap']:.3f}, matched={up['matched']})")
        print(f"{ds}: Spike h={dn['hidden_dim']} -> {dn['params']} "
              f"(gap {dn['rel_gap']:.3f}, matched={dn['matched']})")

    # ---- run the four arms ----
    arms = []
    for ds in datasets:
        up = matching[ds]["up_msg_to_spike"]["hidden_dim"]
        dn = matching[ds]["down_spike_to_msg"]["hidden_dim"]
        if up:
            arms.append((ds, "MSGStar_matched_up", "MSGStar", up))
        if dn:
            arms.append((ds, "SpikeRPGNN_matched_down", "SpikeRPGNN", dn))
        arms.append((ds, "SpikeRPGNN_original", "SpikeRPGNN", 128))
        arms.append((ds, "MSGStar_original", "MSGStar", 32))

    results = defaultdict(list)
    t0 = time.time()
    total = len(arms) * len(seeds)
    i = 0
    for ds, tag, method, h in arms:
        for seed in seeds:
            i += 1
            marker = os.path.join(done_dir, f"{ds}__{tag}__s{seed}.done")
            if os.path.exists(marker):
                results[(ds, tag)].append(json.load(open(marker, encoding="utf-8")))
                print(f"[{i}/{total}] SKIP {ds}/{tag}/s{seed}")
                continue
            print(f"[{i}/{total}] RUN  {ds}/{tag}/h{h}/s{seed}", flush=True)
            try:
                r = train_one(ds, method, h, 3, seed, args.epochs, args.shot)
                r.update({"dataset": ds, "arm": tag, "method": method,
                          "hidden_dim": h, "seed": seed,
                          "epochs": args.epochs, "shot": args.shot})
            except RuntimeError as e:
                r = {"dataset": ds, "arm": tag, "method": method,
                     "hidden_dim": h, "seed": seed, "error": str(e)[:300]}
            json.dump(r, open(marker, "w", encoding="utf-8"))
            results[(ds, tag)].append(r)
            json.dump({"partial": True,
                       "rows": [x for v in results.values() for x in v]},
                      open(args.out, "w", encoding="utf-8"), indent=2)

    # ---- gate: did the headline survive parameter matching? ----
    gates = {}
    for ds in datasets:
        def agg(tag):
            v = [r["best_acc"] for r in results[(ds, tag)] if "error" not in r]
            return (sum(v) / len(v), len(v)) if v else (None, 0)
        sp_o, n1 = agg("SpikeRPGNN_original")
        ms_o, n2 = agg("MSGStar_original")
        sp_d, n3 = agg("SpikeRPGNN_matched_down")
        ms_u, n4 = agg("MSGStar_matched_up")
        gates[ds] = {
            "original_gap_pp": None if sp_o is None or ms_o is None
                               else round((sp_o - ms_o) * 100, 2),
            "spike_lowered_vs_msg_original_pp": None
                if sp_d is None or ms_o is None else round((sp_d - ms_o) * 100, 2),
            "spike_original_vs_msg_raised_pp": None
                if sp_o is None or ms_u is None else round((sp_o - ms_u) * 100, 2),
            "n_runs": {"spike_orig": n1, "msg_orig": n2,
                       "spike_down": n3, "msg_up": n4},
        }
        g = gates[ds]
        surv = []
        for k in ("spike_lowered_vs_msg_original_pp",
                  "spike_original_vs_msg_raised_pp"):
            if g[k] is not None:
                surv.append(g[k] > 0)
        g["advantage_survives_matching"] = all(surv) if surv else None
        if g["advantage_survives_matching"] is False:
            g["verdict"] = ("The +27.3 headline does NOT survive parameter "
                            "matching. Per Phase 3, delete the headline and "
                            "restate the contribution as 'matches full-BPTT "
                            "accuracy at O(M) memory'.")
        elif g["advantage_survives_matching"] is True:
            g["verdict"] = ("Advantage survives in both matching directions; "
                            "report the matched numbers explicitly.")
        else:
            g["verdict"] = "insufficient data"

    out = {
        "config": {"datasets": datasets, "seeds": seeds, "epochs": args.epochs,
                   "shot": args.shot, "match_tolerance": MATCH_TOL,
                   "protocol": "mirrors run_msg_baseline_comparison.py"},
        "device": DEV, "torch": torch.__version__, "git_commit": git_commit(),
        "baseline_param_counts": counts,
        "matching_solution": matching,
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 1),
        "rows": [x for v in results.values() for x in v],
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print(f"\nelapsed {out['elapsed_s']}s -> {args.out}")
    print("\n" + "=" * 74)
    print(f"{'dataset':<9}{'arm':<26}{'hidden':>7}{'params':>10}{'acc':>9}")
    print("-" * 74)
    for (ds, tag), rs in sorted(results.items()):
        ok = [r for r in rs if "error" not in r]
        if not ok:
            print(f"{ds:<9}{tag:<26}{'-':>7}{'-':>10}  ERROR")
            continue
        acc = sum(r["best_acc"] for r in ok) / len(ok)
        print(f"{ds:<9}{tag:<26}{ok[0]['hidden_dim']:>7}"
              f"{ok[0]['params']:>10}{acc:>9.4f}")
    print("\nGATES:")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
