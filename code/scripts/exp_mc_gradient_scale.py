"""机制探针：度量校正在真实训练轨迹上到底把梯度改了多少？

出发点（一个必须解释的矛盾）
----------------------------
B 组: truncated M<T 时 MC 与 NoCorr 的单步梯度夹角 |Δcos| 中位 3e-6；
D 组: 同一套代码、同一 seed、n=9 训练 50 epoch 后，MC 比 NoCorr 低
      6.56pp (MUTAG, p_FDR=0.011) 与 1.84pp (PROTEINS, p_FDR=0.016)，
      两个数据集方向一致。
1% 量级的梯度缩放不足以直接解释 6.56pp，因此必须测出：
  (a) 训练过程中 MC 的净梯度缩放因子到底是多少（是否随 epoch 漂移）；
  (b) 累计的 PerStepMetricCorrection factor（窗口端点比）实测值；
  (c) 该缩放是"整体等比"（方向不变、只改模长）还是改变方向。

做法
----
训练一个 use_metric_correction=True 的 truncated(M=8) 模型，每 K epoch 存
权重快照；对每个快照，在**同一个固定 batch**上分别以 MC=on / MC=off 构建
两个模型（权重完全相同），比较全部共享参数的梯度：
    gnorm_ratio = ||g_on|| / ||g_off||     <- 净缩放
    cos         = cos(g_on, g_off)         <- 方向变化
并用 forward hook 采集每一步 PerStepMetricCorrection 的 factor，累乘得到
窗口端点比的实测值。

判定（预注册，写死在 check_gates 里）
------------------------------------
  G1 若 gnorm_ratio 全程落在 [0.99, 1.01] 且 cos > 0.9999，
     -> 结论 A：单步效应确为可忽略的等比缩放；训练后 6.56pp 的差异
        只能归因于长期动力学放大，论文必须如此表述，不得声称
        "度量校正改善/损害了梯度"。
  G2 若任一快照 gnorm_ratio 落在 [0.99,1.01] 之外或 cos < 0.999，
     -> 结论 B：MC 在训练中确实显著改变梯度，须报告该快照的 epoch
        与量级，并重新评估 6.56pp 的因果解释。
  G3 哨兵：MC=on 与 MC=off 用同一权重同一 batch，若两者梯度逐位相同
     (rel_err < 1e-12) 则开关未生效，判 FAILURE。

用法
----
    python exp_mc_gradient_scale.py --epochs 20 --seeds 0,7
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.join(_HERE, "programfiles")
for _p in (_PF, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler, batch_episode_data)
from msg_baseline.utils import set_seed  # noqa: E402
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from spikergnn.dvm import PerStepMetricCorrection

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join(_HERE, "results_v3")


def make_cfg(T, window, mc, metric_clamp=None):
    kw = dict(hidden_dim=128, num_layers=3, T=T, tau=20.0,
              ablation="full", use_metric_correction=mc,
              lr=1e-3, weight_decay=5e-4,
              episodes_per_epoch=20, test_episodes=50,
              use_dvm=True, dvm_mode="truncated", dv_m_window=window,
              metric_clamp=metric_clamp)
    return SpikeRPGNNConfig(**kw)


def named_grads(model):
    return {n: p.grad.reshape(-1) for n, p in model.named_parameters()
            if p.grad is not None}


def _factors_hook(store):
    """Capture the per-step correction factor actually used in backward."""
    def hook(module, grad_in, grad_out):  # pragma: no cover - runtime only
        return None
    return hook


class FactorRecorder:
    """Wrap PerStepMetricCorrection.backward to record the factor it applies.

    We patch at the class level so the recorder sees exactly the object that
    stmp.py imports (the module alias is a distinct object otherwise -- a bug
    hit once already).
    """
    def __init__(self):
        self.factors = []
        self.orig_backward = None

    def __enter__(self):
        self.factors = []
        orig = PerStepMetricCorrection.backward

        def patched(ctx, grad_output):
            p_old, p_new = ctx.saved_tensors
            c = ctx.c
            mn = ctx.max_norm
            from spikergnn.dvm import conformal_factor
            lam_old = conformal_factor(p_old, c, max_norm=mn)
            lam_new = conformal_factor(p_new, c, max_norm=mn)
            factor = (lam_old / lam_new.clamp(min=1e-5)) ** 2
            self.factors.append(factor.detach().float().cpu())
            return orig(ctx, grad_output)

        self.orig_backward = orig
        PerStepMetricCorrection.backward = staticmethod(patched)
        return self

    def __exit__(self, *a):
        PerStepMetricCorrection.backward = self.orig_backward


def one_training_epoch(model, sampler, opt, device, n_episodes=20):
    model.train()
    total = 0.0
    for _ in range(n_episodes):
        ep = sampler.sample_episode(mode="train")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
        logits = model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        loss = F.cross_entropy(logits, q_l)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(n_episodes, 1)


@torch.no_grad()
def evaluate(model, sampler, device, n_episodes=30):
    model.eval()
    correct = total = 0
    for _ in range(n_episodes):
        ep = sampler.sample_episode(mode="test")
        s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
        logits = model(s_b.x, s_b.edge_index, s_b.batch,
                       q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
        pred = logits.argmax(dim=-1)
        correct += (pred == q_l).sum().item()
        total += q_l.numel()
    model.train()
    return correct / max(total, 1)


def compare_at_snapshot(state, in_dim, T, M, batch, device, metric_clamp=None):
    """Same weights, same batch: MC on vs MC off."""
    s_b, s_l, q_b, q_l = batch

    def loss_of(m):
        return F.cross_entropy(
            m(s_b.x, s_b.edge_index, s_b.batch,
              q_b.x, q_b.edge_index, q_b.batch, s_l, 2), q_l)

    out = {}
    grads = {}
    for tag, mc in (("mc_on", True), ("mc_off", False)):
        m = create_spikergnn(make_cfg(T, M, mc, metric_clamp=metric_clamp),
                             in_dim=in_dim).to(device)
        m.load_state_dict(state, strict=False)
        m.train()
        if mc:
            with FactorRecorder() as rec:
                loss_of(m).backward()
            # 每一步的 factor 形状为 (N,) —— 逐节点。沿时间链累乘才是
            # 该节点梯度路径上的真实累积因子；把所有节点混在一起相乘
            # 会得到一个无意义的天文数字（第一版就是这样，已修正）。
            # factor 张量可能来自不同层/不同子图，大小未必一致（实测出现
            # 175 vs 28），故按 numel 分组后再沿链累乘，而不是强行 stack。
            if rec.factors:
                flat = [f.reshape(-1).float() for f in rec.factors]
                groups = {}
                for v in flat:
                    groups.setdefault(int(v.numel()), []).append(v)
                step_stats, chain_stats = [], []
                for n, vs in groups.items():
                    mat = torch.stack(vs, dim=0)              # (steps, n)
                    mat = torch.where(torch.isfinite(mat), mat,
                                      torch.ones_like(mat))
                    step_stats.append((float(mat.min()), float(mat.median()),
                                       float(mat.max())))
                    chain = torch.exp(torch.log(mat.clamp(min=1e-12)).sum(dim=0))
                    chain_stats.append(float(chain.median()))
                out["n_correction_steps"] = len(rec.factors)
                out["n_shape_groups"] = len(groups)
                out["group_sizes"] = sorted(groups.keys())
                out["per_step_factor_min"] = min(s[0] for s in step_stats)
                out["per_step_factor_median"] = float(
                    sum(s[1] for s in step_stats) / len(step_stats))
                out["per_step_factor_max"] = max(s[2] for s in step_stats)
                out["chain_factor_median_by_group"] = [
                    round(c, 6) for c in chain_stats]
                out["chain_factor_median"] = float(
                    sorted(chain_stats)[len(chain_stats) // 2])
                out["chain_factor_max"] = max(chain_stats)
            else:
                out["n_correction_steps"] = 0
        else:
            loss_of(m).backward()
        grads[tag] = named_grads(m)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    common = [n for n in grads["mc_on"]
              if n in grads["mc_off"]
              and grads["mc_on"][n].numel() == grads["mc_off"][n].numel()]
    va = torch.cat([grads["mc_on"][n] for n in common])
    vb = torch.cat([grads["mc_off"][n] for n in common])
    out["n_shared_params"] = len(common)
    # 绝对范数必须一并记录：比值本身会被"分母趋零"放大成天文数字，
    # 只有同时看到两侧量级才能判断是 MC 放大还是对照侧梯度消失。
    out["gnorm_mc_on"] = float(va.norm())
    out["gnorm_mc_off"] = float(vb.norm())
    out["gnorm_ratio"] = float(va.norm() / vb.norm().clamp(min=1e-12))
    out["cos"] = float(F.cosine_similarity(va.reshape(1, -1), vb.reshape(1, -1)))
    out["rel_err"] = float((va - vb).norm() / vb.norm().clamp(min=1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--seeds", default="0,7")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--metric-clamp", type=float, default=None,
                    help="trusted sub-ball radius r_max passed to conformal_factor; "
                         "caps lambda so the correction factor is bounded by (lambda_max/2)^2")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT,
                                                  "mc_gradient_scale.json"))
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    t0 = time.time()

    for seed in seeds:
        set_seed(seed)
        ds = TUGraphDataset(name=args.dataset, tu_name=args.dataset,
                            train_ratio=0.8, seed=seed)
        in_dim = max(ds.num_features, 1)
        sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5,
                                 seed=seed)
        model = create_spikergnn(make_cfg(args.T, args.M, True,
                                           metric_clamp=args.metric_clamp),
                                 in_dim=in_dim).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-5)

        # fixed probe batch (identical for every snapshot comparison)
        probe_ep = sampler.sample_episode(mode="train")
        batch = batch_episode_data(probe_ep, DEV)

        for ep in range(1, args.epochs + 1):
            loss = one_training_epoch(model, sampler, opt, DEV)
            sch.step()
            if ep % args.snap_every == 0 or ep == args.epochs:
                state = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
                acc = evaluate(model, sampler, DEV)
                cmp = compare_at_snapshot(state, in_dim, args.T, args.M,
                                          batch, DEV,
                                          metric_clamp=args.metric_clamp)
                row = {"dataset": args.dataset, "seed": seed, "epoch": ep,
                       "train_loss": loss, "val_acc": acc}
                row.update(cmp)
                rows.append(row)
                print(f"  seed{seed} ep{ep:3d} loss={loss:.4f} acc={acc:.4f} "
                      f"gnorm_ratio={cmp['gnorm_ratio']:.6f} "
                      f"cos={cmp['cos']:.8f} "
                      f"chain_factor_med={cmp.get('chain_factor_median')} "
                      f"chain_factor_max={cmp.get('chain_factor_max')}", flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    gates = check_gates(rows)
    out = {
        "config": {"dataset": args.dataset, "seeds": seeds,
                   "epochs": args.epochs, "T": args.T, "M": args.M,
                   "metric_clamp": args.metric_clamp, "device": DEV,
                   "probe_batch": "fixed episode sampled once per seed, "
                                  "reused for every snapshot"},
        "git_commit": git_commit(),
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
        "gates": gates,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nGATES:", json.dumps(gates, indent=2))
    print("->", args.out)


def check_gates(rows):
    if not rows:
        return {"verdict": "NO_DATA"}
    ratios = [r["gnorm_ratio"] for r in rows]
    cosv = [r["cos"] for r in rows]
    rel = [r["rel_err"] for r in rows]

    g3 = {"min_rel_err": min(rel),
          "switch_effective": min(rel) > 1e-12}
    in_band = all(0.99 <= r <= 1.01 for r in ratios)
    cos_ok = all(c > 0.9999 for c in cosv)
    verdict = ("A_negligible_uniform_scaling" if (in_band and cos_ok)
               else "B_materially_changes_gradient")
    return {
        "G3_switch_effective": g3,
        "G1_gnorm_ratio_min": min(ratios),
        "G1_gnorm_ratio_max": max(ratios),
        "G1_all_within_1pct": bool(in_band),
        "G1_min_cos": min(cosv),
        "G1_all_cos_gt_0.9999": bool(cos_ok),
        "verdict": verdict,
        "verdict_text": (
            "A: MC applies a near-uniform, near-unit scaling at every probed "
            "epoch; the 6.56pp training gap cannot be attributed to a "
            "materially different gradient and must be reported as long-horizon "
            "amplification, not as 'the correction degrades the gradient'."
            if verdict == "A_negligible_uniform_scaling" else
            "B: MC materially changes the gradient during training; report the "
            "epoch and magnitude and re-evaluate the causal story for the "
            "observed accuracy gap."),
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
