"""
正确测试ManifoldLIFNeuron - 使用正确的API
"""
import torch
import torch.nn as nn
import sys
import os
import math

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.manifolds import ProductManifold
from spikergnn.neurons import ManifoldLIFNeuron

def test_lif_correct_api():
    """使用正确API测试LIF神经元"""
    print("=" * 80)
    print("测试：ManifoldLIFNeuron（正确API）")
    print("=" * 80)

    # 创建流形
    d_h, d_e = 32, 32
    manifold = ProductManifold(d_h, d_e, c=1.0)

    # 创建LIF神经元
    lif = ManifoldLIFNeuron(
        manifold=manifold,
        beta_leak=0.3,
        gamma_input=0.5,
        alpha_decay=0.9,
        u_threshold=1.0,
        delta_reset=1.0,
        surrogate_beta=2.0,
        surrogate_type='atan',
        p_bar_decay=0.9,
        ablation='full'
    )

    lif.train()

    print(f"\nLIF神经元创建成功")
    print(f"  input_ln exists: {hasattr(lif, 'input_ln')}")
    print(f"  u_th_learnable: {lif.u_th_learnable.item():.4f}")
    print(f"  W weight norm: {lif.W.weight.data.norm().item():.4f}")

    # 准备输入
    N = 15  # 节点数
    T = 8   # 时间步
    d_total = d_h + d_e

    # x_seq: (N, T, d_total)
    x_seq = torch.randn(N, T, d_total) * 0.5

    # p_init: 初始流形状态
    p_hyp_init = torch.randn(N, d_h) * 0.1
    p_euc_init = torch.randn(N, d_e) * 0.1
    p_init = (p_hyp_init, p_euc_init)

    print(f"\n输入形状:")
    print(f"  x_seq: {x_seq.shape}")
    print(f"  p_init[0] (hyp): {p_init[0].shape}")
    print(f"  p_init[1] (euc): {p_init[1].shape}")

    # 测试前向传播
    print(f"\n前向传播...")

    total_spikes = 0
    lif.reset_membrane()

    # 方法1：直接调用forward（一次处理所有时间步）
    try:
        spikes, p_final = lif(x_seq, p_init)

        print(f"  ✅ forward()成功")
        print(f"  spikes shape: {spikes.shape}")
        print(f"  p_final[0] shape: {p_final[0].shape}")

        spike_count = spikes.sum().item()
        total_spikes += spike_count

        print(f"\n  总spikes: {spike_count}")
        print(f"  spike rate: {spike_count / (N * T * d_total):.4f}")

    except Exception as e:
        print(f"  ❌ forward()失败: {e}")
        import traceback
        traceback.print_exc()

    # 方法2：逐时间步调用（模拟STMP的调用方式）
    print(f"\n逐时间步调用LIF...")

    lif2 = ManifoldLIFNeuron(
        manifold=manifold,
        beta_leak=0.3,
        gamma_input=0.5,
        alpha_decay=0.9,
        u_threshold=1.0,
        surrogate_type='atan',
        ablation='full'
    )
    lif2.train()

    total_spikes_2 = 0
    lif2.reset_membrane()

    # 初始状态
    p_hyp_t = p_hyp_init.clone()
    p_euc_t = p_euc_init.clone()

    for t in range(T):
        # 输入（单个时间步）
        x_t = x_seq[:, t:t+1, :]  # (N, 1, d_total)

        # 调用forward
        spikes_t, (p_hyp_new, p_euc_new) = lif2(x_t, (p_hyp_t, p_euc_t))

        spike_count_t = spikes_t.sum().item()
        total_spikes_2 += spike_count_t

        # 更新状态
        p_hyp_t = p_hyp_new
        p_euc_t = p_euc_new

        # 打印状态
        if t == 0 or t == T-1:
            u_mag = lif2._u_mag.mean().item() if lif2._u_mag is not None else 0
            print(f"  t={t}: spikes={spike_count_t}, u_mag={u_mag:.4f}")

    print(f"\n逐时间步调用结果:")
    print(f"  总spikes: {total_spikes_2}")
    print(f"  spike rate: {total_spikes_2 / (N * T * d_total):.4f}")

    print(f"\n✅ 测试结论:")
    print(f"  方法1（批量）: {'✅ 有脉冲' if total_spikes > 0 else '❌ 无脉冲'}")
    print(f"  方法2（逐步）: {'✅ 有脉冲' if total_spikes_2 > 0 else '❌ 无脉冲'}")

    return total_spikes > 0 or total_spikes_2 > 0


def diagnose_why_no_spikes():
    """诊断为什么没有脉冲"""
    print("\n" + "=" * 80)
    print("诊断：为什么没有脉冲？")
    print("=" * 80)

    # 创建LIF
    d_h, d_e = 32, 32
    manifold = ProductManifold(d_h, d_e, c=1.0)
    lif = ManifoldLIFNeuron(manifold, surrogate_type='atan')
    lif.train()

    N, T = 15, 8
    d_total = d_h + d_e

    # 测试不同输入幅度
    print("\n测试不同输入幅度:")

    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x_seq = torch.randn(N, T, d_total) * scale
        p_init = (torch.randn(N, d_h) * 0.1, torch.randn(N, d_e) * 0.1)

        lif.reset_membrane()
        spikes, _ = lif(x_seq, p_init)

        spike_count = spikes.sum().item()
        spike_rate = spike_count / (N * T * d_total)

        print(f"  scale={scale}: spikes={spike_count}, rate={spike_rate:.4f}")

    # 检查阈值
    print(f"\n阈值检查:")
    print(f"  u_th_learnable: {lif.u_th_learnable.item():.4f}")
    print(f"  建议: 如果u_th太大，降低初始化值")

    # 检查LayerNorm效果
    print(f"\nLayerNorm效果:")
    I_t = torch.randn(N, d_total) * 0.5
    I_t_normed = lif.input_ln(I_t)
    w_input = lif.W(I_t_normed)
    w_input_mag = w_input.norm(dim=-1).mean().item()

    print(f"  Input mag before LN: {I_t.norm(dim=-1).mean().item():.4f}")
    print(f"  Input mag after LN: {I_t_normed.norm(dim=-1).mean().item():.4f}")
    print(f"  w_input mag: {w_input_mag:.4f}")
    print(f"  u_th: {lif.u_th_learnable.item():.4f}")
    print(f"  建议: w_input_mag应该接近u_th才能产生脉冲")


def fix_and_verify():
    """尝试修复并验证"""
    print("\n" + "=" * 80)
    print("修复方案验证")
    print("=" * 80)

    d_h, d_e = 32, 32
    manifold = ProductManifold(d_h, d_e, c=1.0)

    # 方案1：降低阈值
    print("\n方案1：降低u_th从1.0到0.3")
    lif1 = ManifoldLIFNeuron(manifold, u_threshold=0.3, surrogate_type='atan')
    lif1.train()

    N, T = 15, 8
    x_seq = torch.randn(N, T, 64) * 0.5
    p_init = (torch.randn(N, d_h) * 0.1, torch.randn(N, d_e) * 0.1)

    lif1.reset_membrane()
    spikes1, _ = lif1(x_seq, p_init)
    count1 = spikes1.sum().item()

    print(f"  spikes: {count1}")
    print(f"  结论: {'✅ 有效' if count1 > 0 else '❌ 仍无效'}")

    # 方案2：增大输入幅度
    print("\n方案2：增大输入幅度（scale=2.0）")
    lif2 = ManifoldLIFNeuron(manifold, surrogate_type='atan')
    lif2.train()

    x_seq_large = torch.randn(N, T, 64) * 2.0

    lif2.reset_membrane()
    spikes2, _ = lif2(x_seq_large, p_init)
    count2 = spikes2.sum().item()

    print(f"  spikes: {count2}")
    print(f"  结论: {'✅ 有效' if count2 > 0 else '❌ 仍无效'}")

    # 方案3：检查STMP中的实际输入
    print("\n方案3：诊断STMP中的实际输入幅度")
    print(f"  需要检查STMP.forward()中传给LIF的输入幅度")
    print(f"  可能问题：RGC的输出幅度太小")

    return count1 > 0 or count2 > 0


def main():
    print("🔬 ManifoldLIFNeuron 脉冲触发诊断")
    print("=" * 80)

    # 测试1：正确API
    result1 = test_lif_correct_api()

    # 诊断2：为什么没有脉冲
    diagnose_why_no_spikes()

    # 修复3：尝试修复
    result3 = fix_and_verify()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"测试1（正确API）: {'✅ 有脉冲' if result1 else '❌ 无脉冲'}")
    print(f"诊断2（幅度分析）: 见上方输出")
    print(f"修复3（方案验证）: {'✅ 有脉冲' if result3 else '❌ 无脉冲'}")

    if not result1:
        print(f"\n❌ 根本问题：即使使用正确API，LIF仍不触发脉冲")
        print(f"   可能原因：")
        print(f"   1. u_th_learnable太大（初始化=1.0）")
        print(f"   2. 输入幅度太小（即使*0.5后）")
        print(f"   3. LayerNorm后的幅度仍不匹配阈值")
        print(f"\n✅ 推荐修复：")
        print(f"   1. 降低u_th初始化值（从1.0降到0.3）")
        print(f"   2. 在LIF.forward()中添加调试输出")
        print(f"   3. 检查STMP中实际传给LIF的特征幅度")


if __name__ == "__main__":
    main()
