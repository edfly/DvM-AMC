"""
使用真实数据集测试 - 检查中间输出幅度
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler

def test_with_real_data():
    """使用真实数据测试"""
    print("=" * 80)
    print("测试：使用真实MUTAG数据")
    print("=" * 80)

    # 配置
    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        T=8,
        u_threshold=1.0,
        ablation="full",
    )

    # 模型
    print("\n创建模型...")
    model = SpikeRPGNN(config, in_dim=7)
    model.train()

    # 数据集
    print("\n加载MUTAG数据集...")
    dataset = TUGraphDataset(name="MUTAG", tu_name="MUTAG")

    # EpisodeSampler
    sampler = EpisodeSampler(dataset, n_way=2, k_shot=1, q_query=5, seed=42)

    # 采样episode
    print("\n采样episode...")
    episode = sampler.sample_episode()

    # 批处理
    from msg_baseline.data_utils import batch_episode_data
    support_batch, support_labels, query_batch, query_labels = batch_episode_data(
        episode, device="cpu"
    )

    print(f"  Support batch: {support_batch.x.shape[0]} nodes")
    print(f"  Query batch: {query_batch.x.shape[0]} nodes")
    print(f"  Support labels: {support_labels}")
    print(f"  Query labels: {query_labels.shape}")

    # 检查输入特征
    print(f"\n输入特征分析:")
    print(f"  support_batch.x norm: {support_batch.x.norm(dim=-1).mean().item():.4f}")
    print(f"  query_batch.x norm: {query_batch.x.norm(dim=-1).mean().item():.4f}")

    # 编码support set
    print(f"\n编码support set...")
    model.stmp.lif.reset_membrane()

    # Hook：检查MSPE输出
    mspe_outputs = []

    def hook_mspe_output(module, input, output):
        # output = (spike_seq, p_init)
        spike_seq, p_init = output
        mspe_outputs.append({
            'spike_seq_norm': spike_seq.norm().item(),
            'p_init_hyp_norm': p_init[0].norm().item(),
            'p_init_euc_norm': p_init[1].norm().item(),
        })
        return output

    mspe_handle = model.mspe.register_forward_hook(hook_mspe_output)

    # Hook：检查STMP输入
    stmp_inputs = []

    def hook_stmp_input(module, input):
        # input[0] = p_init, input[1] = spike_seq, input[2] = edge_index, input[3] = batch
        p_init, spike_seq = input[0], input[1]
        stmp_inputs.append({
            'p_init_hyp_norm': p_init[0].norm().item(),
            'p_init_euc_norm': p_init[1].norm().item(),
            'spike_seq_norm': spike_seq.norm().item(),
        })
        return input

    stmp_handle = model.stmp.register_forward_hook(hook_stmp_input)

    # 前向传播
    support_z = model.encode(
        support_batch.x, support_batch.edge_index, support_batch.batch
    )

    mspe_handle.remove()
    stmp_handle.remove()

    # 获取spike计数
    support_spikes = model.count_spikes()

    print(f"\nSupport编码结果:")
    print(f"  z_hyp shape: {support_z[0].shape}")
    print(f"  z_euc shape: {support_z[1].shape}")
    print(f"  Support spikes: {support_spikes}")

    # 分析中间输出
    if mspe_outputs:
        mspe_info = mspe_outputs[0]
        print(f"\nMSPE输出分析:")
        print(f"  spike_seq norm: {mspe_info['spike_seq_norm']:.4f}")
        print(f"  p_init hyp norm: {mspe_info['p_init_hyp_norm']:.4f}")
        print(f"  p_init euc norm: {mspe_info['p_init_euc_norm']:.4f}")

    if stmp_inputs:
        stmp_info = stmp_inputs[0]
        print(f"\nSTMP输入分析:")
        print(f"  p_init hyp norm: {stmp_info['p_init_hyp_norm']:.4f}")
        print(f"  p_init euc norm: {stmp_info['p_init_euc_norm']:.4f}")
        print(f"  spike_seq norm: {stmp_info['spike_seq_norm']:.4f}")

    # 编码query set
    print(f"\n编码query set...")
    model.stmp.lif.reset_membrane()
    query_z = model.encode(
        query_batch.x, query_batch.edge_index, query_batch.batch
    )
    query_spikes = model.count_spikes()

    print(f"  Query spikes: {query_spikes}")

    # 总spikes
    total_spikes = support_spikes + query_spikes
    print(f"\n总spikes: {total_spikes}")

    # 结论
    print(f"\n{'='*60}")
    print(f"结论:")
    print(f"{'='*60}")
    if total_spikes == 0:
        print(f"  ❌ spike_counts=0 问题复现！")
        print(f"  需要检查：")
        print(f"    1. MSPE输出幅度是否太小")
        print(f"    2. RGC输出幅度是否太小")
        print(f"    3. LIF输入幅度是否太小")
    else:
        print(f"  ✅ 脉冲正常产生！")
        print(f"  spike_count: {total_spikes}")
        print(f"  问题可能在：")
        print(f"    - 训练循环中的其他问题")
        print(f"    - 特定数据集（ENZYMES）的问题")


def compare_random_vs_real():
    """对比随机数据和真实数据"""
    print("\n" + "=" * 80)
    print("对比：随机数据 vs 真实数据")
    print("=" * 80)

    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        T=8,
        u_threshold=1.0,
        ablation="full",
    )

    # 测试1：随机数据
    print("\n测试1：随机数据...")
    model1 = SpikeRPGNN(config, in_dim=7)
    model1.train()

    # 随机输入
    num_nodes = 15
    x_random = torch.randn(num_nodes, 7)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model1.stmp.lif.reset_membrane()
    z_random = model1.encode(x_random, edge_index, batch)
    spikes_random = model1.count_spikes()

    print(f"  Spikes: {spikes_random}")

    # 测试2：真实数据
    print("\n测试2：真实数据...")
    model2 = SpikeRPGNN(config, in_dim=7)
    model2.train()

    # 加载真实数据
    dataset = TUGraphDataset(name="MUTAG", tu_name="MUTAG")
    sampler = EpisodeSampler(dataset, n_way=2, k_shot=1, q_query=5, seed=42)
    support_data, query_data = sampler.sample_episode()

    from msg_baseline.data_utils import batch_episode_data
    support_batch, _, query_batch, _ = batch_episode_data(
        (support_data, query_data), device="cpu"
    )

    model2.stmp.lif.reset_membrane()
    z_real = model2.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
    spikes_real = model2.count_spikes()

    print(f"  Spikes: {spikes_real}")

    # 对比
    print(f"\n对比:")
    print(f"  随机数据 spikes: {spikes_random}")
    print(f"  真实数据 spikes: {spikes_real}")

    if spikes_random > 0 and spikes_real == 0:
        print(f"\n  ❌ 问题确认：真实数据导致无脉冲")
        print(f"  根本原因：真实数据的特征幅度或分布导致")
        print(f"  修复方案：")
        print(f"    1. 在MSPE中添加归一化")
        print(f"    2. 增大LIF输入幅度")
        print(f"    3. 降低u_th阈值")


def main():
    print("🔬 SpikeRPGNN 真实数据测试")
    print("=" * 80)

    # 测试1：真实数据
    test_with_real_data()

    # 对比2：随机vs真实
    compare_random_vs_real()

    print("\n" + "=" * 80)
    print("下一步")
    print("=" * 80)
    print("如果真实数据导致spike_counts=0：")
    print("  1. 在MSPE.forward()中添加调试输出")
    print("  2. 打印MSPE输出的幅度")
    print("  3. 打印RGC输出的幅度")
    print("  4. 根据输出调整归一化或阈值")


if __name__ == "__main__":
    main()
