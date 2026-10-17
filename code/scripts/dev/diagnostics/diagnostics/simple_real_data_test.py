"""
简化测试 - 检查真实数据是否产生脉冲
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data

def simple_test():
    """简化测试：真实数据编码并检查spikes"""
    print("=" * 80)
    print("简化测试：真实数据")
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
    support_batch, support_labels, query_batch, query_labels = batch_episode_data(
        episode, device="cpu"
    )

    print(f"  Support: {support_batch.x.shape[0]} nodes")
    print(f"  Query: {query_batch.x.shape[0]} nodes")

    # 编码support
    print("\n编码support...")
    model.stmp.lif.reset_membrane()
    support_z = model.encode(
        support_batch.x, support_batch.edge_index, support_batch.batch
    )
    support_spikes = model.count_spikes()

    print(f"  Support spikes: {support_spikes}")

    # 编码query
    print("\n编码query...")
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
        print(f"\n  下一步：在LIF.forward()中添加调试输出")
        print(f"  1. 打印w_input_mag每个时间步")
        print(f"  2. 打印u_mag每个时间步")
        print(f"  3. 打印spikes每个时间步")
        print(f"  4. 根据输出判断为什么没有脉冲")
    else:
        print(f"  ✅ 脉冲正常产生！")
        print(f"  spike_count: {total_spikes}")
        print(f"\n  问题可能在：")
        print(f"    - 训练循环中的其他问题")
        print(f"    - 特定数据集（ENZYMES）的问题")
        print(f"    - 超参数配置问题")


def test_different_u_th():
    """测试不同u_threshold"""
    print("\n" + "=" * 80)
    print("测试：不同u_threshold")
    print("=" * 80)

    # 数据集
    dataset = TUGraphDataset(name="MUTAG", tu_name="MUTAG")
    sampler = EpisodeSampler(dataset, n_way=2, k_shot=1, q_query=5, seed=42)
    episode = sampler.sample_episode()
    support_batch, _, query_batch, _ = batch_episode_data(episode, device="cpu")

    # 测试不同阈值
    for u_th in [0.1, 0.3, 0.5, 1.0, 2.0]:
        config = SpikeRPGNNConfig(
            d_h=16, d_e=16, hidden_dim=32,
            T=8,
            u_threshold=u_th,
            ablation="full",
        )

        model = SpikeRPGNN(config, in_dim=7)
        model.train()

        # 编码
        model.stmp.lif.reset_membrane()
        _ = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
        spikes = model.count_spikes()

        print(f"  u_th={u_th}: spikes={spikes}")


def main():
    print("🔬 SpikeRPGNN 简化真实数据测试")
    print("=" * 80)

    # 测试1：简化测试
    simple_test()

    # 测试2：不同阈值
    test_different_u_th()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("如果spike_counts仍为0：")
    print("  1. 问题在MSPE或STMP的某个部分")
    print("  2. 需要在LIF.forward()中添加详细调试输出")
    print("  3. 添加print语句到neurons.py的forward()方法")


if __name__ == "__main__":
    main()
