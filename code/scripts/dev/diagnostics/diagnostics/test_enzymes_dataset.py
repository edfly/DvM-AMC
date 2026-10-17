"""
ENZYMES数据集测试 - 检查是否产生脉冲
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
from spikergnn.train import train_one_epoch, evaluate

def test_enzymes_dataset():
    """测试ENZYMES数据集"""
    print("=" * 80)
    print("测试：ENZYMES数据集")
    print("=" * 80)

    # 配置（匹配实际实验）
    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        T=8,
        u_threshold=1.0,
        ablation="full",
        epochs=2,
        train_episodes=30,
        episodes_per_epoch=15,
        test_episodes=10,
        lr=1e-3,
    )

    # 数据集
    print("\n加载ENZYMES数据集...")
    try:
        dataset = TUGraphDataset(name="ENZYMES", tu_name="ENZYMES")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print(f"  可能原因：数据集文件不存在")
        return

    # 模型
    print(f"\n创建模型...")
    # 检查ENZYMES的特征维度
    sample = dataset[0]
    in_dim = sample.x.shape[1]
    print(f"  输入特征维度: {in_dim}")

    model = SpikeRPGNN(config, in_dim=in_dim)
    model.train()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # EpisodeSampler
    sampler = EpisodeSampler(
        dataset, n_way=6, k_shot=1, q_query=5, seed=42
    )

    # 训练循环
    print(f"\n开始训练（2 epochs）...")

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        # 训练一个epoch
        train_result = train_one_epoch(model, sampler, optimizer, config)

        print(f"  Train accuracy: {train_result['mean_acc']:.4f}")
        print(f"  Train loss: {train_result['mean_loss']:.4f}")
        print(f"  Train spike_count: {train_result['spike_count']}")

    # 评估
    print(f"\n评估...")
    eval_result = evaluate(model, sampler, config)

    print(f"\n评估结果:")
    print(f"  Test accuracy: {eval_result['mean_acc']:.4f}")
    print(f"  Test spike_count: {eval_result['spike_count']}")

    # 结论
    total_spikes = train_result['spike_count'] + eval_result['spike_count']
    print(f"\n{'='*60}")
    print(f"结论:")
    print(f"{'='*60}")
    if total_spikes == 0:
        print(f"  ❌ ENZYMES数据集导致spike_counts=0！")
        print(f"\n  根本原因：")
        print(f"    1. ENZYMES特征幅度可能太小")
        print(f"    2. ENZYMES特征分布导致LIF不触发")
        print(f"    3. 6-way分类的梯度动态不同")
        print(f"\n  修复方案：")
        print(f"    1. 在MSPE中添加特征归一化")
        print(f"    2. 降低u_threshold（从1.0到0.3）")
        print(f"    3. 增大LIF输入幅度")
    else:
        print(f"  ✅ ENZYMES数据集也产生脉冲！")
        print(f"  Total spikes: {total_spikes}")
        print(f"\n  问题可能在：")
        print(f"    - 实际实验的代码版本旧")
        print(f"    - 实际实验的超参数不同")


def compare_datasets():
    """对比MUTAG和ENZYMES"""
    print("\n" + "=" * 80)
    print("对比：MUTAG vs ENZYMES")
    print("=" * 80)

    datasets = ["MUTAG", "ENZYMES"]

    for name in datasets:
        print(f"\n{name}数据集:")
        try:
            dataset = TUGraphDataset(name=name, tu_name=name)

            # 检查第一个样本
            sample = dataset[0]
            print(f"  特征维度: {sample.x.shape[1]}")
            print(f"  特征幅度: {sample.x.norm(dim=-1).mean().item():.4f}")

            # 检查类别数
            from msg_baseline.config import DATASET_CONFIGS
            if name in DATASET_CONFIGS:
                n_way = DATASET_CONFIGS[name]["n_way"]
                print(f"  类别数: {n_way}")

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")


def test_lower_threshold_for_enzymes():
    """测试：降低u_threshold for ENZYMES"""
    print("\n" + "=" * 80)
    print("测试：降低u_threshold（ENZYMES）")
    print("=" * 80)

    try:
        dataset = TUGraphDataset(name="ENZYMES", tu_name="ENZYMES")
        sample = dataset[0]
        in_dim = sample.x.shape[1]
    except:
        print("  ❌ ENZYMES数据集加载失败")
        return

    # 测试不同阈值
    for u_th in [0.1, 0.3, 0.5, 1.0]:
        print(f"\nu_threshold={u_th}:")

        config = SpikeRPGNNConfig(
            d_h=16, d_e=16, hidden_dim=32,
            T=8,
            u_threshold=u_th,
            ablation="full",
        )

        model = SpikeRPGNN(config, in_dim=in_dim)
        model.train()

        # 创建虚拟输入
        num_nodes = 10
        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # 编码
        model.stmp.lif.reset_membrane()
        _ = model.encode(x, edge_index, batch)
        spikes = model.count_spikes()

        print(f"  spikes: {spikes}")


def main():
    print("🔬 SpikeRPGNN ENZYMES数据集测试")
    print("=" * 80)

    # 对比1：数据集特征
    compare_datasets()

    # 测试2：ENZYMES训练
    test_enzymes_dataset()

    # 测试3：降低阈值
    test_lower_threshold_for_enzymes()

    print("\n" + "=" * 80)
    print("最终建议")
    print("=" * 80)
    print("1. 如果ENZYMES也产生脉冲：")
    print("   → 重新运行完整实验（确保使用最新代码）")
    print("2. 如果ENZYMES无脉冲：")
    print("   → 降低u_threshold到0.3或0.1")
    print("   → 在MSPE中添加特征归一化")
    print("   → 检查ENZYMES特征幅度并调整")


if __name__ == "__main__":
    main()
