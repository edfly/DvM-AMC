"""
使用正确API的SpikeRPGNN测试
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN

def test_with_correct_api():
    """使用正确API测试SpikeRPGNN"""
    print("=" * 80)
    print("测试：使用正确API")
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

    # 检查LIF
    lif = model.stmp.lif
    print(f"\nLIF状态:")
    print(f"  u_th_learnable: {lif.u_th_learnable.item():.4f}")
    print(f"  input_ln exists: {hasattr(lif, 'input_ln')}")

    # 创建图数据（分离的张量）
    print(f"\n创建图数据...")

    num_nodes = 15

    # 支持集
    support_x = torch.randn(num_nodes, 7)  # 节点特征
    support_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    support_batch = torch.zeros(num_nodes, dtype=torch.long)

    # 查询集
    query_x = torch.randn(num_nodes + 3, 7)  # 不同大小的查询集
    query_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    query_batch = torch.zeros(num_nodes + 3, dtype=torch.long)

    print(f"  Support: {num_nodes} nodes")
    print(f"  Query: {num_nodes + 3} nodes")

    # 编码支持集
    print(f"\n编码支持集...")
    model.stmp.lif.reset_membrane()
    support_z = model.encode(support_x, support_edge_index, support_batch)

    # 获取spike计数
    support_spikes = model.count_spikes()

    print(f"  Support z shapes:")
    print(f"    hyp: {support_z[0].shape}")
    print(f"    euc: {support_z[1].shape}")
    print(f"  Support spikes: {support_spikes}")

    # 编码查询集
    print(f"\n编码查询集...")
    model.stmp.lif.reset_membrane()
    query_z = model.encode(query_x, query_edge_index, query_batch)

    query_spikes = model.count_spikes()

    print(f"  Query spikes: {query_spikes}")

    # 总spikes
    total_spikes = support_spikes + query_spikes
    print(f"\n总spikes: {total_spikes}")

    # 分类
    print(f"\n分类...")
    support_labels = torch.tensor([0, 1])  # 2-way
    n_way = 2

    # 需要扩展support_z到每个query样本
    # 简化：假设每个类有1个原型
    logits = model.classify(query_z, support_z, support_labels, n_way)
    print(f"  Logits shape: {logits.shape}")

    # 检查LIF状态
    print(f"\nLIF最终状态:")
    print(f"  u_th_learnable: {lif.u_th_learnable.item():.4f}")
    if lif._u_mag is not None:
        print(f"  _u_mag mean: {lif._u_mag.mean().item():.4f}")

    # 结论
    print(f"\n{'='*60}")
    print(f"结论:")
    print(f"{'='*60}")
    if total_spikes == 0:
        print(f"  ❌ spike_counts=0 问题复现！")
        print(f"  根本原因需要深入调查：")
        print(f"    1. 检查LIF.forward()中的输入幅度")
        print(f"    2. 检查u_th_learnable是否太大")
        print(f"    3. 在LIF.forward()中添加调试输出")
    else:
        print(f"  ✅ 脉冲正常产生！")
        print(f"  spike_count: {total_spikes}")
        print(f"  问题可能在训练循环的其他部分")


def check_lif_input_in_training():
    """在训练场景中检查LIF的输入幅度"""
    print("\n" + "=" * 80)
    print("检查：训练场景中的LIF输入幅度")
    print("=" * 80)

    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        T=8,
        u_threshold=1.0,
        ablation="full",
    )

    model = SpikeRPGNN(config, in_dim=7)
    model.train()

    lif = model.stmp.lif

    # 创建输入
    num_nodes = 10
    x = torch.randn(num_nodes, 7)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # 编码（会调用STMP → LIF）
    model.stmp.lif.reset_membrane()

    # Hook：捕获LIF的输入
    lif_inputs = []

    def hook(module, input):
        # input[0] = x_seq, input[1] = p_init
        lif_inputs.append({
            'x_seq_shape': input[0].shape,
            'x_seq_norm': input[0].norm(dim=-1).mean().item()
        })
        return input

    handle = lif.register_forward_hook(hook)

    # 前向传播
    z = model.encode(x, edge_index, batch)

    handle.remove()

    # 分析LIF输入
    if lif_inputs:
        info = lif_inputs[0]
        print(f"\nLIF输入分析:")
        print(f"  x_seq shape: {info['x_seq_shape']}")
        print(f"  x_seq norm: {info['x_seq_norm']:.4f}")

        # 模拟LIF内部的LayerNorm
        with torch.no_grad():
            # 获取STMP传递给LIF的实际输入
            # 这需要修改STMP代码或添加更多hooks

            print(f"\n建议：")
            print(f"  1. 在STMP.forward()中添加调试输出")
            print(f"  2. 打印rgc_features的幅度")
            print(f"  3. 打印LIF.input_ln的输出幅度")


def main():
    print("🔬 SpikeRPGNN 正确API测试")
    print("=" * 80)

    # 测试1：正确API
    test_with_correct_api()

    # 检查2：LIF输入幅度
    check_lif_input_in_training()

    print("\n" + "=" * 80)
    print("下一步")
    print("=" * 80)
    print("如果spike_counts仍为0：")
    print("  1. 在ManifoldLIFNeuron.forward()中添加调试输出")
    print("  2. 打印以下变量的每个时间步：")
    print("     - I_t.norm()")
    print("     - I_t_normed.norm()")
    print("     - w_input_mag")
    print("     - u_mag")
    print("     - spikes.sum()")
    print("  3. 根据输出判断为什么没有脉冲")


if __name__ == "__main__":
    main()
