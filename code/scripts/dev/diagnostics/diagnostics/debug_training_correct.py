"""
正确的训练调试脚本 - 使用正确的API
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

def debug_training_with_correct_api():
    """使用正确API调试训练"""
    print("=" * 80)
    print("调试：使用正确API模拟训练")
    print("=" * 80)

    # 配置
    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        T=8,
        u_threshold=1.0,
        ablation="full",
        epochs=2,
        train_episodes=50,
        test_episodes=10,
    )

    # 数据集
    print("\n加载MUTAG数据集...")
    dataset = TUGraphDataset(name="MUTAG", tu_name="MUTAG")
    print(f"  数据集大小: {len(dataset)}")

    # 模型
    print(f"\n创建模型...")
    model = SpikeRPGNN(config, num_classes=2)
    model.train()

    # 检查LIF初始状态
    lif = model.stmp.lif
    print(f"\nLIF初始状态:")
    print(f"  u_th_learnable: {lif.u_th_learnable.item():.4f}")
    print(f"  input_ln: {hasattr(lif, 'input_ln')}")
    print(f"  W weight norm: {lif.W.weight.data.norm().item():.4f}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # EpisodeSampler
    sampler = EpisodeSampler(
        dataset, n_way=2, k_shot=1, q_query=5, seed=42
    )

    # 训练循环（带调试输出）
    print(f"\n开始训练（仅5个episode用于调试）...")

    for epoch in range(2):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")

        model.train()
        total_spikes_epoch = 0

        for episode_idx in range(5):
            # 采样episode
            support_data, query_data = sampler.sample_episode()

            # 批处理
            support_batch = batch_episode_data(support_data)
            query_batch = batch_episode_data(query_data)

            # 编码support set并计数spikes
            model.stmp.lif.reset_membrane()
            support_z = model.encode(support_batch)
            support_spikes = model.count_spikes(support_batch)

            # 编码query set并计数spikes
            model.stmp.lif.reset_membrane()
            query_z = model.encode(query_batch)
            query_spikes = model.count_spikes(query_batch)

            total_spikes = support_spikes + query_spikes
            total_spikes_epoch += total_spikes

            # 输出LIF状态
            print(f"  Episode {episode_idx + 1}:")
            print(f"    support_spikes: {support_spikes}")
            print(f"    query_spikes: {query_spikes}")
            print(f"    u_th: {lif.u_th_learnable.item():.4f}")

            # 检查膜电位
            if lif._u_mag is not None:
                print(f"    u_mag mean: {lif._u_mag.mean().item():.4f}")

            # 前向传播
            logits = model.classify(support_z, query_z)

            # 计算损失
            labels = torch.cat([
                torch.arange(2) for _ in range(len(query_data))
            ]).view(2, len(query_data)).t().flatten()

            if logits.device != torch.device('cpu'):
                labels = labels.to(logits.device)

            loss = nn.CrossEntropyLoss()(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"    loss: {loss.item():.4f}")

            # 检查梯度
            if lif.u_th_learnable.grad is not None:
                print(f"    u_th.grad: {lif.u_th_learnable.grad.item():.6f}")

        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"  总spikes: {total_spikes_epoch}")
        print(f"  u_th: {lif.u_th_learnable.item():.4f}")

    print(f"\n✅ 调试完成")
    print(f"  最终u_th: {lif.u_th_learnable.item():.4f}")

    # 分析
    print(f"\n分析:")
    if total_spikes_epoch == 0:
        print(f"  ❌ 问题确认：spike_counts=0")
        print(f"  可能原因：")
        print(f"    1. LIF输入幅度太小")
        print(f"    2. u_th学习率太高，增长太快")
        print(f"    3. RGC输出幅度在训练中衰减")
    else:
        print(f"  ✅ 脉冲正常产生")
        print(f"  问题可能在其他部分")


def check_lif_input_magnitude():
    """检查LIF的实际输入幅度"""
    print("\n" + "=" * 80)
    print("检查：LIF实际输入幅度")
    print("=" * 80)

    config = SpikeRPGNNConfig(
        d_h=32, d_e=32, hidden_dim=64,
        ablation="full"
    )

    model = SpikeRPGNN(config, num_classes=2)
    model.eval()

    # 创建虚拟数据
    from torch_geometric.data import Data

    dummy_data = Data(
        x=torch.randn(15, 1),  # 15个节点
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        batch=torch.zeros(15, dtype=torch.long)
    )

    # 编码
    with torch.no_grad():
        # 获取MSPE输出
        p_init = model.mspe(dummy_data)

        # 手动运行STMP并检查LIF输入
        stmp = model.stmp
        lif = stmp.lif

        p_hyp, p_euc = p_init
        N = p_hyp.shape[0]

        # 运行一个时间步
        for rgc_layer in stmp.rgc_layers:
            (p_hyp, p_euc) = rgc_layer((p_hyp, p_euc), dummy_data.edge_index)

        # RGC输出
        rgc_features = stmp.manifold.merge(p_hyp, p_euc)  # (N, d_total)
        print(f"\nRGC输出幅度:")
        print(f"  rgc_features norm: {rgc_features.norm(dim=-1).mean().item():.4f}")

        # LIF输入（_after LayerNorm）
        rgc_features_step = rgc_features.unsqueeze(1)  # (N, 1, d_total)

        # 手动调用LIF.forward()并检查的档输入
        lif.reset_membrane()

        # Hook to capture input to LIF
        lif_inputs = []

        def hook(module, input):
            lif_inputs.append(input[0].clone())
            return input

        handle = lif.register_forward_hook(hook)

        spikes, (p_hyp_new, p_euc_new) = lif(rgc_features_step, (p_hyp, p_euc))

        handle.remove()

        if lif_inputs:
            actual_input = lif_inputs[0]
            print(f"\nLIF实际输入:")
            print(f"  input shape: {actual_input.shape}")
            print(f"  input norm: {actual_input.norm(dim=-1).mean().item():.4f}")

        # 检查LayerNorm输出
        with torch.no_grad():
            I_t = rgc_features  # (N, d_total)
            I_t_normed = lif.input_ln(I_t)
            w_input = lif.W(I_t_normed)
            w_input_mag = w_input.norm(dim=-1).mean().item()

            print(f"\nLayerNorm效果:")
            print(f"  I_t norm: {I_t.norm(dim=-1).mean().item():.4f}")
            print(f"  I_t_normed norm: {I_t_normed.norm(dim=-1).mean().item():.4f}")
            print(f"  w_input mag: {w_input_mag:.4f}")
            print(f"  u_th: {lif.u_th_learnable.item():.4f}")

            if w_input_mag < lif.u_th_learnable.item():
                print(f"  ⚠️  输入幅度小于阈值，可能无脉冲")
            else:
                print(f"  ✅ 输入幅度大于阈值，应该有脉冲")


def main():
    print("🔬 SpikeRPGNN 训练调试（正确API）")
    print("=" * 80)

    # 调试1：模拟训练
    debug_training_with_correct_api()

    # 检查2：LIF输入幅度
    check_lif_input_magnitude()

    print("\n" + "=" * 80)
    print("总结与建议")
    print("=" * 80)
    print("1. 如果spike_counts仍为0：")
    print("   - 降低u_th初始化值（从1.0到0.3）")
    print("   - 降低u_th的学习率（从1e-3到1e-5）")
    print("   - 在LIF.forward()中添加调试输出")
    print("2. 如果spike_counts>0但准确率仍低：")
    print("   - 检查模型是否真正在学习")
    print("   - 调整其他超参数")


if __name__ == "__main__":
    main()
