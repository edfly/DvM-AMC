"""
10 epochs验证脚本 - 观察损失下降和准确率提升
验证Bug修复的长期效果
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch, evaluate, EpisodeSampler, TUGraphDataset, set_seed

def verify_10_epochs():
    """运行10个epoch验证损失下降"""
    
    print("=" * 70)
    print("10 Epochs验证：观察损失下降和准确率提升")
    print("=" * 70)
    
    # 设置随机种子
    set_seed(42)
    
    # 配置
    config = SpikeRPGNNConfig(
        hidden_dim=64,     # d_h + d_e = 32 + 32 = 64
        d_h=32,
        d_e=32,
        T=8,               # 时间步
        u_threshold=1.0,   # 阈值
        use_dvm=True,
        lambda_sr=0.01,
        ablation='full',
    )
    
    print(f"\n配置参数:")
    print(f"  d_h={config.d_h}, d_e={config.d_e}")
    if hasattr(config, 'timesteps'):
        print(f"  timesteps={config.timesteps}")
    elif hasattr(config, 'time_steps'):
        print(f"  time_steps={config.time_steps}")
    print(f"  ablation={config.ablation}")
    
    # 创建模型
    print(f"\n创建模型...")
    # SpikeRPGNN接受 config 和 in_dim（输入特征维度）
    # ENZYMES特征维度=3
    model = SpikeRPGNN(config, in_dim=3)
    model.train()
    
    # 设备
    device = torch.device('cpu')
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    
    # 加载ENZYMES数据集
    print(f"\n加载ENZYMES数据集...")
    dataset = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES')
    # TUGraphDataset不支持len()，但从打印信息可知大小
    # [ENZYMES] Loaded 480 train / 120 test graphs | 6 classes | 3 features
    print(f"  数据集: ENZYMES (480 train / 120 test)")
    print(f"  类别数: 6")
    print(f"  特征维度: 3")
    
    # 创建采样器
    sampler = EpisodeSampler(
        dataset,
        n_way=6,           # ENZYMES是6分类
        k_shot=1,           # 1-shot
        q_query=5,
        seed=42,
    )
    print(f"  EpisodeSampler创建成功")
    
    # 训练循环（10 epochs）
    print(f"\n开始训练（10个epoch）...")
    print(f"-" * 70)
    
    model.train()
    train_losses = []
    train_accs = []
    train_spikes = []
    
    start_time = time.time()
    
    for epoch in range(10):
        epoch_start = time.time()
        
        # 训练一个epoch
        results = train_one_epoch(model, sampler, optimizer, config, device)
        
        # 注意：正确的键名是 mean_loss, mean_acc, spike_count
        loss = results.get('mean_loss', 0.0)
        accuracy = results.get('mean_acc', 0.0)
        spikes = int(results.get('spike_count', 0))
        
        train_losses.append(loss)
        train_accs.append(accuracy)
        train_spikes.append(spikes)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/10: "
              f"Loss={loss:.4f}, "
              f"Acc={accuracy:.4f}, "
              f"Spikes={spikes:6d}, "
              f"Time={epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n" + "-" * 70)
    print(f"训练完成！")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均耗时/epoch: {total_time/10:.2f}s")
    
    # 评估
    print(f"\n开始评估...")
    model.eval()
    
    with torch.no_grad():
        eval_results = evaluate(model, sampler, config, device)
        
        # 注意：evaluate返回 mean_acc 和 spike_count，没有loss
        test_accuracy = eval_results.get('mean_acc', 0.0)
        test_spikes = int(eval_results.get('spike_count', 0))
    
    print(f"  测试准确率: {test_accuracy:.4f}")
    print(f"  测试脉冲数: {test_spikes}")
    
    # 分析结果
    print(f"\n" + "=" * 70)
    print(f"分析结果:")
    print(f"=" * 70)
    
    # 检查1: 损失下降
    loss_init = train_losses[0]
    loss_final = train_losses[-1]
    loss_diff = loss_init - loss_final
    loss_percent = (loss_diff / loss_init) * 100 if loss_init > 0 else 0
    
    if loss_diff > 0:
        print(f"✅ [通过] 损失下降")
        print(f"   初始损失: {loss_init:.4f}")
        print(f"   最终损失: {loss_final:.4f}")
        print(f"   下降幅度: {loss_diff:.4f} ({loss_percent:.1f}%)")
    else:
        print(f"❌ [失败] 损失未下降")
        print(f"   初始损失: {loss_init:.4f}")
        print(f"   最终损失: {loss_final:.4f}")
        print(f"   变化幅度: {loss_diff:.4f} ({loss_percent:.1f}%)")
    
    # 检查2: 准确率提升
    acc_init = train_accs[0]
    acc_final = train_accs[-1]
    acc_diff = acc_final - acc_init
    
    if acc_diff > 0:
        print(f"✅ [通过] 准确率提升")
        print(f"   初始准确率: {acc_init:.4f}")
        print(f"   最终准确率: {acc_final:.4f}")
        print(f"   提升幅度: {acc_diff:.4f}")
    else:
        print(f"⚠️  [警告] 准确率未提升")
        print(f"   初始准确率: {acc_init:.4f}")
        print(f"   最终准确率: {acc_final:.4f}")
        print(f"   变化幅度: {acc_diff:.4f}")
    
    # 检查3: 脉冲数稳定
    spikes_mean = np.mean(train_spikes)
    spikes_std = np.std(train_spikes)
    spikes_cv = (spikes_std / spikes_mean) * 100 if spikes_mean > 0 else 0
    
    if spikes_mean > 0:
        print(f"✅ [通过] 脉冲数 > 0")
        print(f"   平均脉冲数: {spikes_mean:.0f}")
        print(f"   标准差: {spikes_std:.0f}")
        print(f"   变异系数: {spikes_cv:.1f}%")
    else:
        print(f"❌ [失败] 脉冲数 = 0")
    
    # 检查4: 测试准确率 > 随机基线
    random_baseline = 1.0 / 6.0
    
    if test_accuracy > random_baseline:
        print(f"✅ [通过] 测试准确率 > 随机基线")
        print(f"   测试准确率: {test_accuracy:.4f}")
        print(f"   随机基线: {random_baseline:.4f}")
    else:
        print(f"⚠️  [警告] 测试准确率 = 随机基线")
        print(f"   测试准确率: {test_accuracy:.4f}")
        print(f"   随机基线: {random_baseline:.4f}")
    
    # 绘制训练曲线
    print(f"\n生成训练曲线...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(range(1, 11), train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(range(1, 11), train_accs, 'g-', linewidth=2)
    axes[1].axhline(y=random_baseline, color='r', linestyle='--', label='Random')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 脉冲数曲线
    axes[2].plot(range(1, 11), train_spikes, 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Spikes')
    axes[2].set_title('Training Spikes')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spikergnn_results/verification_10_epochs.png', dpi=150)
    print(f"  训练曲线已保存: spikergnn_results/verification_10_epochs.png")
    
    # 决定是否可以启动完整实验
    print(f"\n" + "=" * 70)
    print(f"验证结论:")
    print(f"=" * 70)
    
    can_proceed = True
    
    if loss_diff <= 0:
        print(f"❌ 损失未下降，建议检查训练配置")
        can_proceed = False
    
    if spikes_mean == 0:
        print(f"❌ 脉冲数=0，Bug 7修复未生效")
        can_proceed = False
    
    if can_proceed:
        print(f"✅ 验证通过！可以启动完整实验。")
        print(f"   建议监控指标:")
        print(f"     - spike_counts > 0")
        print(f"     - 损失持续下降")
        print(f"     - 准确率 > 随机基线")
        return True
    else:
        print(f"❌ 验证失败！需要进一步诊断。")
        return False

if __name__ == "__main__":
    success = verify_10_epochs()
    
    if success:
        print(f"\n" + "=" * 70)
        print(f"✅ 验证通过！建议启动完整实验。")
        print(f"=" * 70)
    else:
        print(f"\n" + "=" * 70)
        print(f"❌ 验证失败！请不要启动完整实验。")
        print(f"=" * 70)
