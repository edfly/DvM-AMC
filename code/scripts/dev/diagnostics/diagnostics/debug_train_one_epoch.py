"""
调试脚本 - 检查train_one_epoch的返回值
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch, evaluate, EpisodeSampler, TUGraphDataset, set_seed

def debug_train_one_epoch():
    """调试train_one_epoch返回值"""
    
    print("=" * 60)
    print("调试：检查train_one_epoch返回值")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 配置
    config = SpikeRPGNNConfig(
        hidden_dim=64,
        d_h=32,
        d_e=32,
        T=8,
        u_threshold=1.0,
        use_dvm=True,
        lambda_sr=0.01,
        ablation='full',
    )
    
    print(f"\n配置创建成功")
    print(f"  hidden_dim={config.hidden_dim}")
    print(f"  d_h={config.d_h}, d_e={config.d_e}")
    
    # 创建模型
    print(f"\n创建模型...")
    model = SpikeRPGNN(config, in_dim=3)
    model.train()
    
    device = torch.device('cpu')
    model = model.to(device)
    
    print(f"  模型创建成功")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    
    # 数据集
    print(f"\n加载数据集...")
    dataset = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES')
    
    sampler = EpisodeSampler(
        dataset,
        n_way=6,
        k_shot=1,
        q_query=5,
        seed=42,
    )
    
    print(f"  EpisodeSampler创建成功")
    
    # 运行一个epoch并检查结果
    print(f"\n运行train_one_epoch...")
    
    results = train_one_epoch(model, sampler, optimizer, config, device)
    
    print(f"\n返回值类型: {type(results)}")
    print(f"返回值内容:")
    
    if isinstance(results, dict):
        for key, value in results.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
    else:
        print(f"  {results}")
    
    # 检查模型是否真的训练了
    print(f"\n检查模型参数变化...")
    
    # 保存初始参数
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    
    # 再运行一个epoch
    print(f"再运行一个epoch...")
    results2 = train_one_epoch(model, sampler, optimizer, config, device)
    
    # 检查参数是否变化
    print(f"\n检查参数变化...")
    param_changed = False
    for name, param in model.named_parameters():
        if torch.equal(initial_params[name], param.data):
            pass  # 未变化
        else:
            param_changed = True
            print(f"  ✅ {name} 已更新")
            break
    
    if not param_changed:
        print(f"  ❌ 所有参数均未更新！")
        print(f"  可能原因：")
        print(f"    1. train_one_epoch没有执行优化步骤")
        print(f"    2. 梯度为0")
        print(f"    3. 优化器未正确连接")
    
    # 手动测试脉冲
    print(f"\n手动测试脉冲产生...")
    model.train()
    
    # 采样一个episode
    episode = sampler.sample_episode()
    
    # 前向传播
    support_data = episode['support']
    query_data = episode['query']
    
    # 简单批处理（假设support_data是图列表）
    from torch_geometric.data import Batch
    
    support_batch = Batch.from_data_list(support_data)
    support_batch = support_batch.to(device)
    
    # 编码
    z_support = model.encode(support_batch.x, support_batch.edge_index, support_batch.batch)
    
    # 检查脉冲
    spike_count = model.count_spikes()
    
    print(f"  手动测试脉冲数: {spike_count}")
    
    if spike_count > 0:
        print(f"  ✅ 模型可以产生脉冲！")
        print(f"  问题可能在train_one_epoch的实现")
    else:
        print(f"  ❌ 模型无法产生脉冲")
        print(f"  问题在模型本身")

if __name__ == "__main__":
    debug_train_one_epoch()
