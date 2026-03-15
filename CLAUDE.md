# dllm-perovskite: ABO₃钙钛矿晶体结构扩散生成

## 项目概述
混合架构（EGNN + Equivariant Transformer）用于钙钛矿晶体结构的扩散生成模型。

## 环境配置

### 本地环境
- **设备**: Mac (CPU only, 可用MPS但当前使用CPU)
- **Python**: 3.13
- **PyTorch**: 2.10.0
- **关键依赖**: torch_cluster 1.6.3 (从源码编译)
- **Conda环境**: base

### 数据集
- **路径**: `data/processed/perovskites_20k_relaxed.h5`
- **样本数**: 4429个严格ABO₃钙钛矿
  - 训练集: 3661
  - 验证集: 371
  - 测试集: 397

### 模型架构
- **类型**: HybridEGNNTransformer
- **配置**: 
  - 3层FastEGNN（局部几何特征）
  - 2层Equivariant Transformer（全局依赖）
  - hidden_dim=128, num_heads=4, dropout=0.1
  - cutoff=6.0, max_neighbors=32

### 训练配置
- **检查点目录**: `checkpoints/exp_hybrid/`
- **最佳模型**: `checkpoints/exp_hybrid/best_model.pt`
- **配置文件**: `configs/base.yaml`
- **训练脚本**: `train.py`
- **生成脚本**: `generate.py`
- **评估脚本**: `validate.py`

### 当前状态
- ✅ 完成50 epochs训练（2026-03-15 22:34）
- ✅ 训练损失: ~1.98
- ✅ 验证损失: ~1.90
- ✅ 无NaN问题
- ⏳ 待评估生成质量
- ⏳ 待启用物理损失
- ⏳ 待与基线对比

### 已知问题
- 物理损失暂时禁用（physics_loss_weight=0.0）
- 极端晶格角度（5.459°-167.772°）可能导致数值不稳定
- CPU训练速度慢（~1.5-2.0 it/s）

### GPU服务器（如果可用）
- **SSH**: [待配置]
- **GPU**: [待配置]
- **Conda**: [待配置]
- **代码目录**: [待配置]

## 实验计划

### Phase 1: 模型评估（当前阶段）
1. 生成样本质量评估
2. 有效性检查（晶体结构合理性）
3. 多样性分析
4. 新颖性评估

### Phase 2: 消融实验
1. 纯EGNN vs 纯Transformer vs 混合架构
2. 不同层数配置
3. 不同注意力头数
4. 不同cutoff距离

### Phase 3: 超参数优化
1. 学习率调优
2. Batch size优化
3. Dropout率调整
4. 扩散步数优化

### Phase 4: 物理损失启用
1. 逐步增加physics_loss_weight
2. 数值稳定性监控
3. 生成质量对比

### Phase 5: 基线对比
1. CDVAE
2. DiffCSP
3. 其他晶体生成模型

## 文件结构
```
dllm-perovskite/
├── configs/
│   └── base.yaml          # 主配置文件
├── data/
│   └── processed/
│       └── perovskites_20k_relaxed.h5
├── models/
│   ├── diffusion.py       # 扩散模型
│   ├── egnn.py            # EGNN层
│   ├── fast_egnn.py       # 优化的EGNN
│   ├── equivariant_transformer.py  # 等变Transformer
│   ├── hybrid_model.py    # 混合架构
│   └── physics_loss.py    # 物理损失
├── checkpoints/
│   └── exp_hybrid/
│       └── best_model.pt  # 最佳模型
├── train.py               # 训练脚本
├── generate.py            # 生成脚本
├── validate.py            # 评估脚本
└── CLAUDE.md             # 本文件
```

## 快速命令

### 训练
```bash
python train.py --config configs/base.yaml --exp_name exp_hybrid
```

### 生成样本
```bash
python generate.py --checkpoint checkpoints/exp_hybrid/best_model.pt --num_samples 100 --output generated_samples.h5
```

### 评估
```bash
python validate.py --checkpoint checkpoints/exp_hybrid/best_model.pt --split test
```

## 论文撰写计划
- [ ] 方法部分：混合架构设计
- [ ] 实验部分：消融实验、基线对比
- [ ] 结果分析：生成质量、多样性、新颖性
- [ ] 讨论：优势、局限性、未来工作
