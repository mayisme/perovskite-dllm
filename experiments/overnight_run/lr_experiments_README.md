# 学习率调优实验说明

## 实验目标

通过系统性的学习率调优实验，找到最适合钙钛矿扩散模型的学习率和调度器配置，提升模型训练效率和最终性能。

## 实验设计

### 实验配置

| 实验名称 | 学习率 | 调度器 | Warmup Steps | 说明 |
|---------|--------|--------|--------------|------|
| lr_5e-5_warmup | 5e-5 | cosine | 500 | 较小学习率 + warmup + cosine衰减 |
| lr_2e-4_warmup | 2e-4 | cosine | 500 | 较大学习率 + warmup + cosine衰减 |
| lr_1e-4_linear | 1e-4 | linear | - | 中等学习率 + linear衰减（baseline） |

### 实验假设

1. **Warmup机制**: 在训练初期使用warmup可以稳定训练，避免梯度爆炸
2. **Cosine衰减**: 相比linear衰减，cosine衰减在训练后期提供更平滑的学习率下降
3. **学习率范围**: 5e-5到2e-4是扩散模型常用的学习率范围

### 固定参数

所有实验使用相同的基础配置（来自 `configs/base.yaml`）：

- **模型**: Hybrid EGNN-Transformer (hidden_dim=128, 3 EGNN layers, 2 Transformer layers)
- **数据**: 20k relaxed perovskites, batch_size=16
- **训练**: 50 epochs, weight_decay=0.01, grad_clip=1.0
- **扩散**: 500 timesteps, cosine schedule, epsilon prediction

## 文件结构

```
experiments/overnight_run/
├── configs/
│   ├── lr_5e-5_warmup.yaml          # 实验1配置
│   ├── lr_2e-4_warmup.yaml          # 实验2配置
│   ├── lr_1e-4_linear.yaml          # 实验3配置
│   ├── lr_5e-5_warmup_fast.yaml     # 快速验证版本1
│   ├── lr_2e-4_warmup_fast.yaml     # 快速验证版本2
│   └── lr_1e-4_linear_fast.yaml     # 快速验证版本3
├── run_lr_experiments.sh            # 批量训练脚本（完整版）
├── run_lr_experiments_fast.sh       # 快速验证脚本（10 epochs）
├── logs/                            # 训练日志目录（自动创建）
└── lr_experiments_README.md         # 本文档
```

## 使用方法

### 1. 快速验证（推荐先运行）

快速验证版本使用10 epochs和小batch size，用于快速测试配置是否正确：

```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite
bash experiments/overnight_run/run_lr_experiments_fast.sh
```

预计耗时: 约1-2小时（CPU）

### 2. 完整实验

验证通过后，运行完整的50 epochs训练：

```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite
bash experiments/overnight_run/run_lr_experiments.sh
```

预计耗时: 约8-12小时（CPU，overnight run）

### 3. 单独运行某个实验

```bash
python train.py --config experiments/overnight_run/configs/lr_5e-5_warmup.yaml
```

## 评估指标

训练完成后，使用以下指标评估模型性能：

### 训练指标
- **训练损失曲线**: 观察收敛速度和稳定性
- **验证损失曲线**: 评估泛化能力
- **学习率曲线**: 验证调度器是否按预期工作

### 生成质量指标
- **结构有效性**: 原子间距、配位数、Goldschmidt容忍因子
- **能量分布**: 生成结构的能量合理性
- **多样性**: 生成结构的化学空间覆盖

## 结果分析

### 1. 训练曲线分析

```bash
python analyze_results.py --experiment_dir experiments/overnight_run/logs
```

### 2. 模型评估

```bash
# 评估最佳模型
python evaluate_model.py --checkpoint checkpoints/lr_5e-5_warmup/best_model.pt

# 生成样本
python generate.py --checkpoint checkpoints/lr_5e-5_warmup/best_model.pt --num_samples 100
```

### 3. 对比分析

对比三个实验的关键指标：

| 指标 | lr_5e-5_warmup | lr_2e-4_warmup | lr_1e-4_linear |
|------|----------------|----------------|----------------|
| 最终训练损失 | - | - | - |
| 最终验证损失 | - | - | - |
| 收敛速度 | - | - | - |
| 训练稳定性 | - | - | - |
| 生成质量 | - | - | - |

## 预期结果

### 假设1: Warmup有助于稳定训练
- **验证方法**: 对比 lr_5e-5_warmup 和 lr_1e-4_linear 的训练初期损失曲线
- **预期**: warmup配置在前500 steps更稳定，损失下降更平滑

### 假设2: Cosine衰减优于Linear衰减
- **验证方法**: 对比 lr_2e-4_warmup 和 lr_1e-4_linear 的后期训练曲线
- **预期**: cosine衰减在后期提供更好的微调效果

### 假设3: 学习率2e-4可能过大
- **验证方法**: 观察 lr_2e-4_warmup 的训练稳定性
- **预期**: 可能出现震荡，需要更长的warmup或更小的学习率

## 下一步计划

根据实验结果，可能的后续方向：

1. **微调最优配置**: 在最佳学习率附近进行更细粒度的搜索
2. **调整Warmup策略**: 尝试不同的warmup steps（100, 300, 1000）
3. **探索其他调度器**: 尝试exponential decay, step decay等
4. **学习率范围扩展**: 如果2e-4效果好，尝试更大的学习率（3e-4, 5e-4）

## 注意事项

1. **CPU训练速度**: 在CPU上训练较慢，建议使用快速验证版本先测试
2. **内存使用**: 注意监控内存使用，必要时减小batch_size
3. **Checkpoint管理**: 每个实验会生成独立的checkpoint目录
4. **日志保存**: 所有训练日志保存在 `experiments/overnight_run/logs/`

## 参考文献

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Learning Rate Schedules for Deep Learning](https://arxiv.org/abs/1908.06477)

---

**创建时间**: 2026-03-15  
**实验状态**: 配置已准备，等待运行  
**预计完成时间**: 2026-03-16（overnight run）
