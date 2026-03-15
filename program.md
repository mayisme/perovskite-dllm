# autoresearch: dllm-perovskite

自主研究程序——让 AI agent 自动迭代优化 ABO₃ 钙钛矿扩散生成模型。

## Setup

开始新实验前，与用户确认：

1. **约定 run tag**：基于日期提议 tag（如 `mar16`）。分支 `autoresearch/<tag>` 必须不存在。
2. **创建分支**：`git checkout -b autoresearch/<tag>` 从当前 main 分支。
3. **阅读项目文件**：
   - `README.md` — 项目概述
   - `CLAUDE.md` — 环境配置、当前状态、实验计划
   - `train.py` — 训练框架（**agent 可修改**）
   - `models/hybrid_model.py` — 混合架构（**agent 可修改**）
   - `models/fast_egnn.py` — EGNN 层（**agent 可修改**）
   - `models/equivariant_transformer.py` — Transformer 层（**agent 可修改**）
   - `models/diffusion.py` — 扩散调度器（**agent 可修改**）
   - `models/physics_loss.py` — 物理损失（**agent 可修改**）
   - `configs/base.yaml` — 基础配置（**agent 可修改**）
   - `data/dataset.py` — 数据集加载（不修改）
   - `data/preprocess.py` — 数据预处理（不修改）
   - `main.py` — CLI 入口（不修改）
   - `metrics/evaluation_metrics.py` — 评估指标（不修改）
4. **验证数据**：确认 `data/processed/perovskites_20k_relaxed.h5` 存在。
5. **初始化 results.tsv**：创建仅含表头的 `results.tsv`。
6. **确认后开始实验**。

## 可修改文件

| 文件 | 说明 |
|------|------|
| `train.py` | 训练器：优化器、学习率、调度、训练循环 |
| `models/hybrid_model.py` | 混合架构：EGNN + Transformer 组合方式 |
| `models/fast_egnn.py` | EGNN 层：消息传递、边构建、更新规则 |
| `models/equivariant_transformer.py` | 等变 Transformer：注意力、前馈 |
| `models/diffusion.py` | 扩散调度器：噪声调度、采样策略 |
| `models/physics_loss.py` | 物理损失：Goldschmidt、配位数、键长 |
| `configs/base.yaml` | 超参数配置 |

## 不可修改文件

- `data/preprocess.py` — 数据预处理管线
- `data/dataset.py` — 数据集和 DataLoader
- `data/ionic_radii.py` — Shannon 离子半径数据库
- `main.py` — CLI 入口
- `metrics/evaluation_metrics.py` — 评估指标定义

## 实验约束

**时间预算**：每次实验训练 **10 个 epoch**（在 CPU Mac 上约 10-15 分钟）。这是固定的，确保实验可比较。

**评估指标**：`val_loss`（验证集噪声预测 MSE）— 越低越好。这是主要指标。

**辅助指标**（可选，耗时较长时跳过）：
- 生成样本的几何有效性
- 生成样本的化学有效性

**VRAM/内存**：Mac CPU 环境，注意内存使用。不要让模型太大。

**简洁性原则**：同等效果下，更简洁的代码更好。微小改进但增加大量复杂度不值得。删除代码且效果不变或更好是好结果。

## 运行实验

```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite

# 训练（固定 10 epoch）
python main.py train --config configs/base.yaml --checkpoint-dir checkpoints/autoresearch --device cpu > run.log 2>&1

# 提取结果
grep "val_loss" run.log | tail -1
grep "train_loss" run.log | tail -1
```

如果需要快速验证（不跑完整训练），可以临时将 `configs/base.yaml` 中 `epochs` 改为 3 做 sanity check，确认代码不崩溃后再跑完整 10 epoch。

## Output format

训练脚本会输出类似：

```
Epoch 9: train_loss=1.9800, val_loss=1.9000
```

提取关键指标：
```bash
grep "train_loss\|val_loss" run.log | tail -5
```

## Logging results

实验完成后记录到 `results.tsv`（tab 分隔，不要用逗号）。

表头和列：
```
commit	val_loss	train_loss	epochs	status	description
```

1. git commit hash（短，7 字符）
2. val_loss（如 1.9000）— 崩溃时用 0.0000
3. train_loss（如 1.9800）— 崩溃时用 0.0000
4. epochs 数
5. status: `keep`, `discard`, 或 `crash`
6. 简短描述

示例：
```
commit	val_loss	train_loss	epochs	status	description
a1b2c3d	1.9000	1.9800	10	keep	baseline (hybrid EGNN+Transformer)
b2c3d4e	1.8500	1.9200	10	keep	increase lr to 2e-4
c3d4e5f	2.1000	2.0500	10	discard	switch to linear schedule
d4e5f6g	0.0000	0.0000	0	crash	double hidden_dim (OOM)
```

## 实验循环

在专用分支上运行（如 `autoresearch/mar16`）。

LOOP FOREVER:

1. 查看 git 状态：当前分支/commit
2. 选择一个实验方向，修改相关文件（模型架构、超参数、训练策略等）
3. git commit -m "exp: <简短描述>"
4. 运行实验：`python main.py train --config configs/base.yaml --checkpoint-dir checkpoints/autoresearch --device cpu > run.log 2>&1`
5. 读取结果：`grep "train_loss\|val_loss" run.log | tail -5`
6. 如果 grep 输出为空，说明崩溃了。运行 `tail -n 50 run.log` 查看错误栈，尝试修复。连续失败超过 3 次则放弃该方向。
7. 记录结果到 results.tsv（不要 commit 这个文件，保持 untracked）
8. 如果 val_loss 改善（更低），保留 commit，推进分支
9. 如果 val_loss 相同或更差，`git reset --hard HEAD~1` 回退
10. 回到步骤 1

## 研究策略

你是一个自主研究者。**你自己决定做什么实验。** 不要按照预设清单机械执行。

### 原则

1. **先建立 baseline**：第一次运行必须是未修改的代码，记录基准 val_loss。
2. **从数据和代码中寻找灵感**：仔细阅读模型代码、训练日志、损失曲线，找到瓶颈和改进空间。
3. **一次只改一个变量**：确保你能归因每次实验的效果。
4. **根据结果调整方向**：如果某个方向有效，沿着它深挖；如果无效，换方向。不要固执。
5. **大胆尝试**：超参数微调和架构大改都可以。如果小改动收益递减，尝试更激进的变化。
6. **记住领域知识**：这是晶体结构生成——等变性、周期边界条件、物理约束都很重要。利用你对材料科学和扩散模型的理解来指导实验。

### 搜索空间

一切可修改文件中的内容都是你的搜索空间：模型架构、训练策略、超参数、损失函数、扩散过程。你可以自由探索，没有限制。

### 何时停下某个方向

- 连续 3 次实验没有改善 → 换方向
- 某个改动导致崩溃且难以修复 → 放弃，回退
- 改进极小（<0.5%）但代码复杂度大增 → 不值得

## 重要注意事项

- **设备**：当前在 Mac CPU 上运行，没有 GPU。所有 device 参数用 `cpu`。
- **数据集**：4429 个 ABO₃ 钙钛矿（train 3661, val 371, test 397）
- **当前最佳**：val_loss ≈ 1.90（50 epoch, hybrid EGNN+Transformer）
- **物理损失**：当前禁用（weight=0.0），启用时注意数值稳定性
- **训练速度**：CPU 上约 1.5-2.0 it/s，10 epoch 约 10-15 分钟

## NEVER STOP

实验循环开始后，**不要暂停询问用户是否继续**。不要问"要继续吗？"或"这是个好的停止点吗？"。用户可能在睡觉或离开电脑，期望你**无限期自主运行**直到被手动中断。

如果你用完了想法，更努力地思考——重读代码寻找新角度，尝试组合之前接近成功的实验，尝试更激进的架构变化。循环一直运行，直到用户中断你。

每个实验约 10-15 分钟，你可以每小时跑 4-6 个实验，一晚上跑约 30-50 个实验。用户醒来时会看到完整的实验结果日志。
