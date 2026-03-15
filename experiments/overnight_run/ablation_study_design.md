# 消融实验设计文档

## 1. 实验目的

本消融实验旨在系统评估DLLM-Perovskite模型中不同架构组件的贡献，具体目标包括：

1. **验证EGNN的必要性**：评估等变图神经网络在捕获局部几何信息中的作用
2. **验证Transformer的必要性**：评估自注意力机制在捕获全局依赖关系中的作用
3. **验证架构顺序的影响**：对比"局部→全局"与"全局→局部"的信息流动方式
4. **确定最优架构配置**：为后续模型优化提供实证依据

## 2. 实验配置

### 2.1 配置对比表

| 配置名称 | EGNN层数 | Transformer层数 | 架构顺序 | 总参数量估计 | 设计意图 |
|---------|---------|----------------|---------|------------|---------|
| **pure_egnn** | 5 | 0 | N/A | ~1.2M | 纯局部几何建模，测试EGNN单独性能 |
| **pure_transformer** | 0 | 4 | N/A | ~1.5M | 纯全局依赖建模，测试Transformer单独性能 |
| **hybrid_current** | 3 | 2 | EGNN→Transformer | ~1.3M | 基线配置，先局部后全局 |
| **hybrid_reverse** | 3 | 2 | Transformer→EGNN | ~1.3M | 反向配置，先全局后局部 |

### 2.2 配置详细说明

#### pure_egnn.yaml
- **架构**：5层EGNN，无Transformer
- **特点**：
  - 完全依赖等变卷积捕获局部几何
  - 通过增加层数（5层）补偿缺失的全局建模能力
  - 保持旋转和平移等变性
- **预期优势**：几何精度高，物理约束强
- **预期劣势**：长程依赖建模能力弱

#### pure_transformer.yaml
- **架构**：4层Transformer，无EGNN
- **特点**：
  - 完全依赖自注意力机制捕获全局依赖
  - 通过增加层数（4层）补偿缺失的局部几何建模
  - 无显式等变性约束
- **预期优势**：全局依赖建模能力强
- **预期劣势**：几何精度可能较低，物理约束较弱

#### hybrid_current.yaml
- **架构**：3层EGNN → 2层Transformer
- **特点**：
  - 先通过EGNN提取局部几何特征
  - 再通过Transformer整合全局信息
  - 符合"从局部到全局"的认知流程
- **预期优势**：平衡局部和全局建模
- **预期劣势**：可能存在信息瓶颈

#### hybrid_reverse.yaml
- **架构**：2层Transformer → 3层EGNN
- **特点**：
  - 先通过Transformer建立全局上下文
  - 再通过EGNN细化局部几何
  - 符合"从全局到局部"的规划流程
- **预期优势**：全局指导下的局部优化
- **预期劣势**：可能丢失早期几何信息

## 3. 评估指标

### 3.1 生成质量指标

| 指标类别 | 具体指标 | 计算方法 | 目标值 |
|---------|---------|---------|-------|
| **结构有效性** | 有效率 | 通过物理验证的样本比例 | >80% |
| | 最小原子间距 | 所有原子对的最小距离 | >1.5Å |
| | 配位数准确率 | 符合标准配位数的比例 | >90% |
| **物理合理性** | Goldschmidt容忍因子 | t = (r_A + r_O) / [√2(r_B + r_O)] | 0.8-1.0 |
| | 键长分布 | 与训练集的KL散度 | <0.1 |
| | 能量分布 | CHGNet预测能量的均值和方差 | 接近训练集 |
| **多样性** | 结构多样性 | 唯一结构数 / 总样本数 | >0.95 |
| | 组分多样性 | 唯一化学式数 / 总样本数 | >0.90 |
| | 空间群覆盖率 | 生成的空间群种类数 | >10 |

### 3.2 训练效率指标

| 指标 | 说明 | 单位 |
|-----|------|-----|
| 训练时间 | 达到收敛所需的总时间 | 小时 |
| 收敛速度 | 验证损失降至阈值的epoch数 | epoch |
| 内存占用 | 训练时的峰值内存 | GB |
| 推理速度 | 生成100个样本的时间 | 秒 |

### 3.3 模型性能指标

| 指标 | 说明 | 计算方法 |
|-----|------|---------|
| 重构误差 | 模型重构训练样本的精度 | MSE(x_0, x̂_0) |
| 去噪能力 | 不同噪声水平下的去噪效果 | PSNR |
| 泛化能力 | 测试集上的性能 | 测试集指标 |

## 4. 对比维度

### 4.1 架构组件贡献分析

**对比组合**：
- **EGNN贡献**：pure_egnn vs pure_transformer
  - 评估等变性约束对几何精度的影响
  - 评估局部建模对物理合理性的影响

- **Transformer贡献**：pure_transformer vs pure_egnn
  - 评估全局注意力对多样性的影响
  - 评估长程依赖对复杂结构的建模能力

- **混合架构优势**：hybrid_current vs (pure_egnn + pure_transformer)
  - 评估组合架构是否优于单一架构
  - 量化协同效应的大小

### 4.2 架构顺序影响分析

**对比组合**：
- **顺序影响**：hybrid_current vs hybrid_reverse
  - 评估"局部→全局"与"全局→局部"的性能差异
  - 分析信息流动方向对生成质量的影响
  - 识别最优的特征提取顺序

### 4.3 参数效率分析

**对比维度**：
- 参数量 vs 性能：评估每个配置的参数效率
- 计算量 vs 性能：评估每个配置的计算效率
- 训练时间 vs 性能：评估每个配置的训练效率

## 5. 预期结果

### 5.1 假设1：混合架构优于单一架构

**预期**：
- hybrid_current 和 hybrid_reverse 的综合性能优于 pure_egnn 和 pure_transformer
- 混合架构在生成质量和多样性上取得更好的平衡

**验证方法**：
- 对比各配置的综合评分（加权平均所有指标）
- 分析各配置在不同指标上的优劣势

### 5.2 假设2：EGNN对几何精度至关重要

**预期**：
- pure_egnn 在结构有效性和物理合理性指标上优于 pure_transformer
- 包含EGNN的配置（pure_egnn, hybrid_current, hybrid_reverse）在几何精度上显著优于 pure_transformer

**验证方法**：
- 对比最小原子间距、配位数准确率、Goldschmidt容忍因子等指标
- 统计物理验证失败的样本类型

### 5.3 假设3：Transformer对多样性至关重要

**预期**：
- pure_transformer 在结构多样性和组分多样性指标上优于 pure_egnn
- 包含Transformer的配置在生成样本的多样性上显著优于 pure_egnn

**验证方法**：
- 对比唯一结构数、唯一化学式数、空间群覆盖率等指标
- 分析生成样本的分布特征

### 5.4 假设4：架构顺序影响性能

**预期**：
- hybrid_current（局部→全局）在几何精度上优于 hybrid_reverse
- hybrid_reverse（全局→局部）在多样性上可能优于 hybrid_current
- 两者的综合性能接近，但在不同任务上各有优势

**验证方法**：
- 直接对比 hybrid_current 和 hybrid_reverse 的所有指标
- 分析两者在不同类型样本上的表现差异

## 6. 实验流程

### 6.1 训练阶段
1. 使用相同的数据集和预处理流程
2. 使用相同的训练超参数（学习率、批大小、优化器等）
3. 训练至收敛或达到最大epoch数（50）
4. 保存最佳模型检查点

### 6.2 生成阶段
1. 使用最佳模型生成100个样本
2. 使用相同的采样参数（采样步数、温度等）
3. 保存生成的结构文件

### 6.3 评估阶段
1. 对生成样本进行物理验证
2. 计算所有评估指标
3. 生成可视化图表
4. 保存评估报告

### 6.4 对比分析
1. 汇总所有配置的评估结果
2. 生成对比表格和图表
3. 进行统计显著性检验
4. 撰写分析报告

## 7. 运行说明

### 7.1 批量运行
```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite
bash experiments/overnight_run/run_ablation_experiments.sh
```

### 7.2 单独运行某个配置
```bash
# 训练
python scripts/train.py \
    --config experiments/overnight_run/configs/pure_egnn.yaml \
    --output_dir experiments/overnight_run/results/pure_egnn

# 生成
python scripts/generate.py \
    --config experiments/overnight_run/configs/pure_egnn.yaml \
    --checkpoint experiments/overnight_run/results/pure_egnn/best_model.pt \
    --output_dir experiments/overnight_run/results/pure_egnn/generated

# 评估
python scripts/evaluate.py \
    --config experiments/overnight_run/configs/pure_egnn.yaml \
    --checkpoint experiments/overnight_run/results/pure_egnn/best_model.pt \
    --output_dir experiments/overnight_run/results/pure_egnn/evaluation
```

### 7.3 查看结果
```bash
# 查看训练日志
tail -f experiments/overnight_run/logs/pure_egnn_*.log

# 查看对比报告
cat experiments/overnight_run/results/ablation_comparison.md
```

## 8. 预期输出

### 8.1 文件结构
```
experiments/overnight_run/
├── configs/
│   ├── pure_egnn.yaml
│   ├── pure_transformer.yaml
│   ├── hybrid_current.yaml
│   └── hybrid_reverse.yaml
├── logs/
│   ├── pure_egnn_20260315_232200.log
│   ├── pure_transformer_20260315_235500.log
│   ├── hybrid_current_20260316_003000.log
│   └── hybrid_reverse_20260316_010500.log
├── results/
│   ├── pure_egnn/
│   │   ├── best_model.pt
│   │   ├── training_curves.png
│   │   ├── generated/
│   │   └── evaluation/
│   ├── pure_transformer/
│   ├── hybrid_current/
│   ├── hybrid_reverse/
│   └── ablation_comparison.md
├── run_ablation_experiments.sh
└── ablation_study_design.md
```

### 8.2 对比报告内容
- 各配置的性能对比表格
- 训练曲线对比图
- 生成样本质量对比图
- 统计显著性检验结果
- 结论和建议

## 9. 注意事项

1. **计算资源**：4个实验预计总耗时8-12小时（CPU模式）
2. **磁盘空间**：每个实验约需1-2GB存储空间
3. **随机种子**：所有实验使用相同的随机种子（42）以确保可重复性
4. **失败处理**：如果某个实验失败，脚本会继续运行其他实验
5. **中断恢复**：可以单独重新运行失败的实验

## 10. 后续工作

根据消融实验结果，可以进行以下后续工作：

1. **架构优化**：基于最优配置进行超参数调优
2. **深度消融**：进一步分析每一层的贡献
3. **组件替换**：尝试其他类型的图神经网络或注意力机制
4. **混合策略**：探索更复杂的混合架构（如交替堆叠）
5. **任务特化**：针对特定类型的钙钛矿优化架构

---

**文档版本**：v1.0  
**创建日期**：2026-03-15  
**最后更新**：2026-03-15  
**作者**：DLLM-Perovskite Team
