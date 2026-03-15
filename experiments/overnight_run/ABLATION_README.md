# 消融实验快速指南

## 📁 文件结构

```
experiments/overnight_run/
├── configs/                          # 实验配置文件
│   ├── pure_egnn.yaml               # 纯EGNN架构（5层）
│   ├── pure_transformer.yaml        # 纯Transformer架构（4层）
│   ├── hybrid_current.yaml          # 当前混合架构（3层EGNN → 2层Transformer）
│   └── hybrid_reverse.yaml          # 反向混合架构（2层Transformer → 3层EGNN）
├── ablation_study_design.md         # 详细实验设计文档
├── config_comparison.md             # 配置对比表格
└── run_ablation_experiments.sh      # 批量运行脚本
```

## 🚀 快速开始

### 1. 批量运行所有实验（推荐）

```bash
cd /Users/xiaoyf/Documents/Python/dllm-perovskite
bash experiments/overnight_run/run_ablation_experiments.sh
```

**预计耗时**：8-12小时（CPU模式）

### 2. 单独运行某个实验

```bash
# 以pure_egnn为例
CONFIG="experiments/overnight_run/configs/pure_egnn.yaml"
OUTPUT="experiments/overnight_run/results/pure_egnn"

# 训练
python scripts/train.py --config $CONFIG --output_dir $OUTPUT

# 生成
python scripts/generate.py \
    --config $CONFIG \
    --checkpoint $OUTPUT/best_model.pt \
    --output_dir $OUTPUT/generated

# 评估
python scripts/evaluate.py \
    --config $CONFIG \
    --checkpoint $OUTPUT/best_model.pt \
    --output_dir $OUTPUT/evaluation
```

## 📊 查看结果

### 训练日志
```bash
tail -f experiments/overnight_run/logs/pure_egnn_*.log
```

### 对比报告
```bash
cat experiments/overnight_run/results/ablation_comparison.md
```

### 可视化结果
```bash
open experiments/overnight_run/results/pure_egnn/training_curves.png
```

## 📖 文档说明

### ablation_study_design.md
完整的实验设计文档，包括：
- 实验目的和假设
- 详细的配置说明
- 评估指标定义
- 预期结果分析
- 运行说明

### config_comparison.md
快速参考的配置对比表格，包括：
- 架构配置对比
- 设计意图对比
- 参数量估计
- 计算复杂度对比
- 关键差异总结

## 🎯 实验目标

1. **验证EGNN的必要性**：评估等变图神经网络对几何精度的贡献
2. **验证Transformer的必要性**：评估自注意力机制对多样性的贡献
3. **验证架构顺序的影响**：对比"局部→全局"与"全局→局部"的效果
4. **确定最优架构配置**：为后续优化提供实证依据

## 📈 评估指标

- **生成质量**：有效率、物理合理性、多样性
- **训练效率**：训练时间、收敛速度、内存占用
- **模型性能**：重构误差、去噪能力、泛化能力

## ⚠️ 注意事项

1. 确保有足够的磁盘空间（每个实验约1-2GB）
2. 所有实验使用相同的随机种子（42）以确保可重复性
3. 如果某个实验失败，脚本会继续运行其他实验
4. 可以单独重新运行失败的实验

## 🔧 故障排除

### 实验失败
查看日志文件定位问题：
```bash
cat experiments/overnight_run/logs/pure_egnn_*.log | grep -i error
```

### 内存不足
减小batch_size或max_atoms：
```yaml
data:
  batch_size: 8  # 从16减小到8
  max_atoms: 20  # 从30减小到20
```

### 训练太慢
使用GPU加速：
```yaml
training:
  device: "cuda"  # 从"cpu"改为"cuda"
```

---

**创建日期**：2026-03-15  
**用途**：消融实验快速参考指南
