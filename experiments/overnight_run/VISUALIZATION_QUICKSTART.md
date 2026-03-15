# 可视化工具快速参考

## 一键生成所有图表
```bash
cd experiments/overnight_run
python visualize_results.py --all
```

## 单独生成图表
```bash
# 训练/验证损失曲线
python visualize_results.py --loss-curves

# 学习率对比
python visualize_results.py --lr-comparison

# 消融实验结果
python visualize_results.py --ablation

# 样本分布
python visualize_results.py --sample-dist

# 基线模型对比
python visualize_results.py --baseline
```

## 输出位置
所有图表保存在 `figures/` 目录：
- `loss_curves.png` - 训练/验证损失曲线
- `lr_comparison.png` - 学习率对比
- `ablation_results.png` - 消融实验结果
- `sample_distribution.png` - 样本分布
- `baseline_comparison.png` - 基线模型对比

## 测试脚本
```bash
python test_visualizer.py
```

## 详细文档
查看 `visualization_guide.md` 获取完整使用说明。
