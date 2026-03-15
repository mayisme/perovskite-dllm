# 可视化工具使用指南

## 概述

`visualize_results.py` 是 DLLM-Perovskite 项目的实验结果可视化工具，提供5种核心可视化功能，帮助分析模型训练、消融实验和生成样本的质量。

## 功能列表

| 功能 | 命令行参数 | 输出文件 | 说明 |
|------|-----------|---------|------|
| 训练/验证损失曲线 | `--loss-curves` | `figures/loss_curves.png` | 展示模型训练过程中的损失变化，标记最佳验证点 |
| 学习率对比 | `--lr-comparison` | `figures/lr_comparison.png` | 对比不同学习率配置的收敛速度和最终性能 |
| 消融实验结果 | `--ablation` | `figures/ablation_results.png` | 对比不同架构配置（pure EGNN/Transformer/hybrid）的性能 |
| 样本分布分析 | `--sample-dist` | `figures/sample_distribution.png` | 分析生成样本的晶格参数分布和相关性 |
| 基线模型对比 | `--baseline` | `figures/baseline_comparison.png` | 与其他生成模型（CDVAE、DiffCSP等）的性能对比 |

## 快速开始

### 1. 安装依赖

```bash
pip install matplotlib seaborn h5py numpy
```

### 2. 基本用法

```bash
# 进入实验目录
cd experiments/overnight_run

# 生成所有图表（推荐）
python visualize_results.py --all

# 只生成特定图表
python visualize_results.py --loss-curves
python visualize_results.py --lr-comparison
python visualize_results.py --ablation
python visualize_results.py --sample-dist
python visualize_results.py --baseline
```

### 3. 指定工作目录

如果不在 `experiments/overnight_run` 目录下运行，可以指定基础目录：

```bash
python visualize_results.py --all --base-dir /path/to/experiments/overnight_run
```

## 详细功能说明

### 1. 训练/验证损失曲线 (`plot_loss_curves`)

**输入文件**: `phase1_analysis_report.json`

**功能**:
- 绘制训练损失和验证损失随epoch的变化
- 标记最佳验证点（红色虚线和星号）
- 显示收敛状态（converged/not converged）

**输出示例**:
```
✓ Loss curves saved to figures/loss_curves.png
```

**图表解读**:
- 蓝色曲线：训练损失
- 紫色曲线：验证损失
- 红色虚线：最佳验证epoch
- 红色星号：最佳验证损失值
- 左上角文本框：收敛状态

**使用场景**:
- 检查模型是否收敛
- 识别过拟合（训练损失持续下降但验证损失上升）
- 确定最佳训练停止点

---

### 2. 学习率对比 (`plot_lr_comparison`)

**输入文件**: `logs/lr_*.log`（所有学习率实验日志）

**功能**:
- 并排对比不同学习率配置的训练/验证损失
- 支持多种学习率调度器（linear、warmup、cosine等）
- 自动从日志文件名提取学习率信息

**日志文件命名规范**:
```
lr_1e-4_linear_fast.log      # 学习率1e-4，linear调度器
lr_2e-4_warmup_fast.log      # 学习率2e-4，warmup调度器
lr_5e-5_cosine.log           # 学习率5e-5，cosine调度器
```

**日志格式要求**:
日志文件应包含类似以下格式的行：
```
Epoch 10: train_loss=1.234, val_loss=1.456
Epoch 20: Train Loss: 1.123, Val Loss: 1.345
```

**输出示例**:
```
✓ Learning rate comparison saved to figures/lr_comparison.png
```

**图表解读**:
- 左图：训练损失对比
- 右图：验证损失对比
- 不同颜色曲线代表不同学习率配置
- 更快收敛的曲线表示更优的学习率设置

**使用场景**:
- 选择最优学习率
- 评估学习率调度器的效果
- 识别学习率过大（震荡）或过小（收敛慢）的问题

---

### 3. 消融实验结果 (`plot_ablation_results`)

**输入文件**: `ablation_results.json`

**功能**:
- 对比不同架构配置的最终验证损失
- 对比不同架构的收敛速度（收敛epoch数）
- 高亮显示具体数值

**输入文件格式**:
```json
{
  "pure_egnn": {
    "final_val_loss": 2.05,
    "convergence_epoch": 45
  },
  "pure_transformer": {
    "final_val_loss": 2.12,
    "convergence_epoch": 50
  },
  "hybrid_current": {
    "final_val_loss": 1.89,
    "convergence_epoch": 40
  },
  "hybrid_reverse": {
    "final_val_loss": 1.95,
    "convergence_epoch": 42
  }
}
```

**输出示例**:
```
✓ Ablation results saved to figures/ablation_results.png
```

**图表解读**:
- 左图：最终验证损失（越低越好）
- 右图：收敛epoch数（越少越好）
- 不同颜色柱状图代表不同架构配置
- 柱状图上方标注具体数值

**架构配置说明**:
- `pure_egnn`: 纯EGNN架构（5层）
- `pure_transformer`: 纯Transformer架构（4层）
- `hybrid_current`: EGNN→Transformer混合架构（3+2层）
- `hybrid_reverse`: Transformer→EGNN混合架构（2+3层）

**使用场景**:
- 验证EGNN和Transformer的必要性
- 确定最优架构配置
- 分析架构顺序的影响

---

### 4. 样本分布分析 (`plot_sample_distribution`)

**输入文件**: `generated_samples.h5`

**功能**:
- 分析生成样本的晶格参数（a, b, c）分布
- 计算晶格参数的统计特性（均值、标准差）
- 可视化晶格参数之间的相关性

**HDF5文件格式要求**:
```python
# 文件应包含 'lattice_params' 数据集
# 形状: (n_samples, 3) 或 (n_samples, 6)
# 前3列为 a, b, c 参数（单位：Å）
```

**输出示例**:
```
✓ Sample distribution saved to figures/sample_distribution.png
```

**图表解读**:
- 左上：a参数直方图（蓝色）
- 右上：b参数直方图（紫色）
- 左下：c参数直方图（橙色）
- 右下：a vs c散点图（绿色）
- 红色虚线：均值
- 左下角文本框：相关系数

**使用场景**:
- 检查生成样本的多样性
- 验证晶格参数是否在合理范围内
- 识别生成模型的偏好（mode collapse）
- 分析晶格参数之间的物理关联

**合理范围参考**:
- 钙钛矿晶格参数通常在 3.5-6.5 Å 范围内
- a ≈ b（立方或四方对称）
- c/a 比值通常在 0.9-1.1 范围内

---

### 5. 基线模型对比 (`plot_baseline_comparison`)

**输入文件**: `baseline_comparison.json`

**功能**:
- 与其他晶体生成模型（CDVAE、DiffCSP等）对比
- 多维度评估：验证损失、有效率、多样性、训练时间
- 高亮最佳性能（红色边框）

**输入文件格式**:
```json
{
  "DLLM-Perovskite": {
    "val_loss": 1.89,
    "validity_rate": 0.92,
    "diversity": 0.87,
    "training_time": 120
  },
  "CDVAE": {
    "val_loss": 2.15,
    "validity_rate": 0.85,
    "diversity": 0.82,
    "training_time": 180
  },
  "DiffCSP": {
    "val_loss": 2.08,
    "validity_rate": 0.88,
    "diversity": 0.79,
    "training_time": 150
  }
}
```

**输出示例**:
```
✓ Baseline comparison saved to figures/baseline_comparison.png
```

**图表解读**:
- 左上：验证损失（越低越好）
- 右上：有效率（越高越好）
- 左下：多样性（越高越好）
- 右下：训练时间（越短越好）
- 红色边框：该指标的最佳模型

**指标说明**:
- **验证损失**: 模型在验证集上的重构损失
- **有效率**: 生成样本通过物理验证的比例
- **多样性**: 生成样本的唯一结构数/总样本数
- **训练时间**: 训练至收敛所需时间（分钟）

**使用场景**:
- 论文写作：与SOTA模型对比
- 模型选择：综合评估多个指标
- 性能分析：识别模型的优势和劣势

---

## 高级用法

### 1. 作为Python模块使用

```python
from visualize_results import ResultVisualizer

# 创建可视化工具
visualizer = ResultVisualizer(base_dir="experiments/overnight_run")

# 生成单个图表
visualizer.plot_loss_curves()
visualizer.plot_lr_comparison()

# 自定义保存路径
visualizer.plot_ablation_results(save_path="custom_path/ablation.png")

# 生成所有图表
visualizer.generate_all_plots()
```

### 2. 自定义图表样式

修改脚本开头的绘图配置：

```python
# 修改DPI（分辨率）
plt.rcParams['figure.dpi'] = 600  # 更高分辨率

# 修改字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# 修改颜色主题
sns.set_palette("husl")  # 或 "Set2", "Paired" 等
```

### 3. 批量处理多个实验

```bash
# 对比多个实验的结果
for exp in exp1 exp2 exp3; do
    python visualize_results.py --all --base-dir experiments/$exp
done
```

## 故障排除

### 问题1: 找不到输入文件

**错误信息**:
```
⚠ Ablation results file not found: ablation_results.json
```

**解决方案**:
- 检查文件是否存在：`ls experiments/overnight_run/ablation_results.json`
- 确认工作目录正确：使用 `--base-dir` 参数指定正确路径
- 如果文件确实不存在，工具会使用示例数据生成占位图表

### 问题2: 日志文件为空

**错误信息**:
```
⚠ No valid learning rate data found
```

**解决方案**:
- 检查日志文件是否为空：`wc -l logs/*.log`
- 确认日志格式符合要求（包含 "Epoch X: train_loss=Y, val_loss=Z"）
- 重新运行学习率实验生成日志

### 问题3: HDF5文件格式错误

**错误信息**:
```
⚠ 'lattice_params' not found in generated_samples.h5
```

**解决方案**:
- 检查HDF5文件内容：
  ```python
  import h5py
  with h5py.File('generated_samples.h5', 'r') as f:
      print(list(f.keys()))
  ```
- 确认数据集名称为 `lattice_params`
- 确认数据形状为 `(n_samples, 3)` 或 `(n_samples, 6)`

### 问题4: 依赖包缺失

**错误信息**:
```
ModuleNotFoundError: No module named 'seaborn'
```

**解决方案**:
```bash
pip install matplotlib seaborn h5py numpy
```

## 输出文件说明

所有图表默认保存在 `figures/` 目录下：

```
figures/
├── loss_curves.png           # 训练/验证损失曲线
├── lr_comparison.png         # 学习率对比
├── ablation_results.png      # 消融实验结果
├── sample_distribution.png   # 样本分布
└── baseline_comparison.png   # 基线模型对比
```

**图表规格**:
- 格式：PNG
- 分辨率：300 DPI（适合论文发表）
- 尺寸：10-16英寸宽，6-12英寸高
- 颜色：色盲友好配色方案

## 最佳实践

1. **定期生成图表**: 每次实验后立即运行 `--all`，及时发现问题
2. **版本控制**: 将图表保存到带时间戳的目录，便于对比不同版本
3. **论文准备**: 使用高DPI（600+）生成最终版本图表
4. **数据备份**: 保留原始JSON/HDF5文件，便于后续重新绘图
5. **自定义分析**: 基于 `ResultVisualizer` 类扩展自己的可视化功能

## 扩展开发

### 添加新的可视化功能

```python
class ResultVisualizer:
    # ... 现有代码 ...
    
    def plot_custom_metric(self, data_file: str, save_path: Optional[str] = None):
        """自定义可视化功能"""
        # 1. 读取数据
        with open(self.base_dir / data_file, 'r') as f:
            data = json.load(f)
        
        # 2. 数据处理
        # ...
        
        # 3. 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        # ...
        
        # 4. 保存
        if save_path is None:
            save_path = self.output_dir / "custom_metric.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Custom metric saved to {save_path}")
        plt.close()
```

## 参考资源

- **Matplotlib文档**: https://matplotlib.org/stable/contents.html
- **Seaborn文档**: https://seaborn.pydata.org/
- **HDF5文档**: https://docs.h5py.org/en/stable/
- **色盲友好配色**: https://colorbrewer2.org/

## 更新日志

- **v1.0** (2026-03-15): 初始版本，包含5种核心可视化功能
  - 训练/验证损失曲线
  - 学习率对比
  - 消融实验结果
  - 样本分布分析
  - 基线模型对比

## 联系与反馈

如有问题或建议，请在项目仓库提交Issue或Pull Request。
