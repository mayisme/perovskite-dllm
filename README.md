# ABO₃钙钛矿晶体结构扩散生成系统

基于扩散模型（DDPM/DDIM）生成有序氧化物ABO₃钙钛矿晶体结构。

## 特性

- **对数空间晶格扩散**：确保生成的晶格参数始终为正
- **PBC-aware图构建**：正确处理周期边界条件
- **精细物理约束**：Goldschmidt容忍因子、配位数、键长/键角
- **三级验证链路**：几何过滤 → ML势弛豫 → DFT确认
- **条件生成**：支持目标带隙和形成能的条件生成

## 安装

```bash
pip install torch pymatgen h5py pyyaml tqdm wandb
```

## 快速开始

### 1. 数据预处理

```bash
python main.py preprocess \
    --input data/raw/perovskites.json \
    --output data/processed/perovskites.h5 \
    --config configs/base.yaml
```

### 2. 训练模型

```bash
python main.py train \
    --config configs/base.yaml \
    --checkpoint-dir checkpoints \
    --device cuda
```

### 3. 生成结构

```bash
python main.py generate \
    --config configs/base.yaml \
    --checkpoint checkpoints/checkpoint_best.pt \
    --band-gap 3.0 \
    --formation-energy -5.0 \
    --num-samples 100 \
    --sampler ddpm \
    --output-dir outputs/generated
```

### 4. 验证结构

```bash
python main.py validate \
    --input-dir outputs/generated
```

## 项目结构

```
dllm-perovskite/
├── data/
│   ├── ionic_radii.py      # Shannon离子半径数据库
│   ├── filter.py           # 数据筛选器
│   ├── preprocess.py       # 数据预处理
│   └── dataset.py          # PyTorch数据集
├── models/
│   ├── diffusion.py        # 扩散调度器
│   ├── egnn.py             # E(n)-等变图神经网络
│   └── physics_loss.py     # 物理损失函数
├── train.py                # 训练框架
├── generate.py             # 生成模块
├── validate.py             # 验证模块
├── visualize.py            # 可视化模块
├── main.py                 # CLI入口
├── configs/
│   └── base.yaml           # 基础配置
└── tests/
    └── test_modules.py     # 模块测试
```

## 配置说明

配置文件 `configs/base.yaml` 包含以下部分：

- **data**: 数据路径、批大小、增强参数
- **model**: EGNN架构参数
- **diffusion**: 扩散时间步、调度类型
- **training**: 学习率、优化器、早停参数
- **physics_loss**: 物理损失权重
- **generation**: 采样参数、引导强度
- **validation**: 验证阈值

## 测试

运行模块测试：

```bash
python tests/test_modules.py
```

## 技术细节

### 对数空间晶格表示

晶格参数 (a,b,c,α,β,γ) 在扩散过程中，长度部分在对数空间表示：

```
(log a, log b, log c, α, β, γ)
```

这确保生成的晶格长度始终为正。

### PBC边构建

使用minimum-image convention计算周期边界条件下的最短原子间距：

```python
for dx, dy, dz in [-1, 0, 1]:
    offset = [dx, dy, dz]
    frac_j_image = frac_coords[j] + offset
    dist = ||cart_i - cart_j_image||
    min_dist = min(min_dist, dist)
```

### 物理损失

- **Goldschmidt容忍因子**: t = (r_A + r_O) / [√2(r_B + r_O)] ∈ [0.8, 1.0]
- **配位数**: B-O配位数 = 6 ± 0.5
- **键长**: B-O ∈ [1.8, 2.2]Å, A-O ∈ [2.5, 3.2]Å
- **Pauli排斥**: 原子间距 > 1.5Å

## 引用

如果使用本系统，请引用：

```bibtex
@software{perovskite_diffusion,
  title={ABO₃ Perovskite Crystal Structure Diffusion Generation System},
  year={2024}
}
```

## 许可证

MIT License
