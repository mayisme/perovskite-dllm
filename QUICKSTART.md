# 快速开始指南

## 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; import pymatgen; print('✓ 依赖安装成功')"
```

## 2. 运行测试

```bash
# 运行模块测试
python tests/test_modules.py

# 运行示例工作流程
python example_workflow.py

# 运行项目检查
./check_project.sh
```

## 3. 准备数据（示例）

如果你有Materials Project的数据：

```bash
python main.py preprocess \
    --input data/raw/perovskites.json \
    --output data/processed/perovskites.h5 \
    --config configs/base.yaml
```

## 4. 训练模型（示例）

```bash
python main.py train \
    --config configs/base.yaml \
    --checkpoint-dir checkpoints \
    --device cuda
```

训练参数可在 `configs/base.yaml` 中调整：
- `training.epochs`: 训练轮数
- `training.lr`: 学习率
- `model.hidden_dim`: 隐藏层维度
- `diffusion.timesteps`: 扩散时间步

## 5. 生成结构（示例）

```bash
python main.py generate \
    --config configs/base.yaml \
    --checkpoint checkpoints/checkpoint_best.pt \
    --band-gap 3.0 \
    --formation-energy -5.0 \
    --num-samples 100 \
    --sampler ddpm \
    --output-dir outputs/generated \
    --device cuda
```

## 6. 验证结构

```bash
python main.py validate \
    --input-dir outputs/generated
```

## 常见问题

### Q: 如何获取训练数据？

A: 从Materials Project下载ABO₃钙钛矿数据：
```python
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        formula="*O3",
        fields=["structure", "band_gap", "formation_energy_per_atom"]
    )
```

### Q: 训练需要多长时间？

A: 取决于数据集大小和硬件：
- 小数据集（~1000结构）+ GPU: 几小时
- 大数据集（~10000结构）+ GPU: 1-2天

### Q: 如何调整物理损失权重？

A: 编辑 `configs/base.yaml`:
```yaml
physics_loss:
  goldschmidt: 1.0
  coordination: 0.5
  bond_length: 0.3
  pauli: 1.0
```

### Q: 如何可视化生成的结构？

A: 使用pymatgen或VESTA：
```python
from pymatgen.core import Structure
from visualize import visualize_structure

structure = Structure.from_file("outputs/generated/generated_0.cif")
visualize_structure(structure, "structure.png")
```

## 项目结构速查

```
dllm-perovskite/
├── data/              # 数据管道
├── models/            # 模型定义
├── configs/           # 配置文件
├── tests/             # 测试脚本
├── train.py           # 训练入口
├── generate.py        # 生成入口
├── validate.py        # 验证入口
├── main.py            # CLI主入口
└── README.md          # 详细文档
```

## 下一步

1. 阅读 `README.md` 了解详细信息
2. 查看 `PROJECT_SUMMARY.md` 了解技术细节
3. 运行 `example_workflow.py` 熟悉工作流程
4. 准备数据并开始训练

## 获取帮助

```bash
# 查看命令帮助
python main.py --help
python main.py train --help
python main.py generate --help
```

## 引用

如果使用本项目，请引用：
```bibtex
@software{perovskite_diffusion,
  title={ABO₃ Perovskite Crystal Structure Diffusion Generation System},
  year={2024}
}
```
