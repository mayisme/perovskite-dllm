# DLLM for Perovskite Materials

基于 Diffusion Language Model 思想的钙钛矿材料生成项目。

## 项目背景

将图像扩散模型（如 Stable Diffusion）的"加噪 → 去噪"思路应用到钙钛矿晶体结构生成：
- 从随机噪声/掩码开始
- 通过迭代去噪生成稳定的 ABO₃ 钙钛矿结构
- 支持条件生成（如指定带隙、稳定性等属性）

## 核心特性

- **Discrete Diffusion**: 在晶体坐标和元素类型上进行扩散
- **E(n)-Equivariant**: 保持旋转/平移不变性
- **Conditional Generation**: 根据目标属性（带隙、稳定性）生成
- **Physics-Informed**: 集成 DFT 能量约束和 Goldschmidt 容忍因子

## 技术路线

### 方案 1: CrystalDiff (推荐起步)
专为 ABO₃ 钙钛矿设计的条件扩散模型
- E(n)-equivariant GNN + DDPM
- 支持从 Materials Project 自动获取数据
- 完整的训练/生成/验证/可视化流程

### 方案 2: MatterGen (大规模)
Microsoft 的通用晶体生成模型
- 支持任意无机晶体（包括钙钛矿）
- 预训练模型可直接使用
- 支持多属性条件生成

### 方案 3: 自定义 Micro-DLLM
从文本 DLLM 迁移到材料领域
- 将晶体序列化为 token 序列
- 使用 BPE tokenization
- 适合快速原型验证

## 项目结构

```
dllm-perovskite/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── setup.py                  # 安装脚本
├── data/                     # 数据集
│   ├── download.py          # 从 Materials Project 下载
│   └── preprocess.py        # 数据预处理
├── models/                   # 模型定义
│   ├── diffusion.py         # 扩散调度器
│   ├── egnn.py              # E(n)-equivariant GNN
│   └── conditional.py       # 条件生成模块
├── train.py                  # 训练脚本
├── generate.py               # 生成脚本
├── validate.py               # 验证脚本（RDF、能量）
├── visualize.py              # 可视化（3D 结构）
└── notebooks/                # Jupyter 实验笔记
    └── quick_start.ipynb    # 快速上手教程
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python data/download.py --dataset perov5 --output data/raw/
```

### 3. 训练模型

```bash
python train.py --config configs/micro.yaml --epochs 100
```

### 4. 生成新结构

```bash
python generate.py --checkpoint checkpoints/best.pt --bandgap 2.0 --num_samples 10
```

### 5. 验证和可视化

```bash
python validate.py --structures outputs/generated/*.cif
python visualize.py --structure outputs/generated/sample_0.cif
```

## 数据集

- **Perov-5**: ~19k 立方钙钛矿（CDVAE 提供）
- **Materials Project**: ABO₃ 子集（~10k 条）
- **自定义**: 支持从 CIF 文件导入

## 模型架构

### Diffusion Schedule
- Cosine schedule (更稳定)
- T=500 步（micro 实验）
- 支持 DDPM/DDIM 采样

### Backbone
- E(n)-equivariant GNN (EGNN)
- 消息传递 + 坐标更新
- 时间嵌入 + 属性嵌入

### 条件机制
- 带隙 (bandgap)
- 形成能 (formation energy)
- Goldschmidt 容忍因子

## 物理约束

- **RDF Loss**: 径向分布函数约束原子间距
- **Energy Penalty**: DFT 能量反馈
- **Symmetry**: 空间群约束（Pm-3m）
- **Tolerance Factor**: 0.8 < t < 1.0

## 实验结果

| 指标 | 目标 | 当前 |
|------|------|------|
| 生成成功率 | >80% | TBD |
| DFT 稳定性 | >70% | TBD |
| 带隙 MAE | <0.5 eV | TBD |
| 结构有效性 | >90% | TBD |

## 参考资源

### 论文
- CDVAE: Crystal Diffusion Variational AutoEncoder
- MatterGen: Microsoft Materials Generation
- UniMat: Unified Materials Representation
- DiffCSP: Diffusion for Crystal Structure Prediction

### 代码
- [CrystalDiff](https://github.com/adiManethia/CrystalDiff)
- [MatterGen](https://github.com/microsoft/mattergen)
- [CDVAE](https://github.com/txie-93/cdvae)
- [dllm](https://github.com/ZHZisZZ/dllm)

### 工具
- pymatgen: 晶体结构处理
- ASE: 原子模拟环境
- VESTA: 3D 可视化
- py3Dmol: Web 可视化

## TODO

- [ ] 实现基础 DDPM 框架
- [ ] 集成 EGNN backbone
- [ ] 添加条件生成
- [ ] 实现 RDF 验证
- [ ] 集成 DFT 能量计算
- [ ] 添加可视化界面
- [ ] 扩展到多元钙钛矿
- [ ] 支持缺陷和掺杂

## License

MIT

## 联系

如有问题或建议，欢迎提 issue。
