# 🧪 ABO₃ Perovskite Diffusion Generation System (perovskite-dllm)

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**基于扩散模型（DDPM/DDIM） + EGNN + 物理约束 的 ABO₃ 钙钛矿晶体结构条件生成系统**。

## ✨ 核心特性
- 强物理约束（Goldschmidt 容忍因子、键长/键角、Pauli 排斥等）
- PBC-aware E(3)-等变 EGNN
- 对数空间晶格参数扩散
- 支持属性条件生成（带隙、形成能等）
- 三级验证（几何过滤 + ML弛豫 + DFT）

## 🚀 快速开始

```bash
git clone https://github.com/mayisme/perovskite-dllm.git
cd perovskite-dllm
pip install -r requirements.txt
