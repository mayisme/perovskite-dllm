# 消融实验配置对比表

## 1. 架构配置对比

| 配置名称 | EGNN层数 | Transformer层数 | 架构顺序 | Hidden Dim | Cutoff Radius | Num Heads | Dropout |
|---------|---------|----------------|---------|-----------|--------------|-----------|---------|
| **pure_egnn** | 5 | 0 | N/A | 128 | 6.0Å | 4 (unused) | 0.1 |
| **pure_transformer** | 0 | 4 | N/A | 128 | 6.0Å (unused) | 4 | 0.1 |
| **hybrid_current** | 3 | 2 | EGNN→Transformer | 128 | 6.0Å | 4 | 0.1 |
| **hybrid_reverse** | 3 | 2 | Transformer→EGNN | 128 | 6.0Å | 4 | 0.1 |

## 2. 设计意图对比

| 配置名称 | 设计意图 | 预期优势 | 预期劣势 |
|---------|---------|---------|---------|
| **pure_egnn** | 测试纯局部几何建模能力 | • 几何精度高<br>• 物理约束强<br>• 等变性保证 | • 长程依赖建模弱<br>• 多样性可能受限 |
| **pure_transformer** | 测试纯全局依赖建模能力 | • 全局依赖建模强<br>• 多样性高<br>• 灵活性好 | • 几何精度可能较低<br>• 无显式等变性<br>• 物理约束较弱 |
| **hybrid_current** | 基线配置，先局部后全局 | • 平衡局部和全局<br>• 符合认知流程<br>• 几何精度有保证 | • 可能存在信息瓶颈<br>• 全局信息传递延迟 |
| **hybrid_reverse** | 反向配置，先全局后局部 | • 全局指导局部优化<br>• 符合规划流程<br>• 可能提升多样性 | • 可能丢失早期几何信息<br>• 等变性约束延迟 |

## 3. 参数量估计

| 配置名称 | EGNN参数 | Transformer参数 | 总参数量 | 相对大小 |
|---------|---------|----------------|---------|---------|
| **pure_egnn** | ~1.2M | 0 | ~1.2M | 基准 |
| **pure_transformer** | 0 | ~1.5M | ~1.5M | +25% |
| **hybrid_current** | ~0.7M | ~0.6M | ~1.3M | +8% |
| **hybrid_reverse** | ~0.7M | ~0.6M | ~1.3M | +8% |

*注：参数量为粗略估计，实际值取决于具体实现细节*

## 4. 计算复杂度对比

| 配置名称 | 时间复杂度 | 空间复杂度 | 相对速度 |
|---------|-----------|-----------|---------|
| **pure_egnn** | O(N²) (邻域搜索) | O(N) | 快 |
| **pure_transformer** | O(N²) (自注意力) | O(N²) | 中等 |
| **hybrid_current** | O(N²) | O(N²) | 中等 |
| **hybrid_reverse** | O(N²) | O(N²) | 中等 |

*N为原子数，假设最大原子数为30*

## 5. 关键差异总结

### 5.1 架构层面
- **pure_egnn**：完全依赖等变卷积，通过增加层数（5层）补偿缺失的全局建模
- **pure_transformer**：完全依赖自注意力，通过增加层数（4层）补偿缺失的局部建模
- **hybrid_current**：标准混合，先提取局部特征再整合全局信息
- **hybrid_reverse**：反向混合，先建立全局上下文再细化局部几何

### 5.2 信息流动
- **pure_egnn**：局部 → 局部 → 局部 → 局部 → 局部
- **pure_transformer**：全局 → 全局 → 全局 → 全局
- **hybrid_current**：局部 → 局部 → 局部 → 全局 → 全局
- **hybrid_reverse**：全局 → 全局 → 局部 → 局部 → 局部

### 5.3 等变性约束
- **pure_egnn**：全程保持等变性
- **pure_transformer**：无显式等变性约束
- **hybrid_current**：前期保持等变性，后期放松
- **hybrid_reverse**：前期无约束，后期引入等变性

## 6. 实验假设

### 假设1：混合架构优于单一架构
- **验证方法**：对比 hybrid_current/hybrid_reverse 与 pure_egnn/pure_transformer
- **预期结果**：混合架构在综合性能上优于单一架构

### 假设2：EGNN对几何精度至关重要
- **验证方法**：对比包含EGNN的配置与 pure_transformer
- **预期结果**：包含EGNN的配置在结构有效性和物理合理性上显著更好

### 假设3：Transformer对多样性至关重要
- **验证方法**：对比包含Transformer的配置与 pure_egnn
- **预期结果**：包含Transformer的配置在结构多样性和组分多样性上显著更好

### 假设4：架构顺序影响性能
- **验证方法**：直接对比 hybrid_current 与 hybrid_reverse
- **预期结果**：
  - hybrid_current 在几何精度上更优
  - hybrid_reverse 在多样性上可能更优
  - 综合性能接近但各有侧重

## 7. 评估维度

### 7.1 生成质量
- 结构有效性（有效率、最小原子间距、配位数准确率）
- 物理合理性（Goldschmidt容忍因子、键长分布、能量分布）
- 多样性（结构多样性、组分多样性、空间群覆盖率）

### 7.2 训练效率
- 训练时间
- 收敛速度
- 内存占用
- 推理速度

### 7.3 模型性能
- 重构误差
- 去噪能力
- 泛化能力

## 8. 配置文件路径

```
experiments/overnight_run/configs/
├── pure_egnn.yaml          # 纯EGNN配置
├── pure_transformer.yaml   # 纯Transformer配置
├── hybrid_current.yaml     # 当前混合配置（基线）
└── hybrid_reverse.yaml     # 反向混合配置
```

## 9. 运行命令

```bash
# 批量运行所有实验
bash experiments/overnight_run/run_ablation_experiments.sh

# 单独运行某个配置
python scripts/train.py --config experiments/overnight_run/configs/pure_egnn.yaml
```

---

**创建日期**：2026-03-15  
**用途**：消融实验配置快速参考
