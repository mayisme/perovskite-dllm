# 基线对比实验设计

## 1. 背景与目标

### 1.1 研究问题
当前项目实现了基于EGNN+Transformer的混合架构扩散模型（DLLM-Perovskite）。为了验证该方法的有效性，需要与两个主流晶体生成基线模型进行对比：
- **CDVAE** (Crystal Diffusion Variational Autoencoder, ICLR 2022)
- **DiffCSP** (Crystal Structure Prediction by Joint Equivariant Diffusion, NeurIPS 2023)

### 1.2 对比目标
- 评估DLLM-Perovskite相对于CDVAE和DiffCSP的性能优势
- 识别各方法在钙钛矿生成任务上的优缺点
- 为后续改进提供实证依据

---

## 2. 基线模型架构分析

### 2.1 CDVAE核心架构

**论文**: Crystal Diffusion Variational Autoencoder for Periodic Material Generation (Xie et al., ICLR 2022)

**核心思想**:
- VAE框架 + 潜在空间扩散
- 分离编码器（DimeNet++）和解码器（GemNet）
- 在潜在空间中进行扩散，而非直接在坐标空间

**关键组件**:
1. **编码器**: DimeNet++ (SE(3)-invariant)
   - 输入: 晶体结构 (晶格 + 分数坐标 + 原子类型)
   - 输出: 潜在向量 z ∈ R^d
   - 特点: 球谐函数 + 方向性消息传递

2. **潜在扩散**:
   - 在潜在空间 z 上进行扩散过程
   - 使用标准高斯噪声
   - 时间步: T=1000

3. **解码器**: GemNet-dQ (SE(3)-equivariant)
   - 输入: 潜在向量 z + 噪声时间步 t
   - 输出: 去噪后的坐标和晶格参数
   - 特点: 二次交互 + 角度信息

**优势**:
- 潜在空间维度低，训练稳定
- DimeNet++/GemNet强大的几何建模能力
- SE(3)不变性/等变性保证

**劣势**:
- 两阶段训练（VAE预训练 + 扩散训练）
- 潜在空间可能丢失细节信息
- 计算成本高（球谐函数展开）

---

### 2.2 DiffCSP核心架构

**论文**: Crystal Structure Prediction by Joint Equivariant Diffusion (Jiao et al., NeurIPS 2023)

**核心思想**:
- 直接在晶格参数和分数坐标上联合扩散
- 使用分数坐标（而非笛卡尔坐标）天然处理周期性
- Periodic-E(3)-equivariant GNN

**关键组件**:
1. **联合扩散**:
   - 晶格参数: L ∈ R^{3×3} (晶格矩阵)
   - 分数坐标: F ∈ R^{N×3}
   - 同时加噪和去噪

2. **Periodic-E(3)-Equivariant GNN**:
   - 消息传递在分数坐标空间
   - 边构建考虑周期边界条件（minimum-image convention）
   - 输出对晶格变换等变

3. **晶格参数化**:
   - 使用Cholesky分解确保正定性
   - L = L_lower × L_lower^T
   - 扩散在Cholesky因子上进行

**优势**:
- 单阶段端到端训练
- 分数坐标天然处理周期性
- 联合建模晶格和坐标的相关性

**劣势**:
- 高维空间扩散（晶格9维 + 坐标3N维）
- 需要精心设计等变性约束
- 对初始化敏感

---

### 2.3 DLLM-Perovskite (当前方法)

**核心思想**:
- EGNN处理局部几何 + Transformer处理全局依赖
- 对数空间晶格扩散（确保正值）
- 物理损失集成（Goldschmidt、配位数、键长）

**关键组件**:
1. **混合架构**:
   - EGNN层: 3层，处理PBC边和局部消息传递
   - Transformer层: 2层，捕获长程依赖

2. **对数空间晶格扩散**:
   - log(a), log(b), log(c), α, β, γ
   - 避免负值问题

3. **物理损失**:
   - Goldschmidt容忍因子
   - 配位数约束
   - 键长分布
   - Pauli排斥

**优势**:
- 混合架构平衡局部和全局
- 物理损失提升化学合理性
- 对数空间稳定训练

**劣势**:
- 架构复杂度较高
- 物理损失权重需要调优
- 未充分利用对称性

---

## 3. 简化版实现方案

为了快速对比，我们设计简化版CDVAE和DiffCSP，保留核心思想但降低计算成本。

### 3.1 Baseline CDVAE Simple

**简化策略**:
1. 用EGNN替代DimeNet++（编码器）
2. 用EGNN替代GemNet（解码器）
3. 潜在维度: 64（原版256）
4. 时间步: 500（原版1000）

**架构**:
```
Encoder: EGNN (3层, hidden_dim=128) → z ∈ R^64
Latent Diffusion: DDPM on z
Decoder: EGNN (3层, hidden_dim=128) → (coords, lattice)
```

**训练流程**:
1. 预训练VAE (20 epochs)
2. 冻结编码器，训练潜在扩散 (30 epochs)

---

### 3.2 Baseline DiffCSP Simple

**简化策略**:
1. 用EGNN替代复杂的Periodic-E(3)-Equivariant GNN
2. 晶格参数化: 直接扩散 (a, b, c, α, β, γ)，不用Cholesky
3. 时间步: 500

**架构**:
```
Joint Diffusion: (lattice_params, frac_coords)
Denoising Model: EGNN (4层, hidden_dim=128)
```

**训练流程**:
1. 端到端联合训练 (50 epochs)

---

## 4. 对比指标定义

### 4.1 Match Rate (匹配率)
**定义**: 生成样本与训练集中最近邻的结构相似度

**计算方法**:
```python
def match_rate(generated_structures, train_structures, threshold=0.1):
    """
    Args:
        generated_structures: 生成的结构列表
        train_structures: 训练集结构列表
        threshold: 相似度阈值（RMSD < threshold 视为匹配）
    
    Returns:
        match_rate: 匹配样本的比例
    """
    matches = 0
    for gen_struct in generated_structures:
        min_rmsd = min([structure_rmsd(gen_struct, train_struct) 
                        for train_struct in train_structures])
        if min_rmsd < threshold:
            matches += 1
    return matches / len(generated_structures)
```

**解释**:
- 高匹配率: 模型记忆训练数据（可能过拟合）
- 低匹配率: 模型生成新颖结构（但可能无效）
- 目标: 中等匹配率（0.3-0.5），平衡记忆和创新

---

### 4.2 Coverage (覆盖率)
**定义**: 生成样本覆盖化学空间的广度

**计算方法**:
```python
def coverage(generated_structures, reference_structures, k=10):
    """
    使用k-最近邻覆盖率
    
    Args:
        generated_structures: 生成的结构列表
        reference_structures: 参考结构列表（如测试集）
        k: 最近邻数量
    
    Returns:
        coverage: 被覆盖的参考结构比例
    """
    covered = set()
    for ref_struct in reference_structures:
        # 找到k个最近的生成结构
        distances = [structure_distance(ref_struct, gen_struct) 
                     for gen_struct in generated_structures]
        k_nearest = sorted(distances)[:k]
        if min(k_nearest) < threshold:
            covered.add(ref_struct)
    return len(covered) / len(reference_structures)
```

**化学空间表示**:
- 组分空间: A-site, B-site元素组合
- 晶格空间: (a, b, c, α, β, γ)
- 性质空间: (带隙, 形成能)

**解释**:
- 高覆盖率: 模型探索化学空间全面
- 低覆盖率: 模型陷入局部模式
- 目标: >0.7

---

### 4.3 Novelty (新颖性)
**定义**: 生成样本中新颖结构的比例

**计算方法**:
```python
def novelty(generated_structures, train_structures, threshold=0.1):
    """
    Args:
        generated_structures: 生成的结构列表
        train_structures: 训练集结构列表
        threshold: 新颖性阈值（RMSD > threshold 视为新颖）
    
    Returns:
        novelty: 新颖样本的比例
    """
    novel = 0
    for gen_struct in generated_structures:
        min_rmsd = min([structure_rmsd(gen_struct, train_struct) 
                        for train_struct in train_structures])
        if min_rmsd > threshold:
            novel += 1
    return novel / len(generated_structures)
```

**新颖性类型**:
1. **组分新颖**: 训练集中未见的A-B元素组合
2. **结构新颖**: 相同组分但不同晶格/坐标
3. **性质新颖**: 极端带隙或形成能

**解释**:
- 高新颖性: 模型创新能力强（但需验证有效性）
- 低新颖性: 模型保守（但生成质量可能更高）
- 目标: 0.5-0.7（平衡新颖性和有效性）

---

### 4.4 Validity (有效性)
**定义**: 生成样本中物理/化学有效结构的比例

**计算方法**:
```python
def validity(generated_structures):
    """
    多级验证
    
    Returns:
        validity_dict: {
            'geometric': 几何有效性,
            'chemical': 化学有效性,
            'physical': 物理有效性
        }
    """
    valid_geometric = 0
    valid_chemical = 0
    valid_physical = 0
    
    for struct in generated_structures:
        # 第一级: 几何验证
        if check_geometry(struct):  # 最小距离、晶格正定性
            valid_geometric += 1
            
            # 第二级: 化学验证
            if check_chemistry(struct):  # Goldschmidt、配位数、氧化态
                valid_chemical += 1
                
                # 第三级: 物理验证（可选，计算昂贵）
                if check_physics(struct):  # ML势能弛豫、DFT
                    valid_physical += 1
    
    return {
        'geometric': valid_geometric / len(generated_structures),
        'chemical': valid_chemical / len(generated_structures),
        'physical': valid_physical / len(generated_structures)
    }
```

**验证标准**:
1. **几何有效性**:
   - 最小原子间距 > 1.5 Å
   - 晶格参数 > 0
   - 晶格矩阵正定

2. **化学有效性**:
   - Goldschmidt容忍因子: 0.8 < t < 1.0
   - 配位数合理: A-site (12), B-site (6), O-site (2)
   - 氧化态平衡: A^{+2} B^{+4} O_3^{-2}

3. **物理有效性**:
   - ML势能弛豫后能量合理
   - DFT优化收敛
   - 无虚频（声子计算）

**解释**:
- 目标: geometric > 0.9, chemical > 0.7, physical > 0.5

---

### 4.5 Training Efficiency (训练效率)
**定义**: 模型训练的时间和收敛速度

**计算方法**:
```python
def training_efficiency(training_log):
    """
    Args:
        training_log: 训练日志（包含loss、时间、epoch）
    
    Returns:
        efficiency_dict: {
            'time_per_epoch': 每轮训练时间（秒）,
            'convergence_epoch': 收敛轮数,
            'final_loss': 最终损失,
            'total_time': 总训练时间（小时）
        }
    """
    return {
        'time_per_epoch': np.mean(training_log['epoch_time']),
        'convergence_epoch': find_convergence_epoch(training_log['val_loss']),
        'final_loss': training_log['val_loss'][-1],
        'total_time': sum(training_log['epoch_time']) / 3600
    }
```

**收敛判断**:
- 验证损失连续5轮不下降
- 或达到预设阈值

**解释**:
- 快速收敛: 模型架构合理、优化器有效
- 慢收敛: 可能需要调整学习率、架构
- 目标: <5小时（在单GPU上）

---

## 5. 实验设置

### 5.1 数据集
- **训练集**: 16,000个ABO₃钙钛矿结构（Materials Project）
- **验证集**: 2,000个结构
- **测试集**: 2,000个结构
- **Composition-aware split**: 确保测试集包含未见组分

### 5.2 硬件
- GPU: NVIDIA RTX 3090 (24GB)
- CPU: 16核
- 内存: 64GB

### 5.3 训练超参数
| 参数 | CDVAE Simple | DiffCSP Simple | DLLM-Perovskite |
|------|--------------|----------------|-----------------|
| Batch Size | 16 | 16 | 16 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Epochs | 50 (20+30) | 50 | 50 |
| Timesteps | 500 | 500 | 500 |
| Hidden Dim | 128 | 128 | 128 |
| Layers | 3+3 | 4 | 3+2 |

### 5.4 生成设置
- 每个模型生成1000个样本
- 条件生成: 带隙=3.0 eV, 形成能=-5.0 eV/atom
- 采样器: DDPM (100步)

---

## 6. 实验流程

### 6.1 阶段1: 模型训练 (3天)
1. **Day 1**: 训练CDVAE Simple
   - 预训练VAE (20 epochs, ~4小时)
   - 训练潜在扩散 (30 epochs, ~6小时)

2. **Day 2**: 训练DiffCSP Simple
   - 端到端训练 (50 epochs, ~10小时)

3. **Day 3**: 训练DLLM-Perovskite
   - 端到端训练 (50 epochs, ~10小时)

### 6.2 阶段2: 结构生成 (1天)
- 每个模型生成1000个样本
- 保存为CIF文件
- 记录生成时间

### 6.3 阶段3: 评估 (2天)
1. **几何/化学验证** (快速)
   - Match Rate
   - Novelty
   - Validity (geometric + chemical)

2. **ML势能验证** (中速)
   - CHGNet弛豫
   - 能量分布
   - Validity (physical)

3. **覆盖率分析** (慢速)
   - 化学空间覆盖
   - 性质空间覆盖

### 6.4 阶段4: 结果分析 (1天)
- 生成对比表格
- 绘制可视化图表
- 撰写分析报告

---

## 7. 预期结果

### 7.1 性能预测

| 指标 | CDVAE Simple | DiffCSP Simple | DLLM-Perovskite |
|------|--------------|----------------|-----------------|
| Match Rate | 0.45 | 0.40 | 0.35 |
| Coverage | 0.65 | 0.70 | 0.75 |
| Novelty | 0.55 | 0.60 | 0.65 |
| Validity (Geo) | 0.85 | 0.90 | 0.92 |
| Validity (Chem) | 0.60 | 0.65 | 0.75 |
| Validity (Phys) | 0.40 | 0.45 | 0.55 |
| Training Time | 10h | 10h | 10h |
| Convergence Epoch | 35 | 40 | 30 |

### 7.2 优势分析

**CDVAE Simple**:
- 优势: 潜在空间平滑，生成稳定
- 劣势: 两阶段训练复杂，可能丢失细节

**DiffCSP Simple**:
- 优势: 端到端训练，联合建模晶格和坐标
- 劣势: 高维扩散，训练不稳定

**DLLM-Perovskite**:
- 优势: 物理损失提升化学有效性，混合架构平衡局部和全局
- 劣势: 架构复杂，超参数多

---

## 8. 后续工作

### 8.1 短期 (1-2周)
1. 实现简化版CDVAE和DiffCSP
2. 运行对比实验
3. 分析结果，撰写报告

### 8.2 中期 (1-2月)
1. 改进DLLM-Perovskite:
   - 引入空间群约束
   - 优化物理损失权重
   - 尝试更强的GNN（如GemNet）

2. 扩展评估:
   - DFT验证top-100样本
   - 实验合成可行性分析

### 8.3 长期 (3-6月)
1. 多尺度生成:
   - 从组分到结构的层次生成
   - 主动学习引导生成

2. 应用导向:
   - 光催化钙钛矿筛选
   - 高通量计算验证

---

## 9. 参考文献

1. Xie, T., Fu, X., Ganea, O. E., Barzilay, R., & Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. *ICLR 2022*.

2. Jiao, R., Huang, W., Lin, P., Han, J., Chen, P., Lu, Y., & Liu, Y. (2023). Crystal structure prediction by joint equivariant diffusion. *NeurIPS 2023*.

3. Klicpera, J., Groß, J., & Günnemann, S. (2020). Directional message passing for molecular graphs. *ICLR 2020*.

4. Klicpera, J., Giri, S., Margraf, J. T., & Günnemann, S. (2021). Fast and uncertainty-aware directional message passing for non-equilibrium molecules. *NeurIPS 2021*.

---

## 10. 附录: 实现清单

### 10.1 代码文件
- [ ] `models/cdvae_simple.py`: 简化版CDVAE实现
- [ ] `models/diffcsp_simple.py`: 简化版DiffCSP实现
- [ ] `configs/baseline_cdvae_simple.yaml`: CDVAE配置
- [ ] `configs/baseline_diffcsp_simple.yaml`: DiffCSP配置
- [ ] `evaluate_baselines.py`: 基线评估脚本
- [ ] `compare_results.py`: 结果对比脚本

### 10.2 评估指标
- [ ] `metrics/match_rate.py`: 匹配率计算
- [ ] `metrics/coverage.py`: 覆盖率计算
- [ ] `metrics/novelty.py`: 新颖性计算
- [ ] `metrics/validity.py`: 有效性验证
- [ ] `metrics/efficiency.py`: 训练效率分析

### 10.3 可视化
- [ ] `visualize_comparison.py`: 对比可视化
  - 训练曲线对比
  - 生成样本分布
  - 化学空间覆盖热图
  - 性质分布对比

### 10.4 文档
- [x] `baseline_comparison_design.md`: 本设计文档
- [ ] `baseline_comparison_results.md`: 实验结果报告
- [ ] `baseline_comparison_analysis.md`: 深度分析报告
