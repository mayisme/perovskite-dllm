# 物理损失启用实验设计

## 实验概述

本实验旨在系统评估物理损失（Physics Loss）对钙钛矿结构生成质量的影响。通过对比不同物理损失权重下的模型表现，确定最优的物理约束强度。

## 实验配置

### 配置文件

| 配置文件 | 物理损失权重 | 约束强度 | 适用场景 |
|---------|------------|---------|---------|
| `physics_0.05.yaml` | 0.05 | 温和 | 初步探索物理约束影响 |
| `physics_0.1.yaml` | 0.1 | 标准 | 平衡约束与多样性 |
| `physics_0.2.yaml` | 0.2 | 强约束 | 优先保证结构合理性 |

### 运行脚本

- **完整训练**: `./run_physics_experiments.sh`
  - 50 epochs，完整评估
  - 预计耗时: 数小时（取决于硬件）
  
- **快速测试**: `./run_physics_experiments_quick.sh`
  - 5 epochs，快速验证
  - 预计耗时: 数十分钟

## 物理损失组件

物理损失由4个关键组件构成，每个组件针对钙钛矿结构的特定物理约束：

### 1. Goldschmidt容忍因子约束

**作用**: 确保生成的钙钛矿结构满足几何稳定性条件

**公式**:
```
t = (r_A + r_X) / [√2 × (r_B + r_X)]
```

其中:
- `r_A`: A位阳离子半径
- `r_B`: B位阳离子半径  
- `r_X`: X位阴离子半径

**理想范围**: 0.8 ≤ t ≤ 1.0

**损失计算**:
```python
goldschmidt_loss = max(0, |t - 0.9| - 0.1)
```

**权重**: 0.1 (相对权重)

**物理意义**: 
- t < 0.8: 结构倾向于形成六方相或其他非钙钛矿相
- t > 1.0: 结构可能不稳定，容易发生相变
- 0.8 ≤ t ≤ 1.0: 稳定的立方或准立方钙钛矿相

### 2. 配位数约束

**作用**: 确保原子的配位环境符合化学规则

**标准配位数**:
- A位阳离子: 12配位（立方八面体间隙）
- B位阳离子: 6配位（八面体中心）
- X位阴离子: 2配位（连接两个B位）

**损失计算**:
```python
coordination_loss = |actual_coordination - expected_coordination|
```

**权重**: 0.05 (相对权重)

**物理意义**:
- 配位数偏差表明局部结构畸变
- 过高的配位数可能导致空间拥挤
- 过低的配位数可能导致结构不稳定

### 3. 键长约束

**作用**: 确保原子间距离在合理的化学键长范围内

**参考键长** (基于离子半径和):
- A-X键: 2.5-3.5 Å
- B-X键: 1.8-2.5 Å
- X-X键: > 3.0 Å (避免阴离子排斥)

**损失计算**:
```python
bond_length_loss = sum(max(0, |d - d_ref| - tolerance))
```

**权重**: 0.03 (相对权重)

**物理意义**:
- 键长过短: 强烈的Pauli排斥，结构不稳定
- 键长过长: 键合作用减弱，结构松散
- 合理键长: 平衡吸引和排斥作用

### 4. Pauli排斥约束

**作用**: 防止原子间距离过近导致的量子力学排斥

**最小距离阈值**:
- 同种原子: 1.5 Å
- 不同原子: 基于离子半径和的70%

**损失计算**:
```python
pauli_loss = sum(max(0, min_distance - actual_distance)^2)
```

**权重**: 0.1 (相对权重)

**物理意义**:
- 原子间距离过近会导致电子云重叠
- 强烈的Pauli排斥使结构能量急剧升高
- 这是结构稳定性的硬约束

## 总物理损失计算

```python
total_physics_loss = physics_loss_weight × (
    0.1 × goldschmidt_loss +
    0.05 × coordination_loss +
    0.03 × bond_length_loss +
    0.1 × pauli_loss
)
```

其中 `physics_loss_weight` 是全局权重参数（0.05, 0.1, 或 0.2）。

## 数值稳定性监控策略

### 1. 梯度监控

**监控指标**:
- 梯度范数 (Gradient Norm)
- 梯度裁剪频率
- 各损失组件的梯度贡献

**异常检测**:
```python
if grad_norm > 10.0:
    log_warning("Large gradient detected")
if grad_norm < 1e-6:
    log_warning("Vanishing gradient detected")
```

### 2. 损失监控

**监控指标**:
- 总损失 (Total Loss)
- 扩散损失 (Diffusion Loss)
- 物理损失 (Physics Loss)
- 各物理损失组件

**异常检测**:
```python
if physics_loss > 10 × diffusion_loss:
    log_warning("Physics loss dominates")
if physics_loss < 0.01 × diffusion_loss:
    log_warning("Physics loss too weak")
```

### 3. 数值稳定性检查

**检查项**:
- NaN/Inf检测
- 损失突变检测
- 学习率自适应调整

**处理策略**:
```python
if torch.isnan(loss) or torch.isinf(loss):
    skip_batch()
    log_error("Numerical instability detected")
```

### 4. 训练曲线分析

**关键指标**:
- 损失收敛速度
- 验证集性能
- 物理约束满足率

**预警阈值**:
- 连续5个epoch验证损失不下降 → 可能过拟合
- 物理损失持续上升 → 权重可能过大
- 扩散损失不收敛 → 权重可能过小

## 预期效果

### 1. physics_loss_weight = 0.05 (温和启用)

**预期表现**:
- ✓ 生成多样性较高
- ✓ 训练稳定性好
- ⚠ 物理约束满足率中等 (70-80%)
- ⚠ 可能生成部分不合理结构

**适用场景**:
- 探索性研究
- 需要高多样性的场景
- 初步验证物理损失效果

### 2. physics_loss_weight = 0.1 (标准启用)

**预期表现**:
- ✓ 平衡多样性与合理性
- ✓ 物理约束满足率高 (80-90%)
- ✓ 生成结构质量稳定
- ⚠ 多样性略有下降

**适用场景**:
- 标准生成任务
- 平衡探索与利用
- 推荐的默认配置

### 3. physics_loss_weight = 0.2 (强约束)

**预期表现**:
- ✓ 物理约束满足率很高 (>90%)
- ✓ 生成结构高度合理
- ⚠ 多样性显著下降
- ⚠ 可能陷入局部最优

**适用场景**:
- 需要高可靠性的场景
- 实验验证前的筛选
- 对结构合理性要求极高的任务

## 评估指标

### 1. 结构合理性指标

- **Goldschmidt容忍因子分布**: 应集中在 [0.8, 1.0]
- **配位数准确率**: 与标准配位数的匹配度
- **键长分布**: 是否在合理范围内
- **最小原子间距**: 是否满足Pauli排斥约束

### 2. 生成质量指标

- **结构多样性**: 通过SOAP描述符计算
- **能量分布**: 使用CHGNet预测形成能
- **稳定性评分**: 综合物理约束满足率

### 3. 训练效率指标

- **收敛速度**: 达到目标损失的epoch数
- **训练稳定性**: 损失曲线的平滑度
- **计算开销**: 每个epoch的训练时间

## 实验流程

### 1. 准备阶段

```bash
# 检查数据集
ls -lh data/processed/perovskites_20k_relaxed.h5

# 检查配置文件
ls -lh configs/physics_*.yaml
```

### 2. 快速测试

```bash
# 运行快速测试（5 epochs）
./run_physics_experiments_quick.sh

# 检查日志
cat experiments/physics_loss_quick_*/*/training.log
```

### 3. 完整训练

```bash
# 运行完整训练（50 epochs）
./run_physics_experiments.sh

# 监控训练进度
tail -f experiments/physics_loss_*/*/training.log
```

### 4. 结果分析

```bash
# 查看实验摘要
cat experiments/physics_loss_*/summary.txt

# 对比不同配置的效果
python analyze_physics_experiments.py \
    --experiment_dir experiments/physics_loss_*
```

## 故障排查

### 问题1: 训练不收敛

**可能原因**:
- 物理损失权重过大
- 学习率不合适
- 数据质量问题

**解决方案**:
1. 降低 `physics_loss_weight`
2. 调整学习率 (尝试 5e-5 或 2e-4)
3. 检查数据预处理

### 问题2: 物理约束不满足

**可能原因**:
- 物理损失权重过小
- 物理损失组件权重不平衡
- 模型容量不足

**解决方案**:
1. 增加 `physics_loss_weight`
2. 调整各组件的相对权重
3. 增加模型层数或隐藏维度

### 问题3: 数值不稳定

**可能原因**:
- 梯度爆炸
- 损失计算中的除零错误
- 数据中的异常值

**解决方案**:
1. 启用梯度裁剪 (`grad_clip: 1.0`)
2. 添加数值稳定性检查
3. 清洗数据，移除异常样本

## 下一步工作

1. **实验执行**: 运行完整的物理损失实验
2. **结果分析**: 对比不同权重的效果
3. **超参数优化**: 基于实验结果微调配置
4. **模型部署**: 选择最优配置用于生成任务

## 参考文献

1. Goldschmidt, V. M. (1926). "Die Gesetze der Krystallochemie"
2. Shannon, R. D. (1976). "Revised effective ionic radii"
3. Bartel, C. J. et al. (2019). "Physical descriptor for the Gibbs energy of inorganic crystalline solids"

---

**创建时间**: 2026-03-15  
**版本**: 1.0  
**作者**: Phase 2.3 Sub-Agent
