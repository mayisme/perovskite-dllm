# 实施计划：ABO₃钙钛矿晶体结构扩散生成系统

## 概述

基于现有代码库进行重构和升级，实现完整的ABO₃钙钛矿扩散生成系统。现有代码提供了基础框架（简单EGNN、基础扩散调度、初步物理损失），需要按照设计文档进行模块化重构、功能增强和严格验证。实施按数据层→模型层→训练层→生成层→验证层→评测/CLI的顺序递增推进。

## 任务

- [x] 1. 数据层：离子半径数据库与数据筛选
  - [-] 1.1 创建 `data/ionic_radii.py`，实现 `IonicRadiiDatabase` 类
    - 使用 pymatgen Species 获取 Shannon 离子半径（配位数=6）
    - 实现 `get_radius(element, oxidation_state, coordination)` 接口
    - 实现默认氧化态映射（A位：Ba+2, Sr+2, Ca+2, Pb+2, La+3；B位：Ti+4, Zr+4, Nb+5, Ta+5, Fe+3）
    - 缓存查询结果，缺失数据返回 None 并记录警告
    - _需求：1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 为 Goldschmidt 公式编写属性测试
    - **属性 9.5：Goldschmidt 公式数值正确性**
    - 验证 t = (r_A + r_O) / [√2(r_B + r_O)] 在合理输入范围内结果正确
    - **验证：需求 1.1, 9.4**

  - [~] 1.3 创建 `data/filter.py`，实现 `PerovskiteFilter` 类
    - 实现能量过滤 `filter_by_energy`（energy_above_hull < 0.1 eV）
    - 实现拓扑过滤 `filter_by_topology`（corner-sharing BO₆八面体连通性）
    - 实现配位数过滤 `filter_by_coordination`（B-O配位数 = 6 ± 0.5）
    - 实现氧化态过滤 `filter_by_oxidation_state`（电中性约束）
    - 实现去重 `deduplicate`（StructureMatcher ltol=0.2, stol=0.3, angle_tol=5）
    - _需求：2.1, 2.2, 2.3, 2.4, 2.5_

  - [~] 1.4 重构 `data/preprocess.py`，实现严格预处理流程
    - 集成 PerovskiteFilter 筛选链（能量→拓扑→配位→氧化态→去重）
    - 标准化为 primitive cell（5原子）
    - 提取晶格参数 (a,b,c,α,β,γ) 和分数坐标归一化到 [0,1)
    - 实现 `composition_aware_split`（按 A-B 元素组合分组，避免测试集泄漏）
    - 以 HDF5 格式保存（train/val/test 分组，含 frac_coords, lattice_params, atom_types, band_gap, formation_energy, space_group）
    - 输出统计报告（原始数量、过滤后数量、去重后数量、成分分布）
    - _需求：2.6, 2.7, 2.8, 2.9, 2.10_

  - [~] 1.5 重构 `data/dataset.py`，升级 `PerovskiteDataset` 类
    - 从 HDF5 分组（train/val/test）直接加载，移除手动 split 逻辑
    - 返回 frac_coords (5,3)、lattice_params (6,)、atom_types (5,)、band_gap、formation_energy
    - 实现数据增强 `augment_structure`（对称容许的小畸变 ±0.02Å）
    - 实现 collate 函数处理固定 5 原子结构的批处理
    - 支持 num_workers 可配置的多工作进程加载
    - _需求：3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2. 检查点 - 数据层验证
  - 确保所有数据层测试通过，如有问题请询问用户。

- [ ] 3. 模型层：扩散调度器与对数空间晶格
  - [~] 3.1 重构 `models/diffusion.py`，实现对数空间晶格扩散
    - 实现 `lattice_to_log_space`：将 (a,b,c,α,β,γ) 转换为 (log a, log b, log c, α, β, γ)
    - 实现 `log_space_to_lattice`：对数空间转回实空间
    - 实现 `wrap_frac_coords`：分数坐标包裹到 [0,1)
    - 重构 `add_noise`：晶格参数在对数空间加噪，分数坐标加噪后包裹
    - 将晶格表示从 3×3 矩阵改为 6 参数 (a,b,c,α,β,γ)
    - 支持立方相简化（只扩散标量 log(a)）
    - 实现晶格范围约束（a,b,c ∈ [3Å, 15Å]，角度 ∈ [60°, 120°]）
    - 支持余弦噪声调度和可配置时间步（T=100 到 T=1000）
    - _需求：4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3_

  - [ ]* 3.2 为扩散调度正确性编写属性测试
    - **属性 9.2a：alpha_bar 单调递减**
    - 验证 ∀ t, alpha_bar[t] > alpha_bar[t+1]
    - **验证：需求 5.1, 5.2**

  - [ ]* 3.3 为晶格参数正性编写属性测试
    - **属性 9.3：对数空间转换后晶格参数始终为正**
    - 验证 ∀ log_a, exp(log_a) > 0
    - **验证：需求 4.1, 4.4**

  - [ ]* 3.4 为扩散边界条件编写属性测试
    - **属性 9.2b：扩散边界条件**
    - 验证 alpha_bar[0] ≈ 1.0 且 alpha_bar[T-1] ≈ 0.0
    - **验证：需求 5.1**

- [ ] 4. 模型层：改进 EGNN 架构
  - [~] 4.1 重构 `models/egnn.py`，升级 `EGNNLayer`
    - 实现 PBC-aware 边构建 `build_edges_pbc`（minimum-image convention + 半径截断）
    - 替换全连接图为基于半径的边构建（cutoff=6.0Å）
    - 添加注意力机制用于加权消息聚合
    - 添加残差连接和层归一化
    - _需求：4(EGNN).1, 4(EGNN).2, 4(EGNN).5_

  - [~] 4.2 重构 `models/egnn.py`，升级 `EGNNModel`
    - 支持条件输入（时间步嵌入 + 目标属性嵌入：band_gap, formation_energy）
    - 输出噪声预测：坐标噪声 (B,N,3) + 晶格参数噪声 (B,6)
    - 处理批次中每个结构的可变原子数
    - 将晶格输出从 3×3 矩阵改为 6 参数
    - _需求：4(EGNN).3, 4(EGNN).4, 4(EGNN).6_

  - [ ]* 4.3 为 EGNN 等变性编写属性测试
    - **属性 9.1a：EGNN 旋转等变性**
    - 验证 ∀ rotation R, EGNN(R·x) ≈ R·EGNN(x)
    - **验证：需求 4(EGNN).3, 16.2**

  - [ ]* 4.4 为 PBC 距离正确性编写属性测试
    - **属性 9.4：minimum-image 距离 ≤ 直接距离**
    - 验证 ∀ coords, d_pbc(x, y) ≤ d_direct(x, y)
    - **验证：需求 4(EGNN).1, 16.3**

- [ ] 5. 模型层：物理信息损失函数
  - [~] 5.1 重构 `models/physics_loss.py`，实现 `PhysicsLoss` 类
    - 实现 `goldschmidt_loss`：容忍因子偏离 [0.8, 1.0] 的惩罚
    - 实现 `coordination_loss`：B-O 配位数偏离 6 的惩罚
    - 实现 `bond_length_loss`：B-O 键长 1.8-2.2Å，A-O 键长 2.5-3.2Å
    - 实现 `bond_angle_loss`：O-B-O 键角偏离 90°/180° 的惩罚
    - 实现 `bond_valence_sum_loss`：氧化态合理性验证
    - 重构 `pauli_repulsion_loss`：保持 PBC-aware 实现
    - 实现 `combined_loss`：可配置权重组合所有物理损失
    - 集成 IonicRadiiDatabase 用于 Goldschmidt 和键长计算
    - 所有物理损失作用于预测的 x₀（非噪声 xₜ）
    - _需求：7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 5.2 为物理损失编写单元测试
    - 测试 Goldschmidt 损失对已知钙钛矿（BaTiO₃）返回低损失
    - 测试 Pauli 排斥对重叠原子返回高惩罚
    - 测试 combined_loss 权重配置正确应用
    - _需求：7.4, 16.5_

- [ ] 6. 检查点 - 模型层验证
  - 确保所有模型层测试通过，如有问题请询问用户。

- [ ] 7. 训练层：扩散训练框架
  - [~] 7.1 重构 `train.py`，实现 `DiffusionTrainer` 类
    - 实现 `train_step`：联合扩散训练（对数空间晶格 + PBC 分数坐标）
    - 实现 `predict_x0_from_noise`：从噪声预测反推 x₀ 用于物理损失
    - 使用 AdamW 优化器（lr=5e-5, weight_decay=0.01）
    - 实现 CosineAnnealingLR 学习率调度
    - 实现梯度裁剪（max_norm=1.0）
    - 支持 CPU、CUDA、MPS 设备自动检测
    - _需求：6.1, 6.2, 7.4, 7.5_

  - [~] 7.2 在 `train.py` 中实现检查点和恢复机制
    - 每 N 个 epoch 保存检查点（模型状态 + 优化器状态 + epoch + 最佳验证损失）
    - 支持从最新检查点恢复训练
    - _需求：6.3, 6.4_

  - [~] 7.3 在 `train.py` 中实现验证和早停
    - 每 N 个 epoch 计算验证损失
    - 基于验证损失实现早停（patience=20）
    - 记录训练/验证损失到 TensorBoard 和 WandB
    - 记录训练时间、内存使用和吞吐量
    - _需求：6.5, 6.6, 6.7, 6.8_

  - [ ]* 7.4 为训练管道编写集成测试
    - 使用小型合成数据测试完整训练循环（2 个 epoch）
    - 验证损失下降、检查点保存、梯度裁剪生效
    - _需求：16.4_

- [ ] 8. 生成层：DDPM/DDIM 采样与条件生成
  - [~] 8.1 重构 `generate.py`，实现 `PerovskiteGenerator` 类
    - 实现 `ddpm_sample_step`：DDPM 逆向采样单步（对数空间晶格）
    - 实现 `ddim_sample_step`：DDIM 加速采样（eta=0.0 确定性）
    - 实现 `classifier_free_guidance`：无分类器引导（guidance_scale=3.0）
    - 实现 `generate` 主方法：从 N(0,1) 初始化 → 逆向扩散 → 转换为 pymatgen Structure
    - 支持条件生成（目标 band_gap 和 formation_energy）
    - 实现温度缩放用于多样性控制
    - _需求：5.4, 5.5, 5.6, 5.7, 8.1, 8.2, 8.4_

  - [~] 8.2 在 `generate.py` 中实现输出和元数据保存
    - 以 CIF 格式保存生成的结构
    - 保存生成元数据（条件参数、采样参数、时间戳）
    - 支持批量生成和摘要统计输出
    - _需求：8.3, 8.5, 8.6, 8.7_

- [ ] 9. 检查点 - 生成层验证
  - 确保所有生成层测试通过，如有问题请询问用户。

- [ ] 10. 验证层：三级验证链路
  - [~] 10.1 重构 `validate.py`，实现 `StructureValidator` 类的第一级验证
    - 实现 `level1_geometric_filter`：原子间最小距离 > 1.5Å
    - 验证 Goldschmidt 容忍因子 ∈ [0.8, 1.0]（集成 IonicRadiiDatabase）
    - 验证 B-O 配位数 = 6 ± 0.5
    - 验证键价和（氧化态合理性）
    - 验证 RDF 匹配预期钙钛矿模式
    - 检查空间群对称性
    - 输出每个标准的通过/失败验证报告
    - _需求：9.1, 9.2, 9.3, 9.4, 9.5, 9.7_

  - [~] 10.2 在 `validate.py` 中实现第二级 ML 势能弛豫
    - 集成 CHGNet 或 M3GNet 进行单点能计算
    - 实现可选的结构弛豫
    - 估算 energy_above_hull，过滤不稳定结构（E_hull > 0.2 eV）
    - _需求：9.6_

  - [~] 10.3 在 `validate.py` 中实现第三级 DFT 确认（可选）
    - 对 top-k 候选结构生成 VASP 输入文件
    - 实现 DFT 结果解析和 reranking
    - _需求：9.6_

  - [ ]* 10.4 为验证逻辑编写单元测试
    - 测试已知有效钙钛矿通过第一级验证
    - 测试已知无效结构被正确拒绝
    - 测试有效率计算正确
    - _需求：16.5_

- [ ] 11. 评测体系：生成质量指标
  - [~] 11.1 创建评测模块，实现 `GenerationMetrics` 类
    - 实现 `compute_validity_rate`：通过第一级验证的比例
    - 实现 `compute_uniqueness`：去重后的比例（StructureMatcher）
    - 实现 `compute_novelty`：与训练集不重复的比例
    - 实现 `compute_property_mae`：band_gap 和 formation_energy 的 MAE
    - 实现 `compute_coverage`：属性空间覆盖百分比
    - 实现 `compute_diversity`：成对结构距离均值
    - 实现 `compute_condition_satisfaction_rate`：满足目标属性 ±tolerance 的比例
    - 生成与 baseline 比较的质量报告
    - _需求：12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 12. 可视化模块
  - [~] 12.1 重构 `visualize.py`，实现 3D 结构可视化
    - 使用原子球体和键渲染 3D 晶体结构（plotly 交互式）
    - 按元素类型着色，显示晶胞边界
    - 支持交互式旋转/缩放，保存为 PNG 或 HTML
    - 支持批量可视化
    - _需求：10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [~] 12.2 在 `visualize.py` 中实现训练指标可视化
    - 绘制训练/验证损失曲线
    - 分别绘制物理损失组件
    - 绘制学习率调度
    - 生成属性分布比较图（真实值 vs 生成值）和 RDF 比较图
    - _需求：11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 13. 配置系统与日志
  - [~] 13.1 重构 `utils/config_utils.py` 和 `configs/base.yaml`
    - 按设计文档更新配置结构（data, model, diffusion, training, physics_loss, generation, validation 分组）
    - 添加扩散配置（timesteps, schedule_type, prediction_type）
    - 添加物理损失权重配置（goldschmidt, coordination, bond_length, bond_angle, bvs, pauli）
    - 添加验证配置（min_distance, goldschmidt_range, ml_potential_type, use_dft）
    - 实现配置参数验证（类型检查和有效范围）
    - _需求：14.1_

  - [~] 13.2 实现日志和监控系统
    - 实现可配置详细程度的日志（同时输出到控制台和文件）
    - 记录异常条件警告（NaN 损失、极端梯度）
    - 启动时记录系统信息（设备、内存、库版本）
    - 支持带时间戳和日志级别的结构化日志
    - _需求：13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 14. CLI 入口与模块集成
  - [~] 14.1 重构 `main.py`，升级 CLI 命令系统
    - 实现 `train` 命令：接受配置文件和覆盖参数
    - 实现 `generate` 命令：接受检查点、A/B 元素、目标属性、采样参数
    - 实现 `validate` 命令：接受结构文件，执行三级验证
    - 实现 `visualize` 命令：接受结构文件
    - 实现 `download-data` 命令：数据获取
    - 实现 `preprocess` 命令：数据预处理
    - 无效参数时显示有用的错误消息和使用示例
    - _需求：14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

  - [~] 14.2 集成所有模块，确保端到端数据流通畅
    - 连接数据层 → 模型层 → 训练层 → 生成层 → 验证层
    - 确保配置系统贯穿所有模块
    - 确保日志系统在所有模块中正确工作
    - _需求：14.1, 14.2, 14.3_

- [ ] 15. 最终检查点 - 全系统验证
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号以确保可追溯性
- 检查点任务确保增量验证，及时发现问题
- 属性测试验证核心正确性属性（等变性、单调性、正性、PBC 距离）
- 单元测试验证具体示例和边界情况
- 现有代码（data/dataset.py, models/diffusion.py, models/egnn.py 等）需要重构而非从零开始
