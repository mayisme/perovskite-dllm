"""
评估指标定义模块

本模块定义了基线对比实验中使用的所有评估指标。
包括: Match Rate, Coverage, Novelty, Validity, Training Efficiency

作者: DLLM-Perovskite Team
日期: 2026-03-15
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


@dataclass
class EvaluationMetrics:
    """评估指标结果容器"""
    match_rate: float
    coverage: float
    novelty: float
    validity_geometric: float
    validity_chemical: float
    validity_physical: float
    training_time: float
    convergence_epoch: int
    final_loss: float


# ============================================================================
# 1. Match Rate (匹配率)
# ============================================================================

def structure_rmsd(struct1: Structure, struct2: Structure) -> float:
    """
    计算两个结构的RMSD（Root Mean Square Deviation）
    
    Args:
        struct1: 第一个结构
        struct2: 第二个结构
    
    Returns:
        rmsd: RMSD值（Å）
    """
    matcher = StructureMatcher(
        ltol=0.2,  # 晶格容差
        stol=0.3,  # 位点容差
        angle_tol=5,  # 角度容差（度）
        primitive_cell=True,
        scale=True,
        attempt_supercell=False
    )
    
    if matcher.fit(struct1, struct2):
        return matcher.get_rms_dist(struct1, struct2)[0]
    else:
        return float('inf')


def match_rate(
    generated_structures: List[Structure],
    train_structures: List[Structure],
    threshold: float = 0.1
) -> float:
    """
    计算匹配率: 生成样本与训练集中最近邻的结构相似度
    
    Args:
        generated_structures: 生成的结构列表
        train_structures: 训练集结构列表
        threshold: 相似度阈值（RMSD < threshold 视为匹配）
    
    Returns:
        match_rate: 匹配样本的比例 [0, 1]
    
    解释:
        - 高匹配率(>0.6): 模型记忆训练数据（可能过拟合）
        - 低匹配率(<0.3): 模型生成新颖结构（但可能无效）
        - 目标: 中等匹配率(0.3-0.5)，平衡记忆和创新
    """
    matches = 0
    for gen_struct in generated_structures:
        min_rmsd = min([
            structure_rmsd(gen_struct, train_struct)
            for train_struct in train_structures
        ])
        if min_rmsd < threshold:
            matches += 1
    
    return matches / len(generated_structures)


# ============================================================================
# 2. Coverage (覆盖率)
# ============================================================================

def structure_distance(struct1: Structure, struct2: Structure) -> float:
    """
    计算两个结构的距离（基于晶格参数和组分）
    
    Args:
        struct1: 第一个结构
        struct2: 第二个结构
    
    Returns:
        distance: 结构距离
    """
    # 晶格参数距离
    lattice1 = struct1.lattice
    lattice2 = struct2.lattice
    
    lattice_dist = np.sqrt(
        (lattice1.a - lattice2.a)**2 +
        (lattice1.b - lattice2.b)**2 +
        (lattice1.c - lattice2.c)**2 +
        (lattice1.alpha - lattice2.alpha)**2 / 100 +
        (lattice1.beta - lattice2.beta)**2 / 100 +
        (lattice1.gamma - lattice2.gamma)**2 / 100
    )
    
    # 组分距离（简化为元素种类差异）
    comp1 = set(struct1.composition.elements)
    comp2 = set(struct2.composition.elements)
    comp_dist = len(comp1.symmetric_difference(comp2))
    
    return lattice_dist + comp_dist


def coverage(
    generated_structures: List[Structure],
    reference_structures: List[Structure],
    k: int = 10,
    threshold: float = 0.2
) -> float:
    """
    计算覆盖率: 生成样本覆盖化学空间的广度
    
    使用k-最近邻覆盖率
    
    Args:
        generated_structures: 生成的结构列表
        reference_structures: 参考结构列表（如测试集）
        k: 最近邻数量
        threshold: 覆盖阈值
    
    Returns:
        coverage: 被覆盖的参考结构比例 [0, 1]
    
    解释:
        - 高覆盖率(>0.7): 模型探索化学空间全面
        - 低覆盖率(<0.5): 模型陷入局部模式
        - 目标: >0.7
    """
    covered = 0
    for ref_struct in reference_structures:
        # 找到k个最近的生成结构
        distances = [
            structure_distance(ref_struct, gen_struct)
            for gen_struct in generated_structures
        ]
        k_nearest = sorted(distances)[:k]
        if min(k_nearest) < threshold:
            covered += 1
    
    return covered / len(reference_structures)


# ============================================================================
# 3. Novelty (新颖性)
# ============================================================================

def novelty(
    generated_structures: List[Structure],
    train_structures: List[Structure],
    threshold: float = 0.1
) -> float:
    """
    计算新颖性: 生成样本中新颖结构的比例
    
    Args:
        generated_structures: 生成的结构列表
        train_structures: 训练集结构列表
        threshold: 新颖性阈值（RMSD > threshold 视为新颖）
    
    Returns:
        novelty: 新颖样本的比例 [0, 1]
    
    新颖性类型:
        1. 组分新颖: 训练集中未见的A-B元素组合
        2. 结构新颖: 相同组分但不同晶格/坐标
        3. 性质新颖: 极端带隙或形成能
    
    解释:
        - 高新颖性(>0.7): 模型创新能力强（但需验证有效性）
        - 低新颖性(<0.3): 模型保守（但生成质量可能更高）
        - 目标: 0.5-0.7（平衡新颖性和有效性）
    """
    novel = 0
    for gen_struct in generated_structures:
        min_rmsd = min([
            structure_rmsd(gen_struct, train_struct)
            for train_struct in train_structures
        ])
        if min_rmsd > threshold:
            novel += 1
    
    return novel / len(generated_structures)


# ============================================================================
# 4. Validity (有效性)
# ============================================================================

def check_geometry(struct: Structure, min_distance: float = 1.5) -> bool:
    """
    几何有效性检查
    
    Args:
        struct: 待检查的结构
        min_distance: 最小原子间距（Å）
    
    Returns:
        valid: 是否几何有效
    
    检查项:
        - 最小原子间距 > 1.5 Å
        - 晶格参数 > 0
        - 晶格矩阵正定
    """
    # 检查晶格参数
    lattice = struct.lattice
    if lattice.a <= 0 or lattice.b <= 0 or lattice.c <= 0:
        return False
    
    # 检查晶格矩阵正定性
    if lattice.volume <= 0:
        return False
    
    # 检查最小原子间距
    for i, site1 in enumerate(struct):
        for j, site2 in enumerate(struct):
            if i >= j:
                continue
            dist = struct.get_distance(i, j)
            if dist < min_distance:
                return False
    
    return True


def check_chemistry(
    struct: Structure,
    goldschmidt_range: Tuple[float, float] = (0.8, 1.0)
) -> bool:
    """
    化学有效性检查
    
    Args:
        struct: 待检查的结构
        goldschmidt_range: Goldschmidt容忍因子范围
    
    Returns:
        valid: 是否化学有效
    
    检查项:
        - Goldschmidt容忍因子: 0.8 < t < 1.0
        - 配位数合理: A-site (12), B-site (6), O-site (2)
        - 氧化态平衡: A^{+2} B^{+4} O_3^{-2}
    """
    # 简化实现: 仅检查组分是否为ABO3
    comp = struct.composition
    elements = [str(el) for el in comp.elements]
    
    # 检查是否为三元素组分
    if len(elements) != 3:
        return False
    
    # 检查是否包含氧
    if 'O' not in elements:
        return False
    
    # 检查氧的比例（应接近3）
    o_ratio = comp['O'] / sum([comp[el] for el in elements if el != 'O'])
    if not (2.5 < o_ratio < 3.5):
        return False
    
    return True


def check_physics(struct: Structure) -> bool:
    """
    物理有效性检查（计算昂贵，可选）
    
    Args:
        struct: 待检查的结构
    
    Returns:
        valid: 是否物理有效
    
    检查项:
        - ML势能弛豫后能量合理
        - DFT优化收敛
        - 无虚频（声子计算）
    
    注意: 本函数为占位符，实际实现需要调用CHGNet/M3GNet或VASP
    """
    # 占位符实现
    return True


def validity(
    generated_structures: List[Structure],
    levels: List[str] = ["geometric", "chemical", "physical"]
) -> Dict[str, float]:
    """
    计算有效性: 生成样本中物理/化学有效结构的比例
    
    Args:
        generated_structures: 生成的结构列表
        levels: 验证级别列表
    
    Returns:
        validity_dict: {
            'geometric': 几何有效性,
            'chemical': 化学有效性,
            'physical': 物理有效性
        }
    
    目标:
        - geometric > 0.9
        - chemical > 0.7
        - physical > 0.5
    """
    valid_geometric = 0
    valid_chemical = 0
    valid_physical = 0
    
    for struct in generated_structures:
        # 第一级: 几何验证
        if "geometric" in levels and check_geometry(struct):
            valid_geometric += 1
            
            # 第二级: 化学验证
            if "chemical" in levels and check_chemistry(struct):
                valid_chemical += 1
                
                # 第三级: 物理验证（可选，计算昂贵）
                if "physical" in levels and check_physics(struct):
                    valid_physical += 1
    
    n = len(generated_structures)
    return {
        'geometric': valid_geometric / n if "geometric" in levels else None,
        'chemical': valid_chemical / n if "chemical" in levels else None,
        'physical': valid_physical / n if "physical" in levels else None
    }


# ============================================================================
# 5. Training Efficiency (训练效率)
# ============================================================================

def find_convergence_epoch(val_losses: List[float], patience: int = 5) -> int:
    """
    找到收敛轮数
    
    Args:
        val_losses: 验证损失列表
        patience: 早停耐心值
    
    Returns:
        convergence_epoch: 收敛轮数
    """
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch, loss in enumerate(val_losses):
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                return best_epoch
    
    return len(val_losses) - 1


def training_efficiency(training_log: Dict) -> Dict[str, float]:
    """
    计算训练效率
    
    Args:
        training_log: 训练日志（包含loss、时间、epoch）
    
    Returns:
        efficiency_dict: {
            'time_per_epoch': 每轮训练时间（秒）,
            'convergence_epoch': 收敛轮数,
            'final_loss': 最终损失,
            'total_time': 总训练时间（小时）
        }
    
    解释:
        - 快速收敛: 模型架构合理、优化器有效
        - 慢收敛: 可能需要调整学习率、架构
        - 目标: <5小时（在单GPU上）
    """
    return {
        'time_per_epoch': np.mean(training_log['epoch_time']),
        'convergence_epoch': find_convergence_epoch(training_log['val_loss']),
        'final_loss': training_log['val_loss'][-1],
        'total_time': sum(training_log['epoch_time']) / 3600
    }


# ============================================================================
# 6. 综合评估函数
# ============================================================================

def evaluate_all_metrics(
    generated_structures: List[Structure],
    train_structures: List[Structure],
    test_structures: List[Structure],
    training_log: Dict,
    config: Optional[Dict] = None
) -> EvaluationMetrics:
    """
    计算所有评估指标
    
    Args:
        generated_structures: 生成的结构列表
        train_structures: 训练集结构列表
        test_structures: 测试集结构列表
        training_log: 训练日志
        config: 配置字典（可选）
    
    Returns:
        metrics: 评估指标结果
    """
    # 默认配置
    if config is None:
        config = {
            'match_rate_threshold': 0.1,
            'coverage_k': 10,
            'coverage_threshold': 0.2,
            'novelty_threshold': 0.1,
            'validity_levels': ['geometric', 'chemical']
        }
    
    # 计算各项指标
    mr = match_rate(
        generated_structures,
        train_structures,
        threshold=config['match_rate_threshold']
    )
    
    cov = coverage(
        generated_structures,
        test_structures,
        k=config['coverage_k'],
        threshold=config['coverage_threshold']
    )
    
    nov = novelty(
        generated_structures,
        train_structures,
        threshold=config['novelty_threshold']
    )
    
    val = validity(
        generated_structures,
        levels=config['validity_levels']
    )
    
    eff = training_efficiency(training_log)
    
    return EvaluationMetrics(
        match_rate=mr,
        coverage=cov,
        novelty=nov,
        validity_geometric=val['geometric'],
        validity_chemical=val['chemical'],
        validity_physical=val.get('physical', 0.0),
        training_time=eff['total_time'],
        convergence_epoch=eff['convergence_epoch'],
        final_loss=eff['final_loss']
    )


# ============================================================================
# 7. 结果对比和可视化
# ============================================================================

def compare_models(
    metrics_dict: Dict[str, EvaluationMetrics],
    save_path: Optional[str] = None
) -> None:
    """
    对比多个模型的评估指标
    
    Args:
        metrics_dict: {model_name: EvaluationMetrics}
        save_path: 保存路径（可选）
    """
    import pandas as pd
    
    # 构建对比表格
    data = []
    for model_name, metrics in metrics_dict.items():
        data.append({
            'Model': model_name,
            'Match Rate': f"{metrics.match_rate:.3f}",
            'Coverage': f"{metrics.coverage:.3f}",
            'Novelty': f"{metrics.novelty:.3f}",
            'Validity (Geo)': f"{metrics.validity_geometric:.3f}",
            'Validity (Chem)': f"{metrics.validity_chemical:.3f}",
            'Validity (Phys)': f"{metrics.validity_physical:.3f}",
            'Training Time (h)': f"{metrics.training_time:.2f}",
            'Convergence Epoch': metrics.convergence_epoch,
            'Final Loss': f"{metrics.final_loss:.4f}"
        })
    
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("="*80)
    print("Available metrics:")
    print("  1. Match Rate - 生成样本与训练集的匹配度")
    print("  2. Coverage - 覆盖化学空间的广度")
    print("  3. Novelty - 新颖结构的比例")
    print("  4. Validity - 结构有效性（几何/化学/物理）")
    print("  5. Training Efficiency - 训练时间和收敛速度")
    print("="*80)
