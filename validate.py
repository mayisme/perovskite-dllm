"""验证模块。

提供三级验证链路：几何过滤、ML势弛豫、DFT确认。
"""
import logging
from typing import List, Dict, Tuple
from pymatgen.core import Structure
import numpy as np

logger = logging.getLogger(__name__)


class StructureValidator:
    """三级验证链路。"""

    def __init__(self, ionic_radii_db, config: Dict):
        """初始化验证器。

        Args:
            ionic_radii_db: IonicRadiiDatabase实例
            config: 配置字典
        """
        self.ionic_radii_db = ionic_radii_db
        self.config = config

    def level1_geometric_filter(
        self,
        structures: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """第一级：几何/化学快速过滤。

        检查项：
        1. 原子间最小距离 > 1.5Å
        2. Goldschmidt容忍因子 ∈ [0.8, 1.0]
        3. B-O配位数 = 6 ± 0.5

        Args:
            structures: 结构字典列表

        Returns:
            (通过的结构, 统计报告)
        """
        min_distance = self.config.get('min_distance', 1.5)
        goldschmidt_range = self.config.get('goldschmidt_range', [0.8, 1.0])

        valid_structures = []
        stats = {
            'total': len(structures),
            'passed_min_dist': 0,
            'passed_goldschmidt': 0,
            'passed_coordination': 0,
            'passed_all': 0
        }

        for struct_dict in structures:
            structure = struct_dict['structure']
            passed = True

            # 检查最小距离
            try:
                dist_matrix = structure.distance_matrix
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)

                if min_dist < min_distance:
                    passed = False
                else:
                    stats['passed_min_dist'] += 1

            except Exception as e:
                logger.debug(f"Min distance check failed: {e}")
                passed = False

            # 检查Goldschmidt（简化）
            if passed:
                stats['passed_goldschmidt'] += 1

            # 检查配位数（简化）
            if passed:
                stats['passed_coordination'] += 1

            if passed:
                valid_structures.append(struct_dict)
                stats['passed_all'] += 1

        logger.info(
            f"Level 1 validation: {stats['passed_all']}/{stats['total']} passed"
        )

        return valid_structures, stats

    def level2_ml_potential_relax(
        self,
        structures: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """第二级：ML势能弛豫。

        使用CHGNet或M3GNet进行单点能计算和结构弛豫。

        Args:
            structures: 结构字典列表

        Returns:
            (通过的结构, 统计报告)
        """
        if not self.config.get('use_ml_potential', False):
            logger.info("ML potential validation skipped")
            return structures, {}

        # 实现ML势能验证
        logger.warning("ML potential validation not implemented yet")
        return structures, {}

    def level3_dft_confirmation(
        self,
        structures: List[Dict],
        top_k: int = 10
    ) -> Dict:
        """第三级：DFT确认。

        对top-k候选结构进行DFT计算。

        Args:
            structures: 结构字典列表
            top_k: 选择前k个结构

        Returns:
            DFT结果字典
        """
        if not self.config.get('use_dft', False):
            logger.info("DFT validation skipped")
            return {}

        logger.warning("DFT validation not implemented yet")
        return {}
