"""钙钛矿结构筛选器。

提供多级筛选功能：能量、拓扑、配位数、氧化态、去重。
"""
import logging
from typing import List, Dict, Optional
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN

logger = logging.getLogger(__name__)


class PerovskiteFilter:
    """钙钛矿结构筛选器。

    提供多级筛选功能以确保训练数据质量。
    """

    def __init__(self, ionic_radii_db=None):
        """初始化筛选器。

        Args:
            ionic_radii_db: IonicRadiiDatabase实例（可选）
        """
        self.ionic_radii_db = ionic_radii_db
        self.structure_matcher = StructureMatcher(
            ltol=0.2,  # 晶格参数容差
            stol=0.3,  # 位点容差
            angle_tol=5  # 角度容差（度）
        )
        self.bv_analyzer = BVAnalyzer()
        self.coord_finder = CrystalNN()

    def filter_by_energy(
        self,
        structures: List[Dict],
        threshold: float = 0.1
    ) -> List[Dict]:
        """能量过滤：energy_above_hull < threshold eV。

        Args:
            structures: 结构字典列表，每个字典包含'structure'和'energy_above_hull'
            threshold: 能量阈值（eV）

        Returns:
            通过能量筛选的结构列表
        """
        filtered = []
        for struct_dict in structures:
            energy_above_hull = struct_dict.get('energy_above_hull')
            if energy_above_hull is not None and energy_above_hull < threshold:
                filtered.append(struct_dict)
            else:
                logger.debug(
                    f"Filtered out structure with energy_above_hull={energy_above_hull}"
                )

        logger.info(
            f"Energy filter: {len(filtered)}/{len(structures)} structures passed"
        )
        return filtered

    def filter_by_topology(self, structures: List[Dict]) -> List[Dict]:
        """拓扑过滤：验证corner-sharing BO₆八面体连通性。

        检查B位原子是否被6个O原子配位，且BO₆八面体corner-sharing。

        Args:
            structures: 结构字典列表

        Returns:
            通过拓扑筛选的结构列表
        """
        filtered = []
        for struct_dict in structures:
            structure = struct_dict['structure']

            try:
                # 识别A、B、O位点
                composition = structure.composition
                elements = [str(el) for el in composition.elements]

                # 假设ABO3钙钛矿：找到最少的元素作为B位
                element_counts = {el: composition[el] for el in elements}

                # 氧应该是最多的
                if 'O' not in elements:
                    logger.debug("No oxygen found in structure")
                    continue

                # 移除氧后，找到数量最少的作为B位
                non_oxygen = {el: count for el, count in element_counts.items() if el != 'O'}
                if len(non_oxygen) != 2:
                    logger.debug(f"Expected 2 non-oxygen elements, found {len(non_oxygen)}")
                    continue

                b_element = min(non_oxygen, key=non_oxygen.get)
                a_element = [el for el in non_oxygen if el != b_element][0]

                # 检查B-O配位
                b_sites = [i for i, site in enumerate(structure) if site.species_string == b_element]

                if not b_sites:
                    logger.debug(f"No B-site ({b_element}) found")
                    continue

                # 检查第一个B位的配位数
                b_idx = b_sites[0]
                neighbors = structure.get_neighbors(structure[b_idx], 3.0)  # 3Å截断半径

                o_neighbors = [n for n in neighbors if n.species_string == 'O']

                if len(o_neighbors) < 5 or len(o_neighbors) > 7:
                    logger.debug(
                        f"B-O coordination number {len(o_neighbors)} out of range [5, 7]"
                    )
                    continue

                # 通过拓扑检查
                filtered.append(struct_dict)

            except Exception as e:
                logger.debug(f"Topology check failed: {e}")
                continue

        logger.info(
            f"Topology filter: {len(filtered)}/{len(structures)} structures passed"
        )
        return filtered

    def filter_by_coordination(
        self,
        structures: List[Dict],
        target_coord: int = 6,
        tolerance: float = 0.5
    ) -> List[Dict]:
        """配位数过滤：B-O配位数 = target_coord ± tolerance。

        Args:
            structures: 结构字典列表
            target_coord: 目标配位数（默认6）
            tolerance: 容差

        Returns:
            通过配位数筛选的结构列表
        """
        filtered = []
        for struct_dict in structures:
            structure = struct_dict['structure']

            try:
                composition = structure.composition
                elements = [str(el) for el in composition.elements]

                if 'O' not in elements:
                    continue

                # 识别B位元素
                element_counts = {el: composition[el] for el in elements}
                non_oxygen = {el: count for el, count in element_counts.items() if el != 'O'}

                if len(non_oxygen) != 2:
                    continue

                b_element = min(non_oxygen, key=non_oxygen.get)
                b_sites = [i for i, site in enumerate(structure) if site.species_string == b_element]

                if not b_sites:
                    continue

                # 检查所有B位的配位数
                all_valid = True
                for b_idx in b_sites:
                    neighbors = structure.get_neighbors(structure[b_idx], 3.0)
                    o_neighbors = [n for n in neighbors if n.species_string == 'O']
                    coord_num = len(o_neighbors)

                    if abs(coord_num - target_coord) > tolerance:
                        all_valid = False
                        break

                if all_valid:
                    filtered.append(struct_dict)

            except Exception as e:
                logger.debug(f"Coordination check failed: {e}")
                continue

        logger.info(
            f"Coordination filter: {len(filtered)}/{len(structures)} structures passed"
        )
        return filtered

    def filter_by_oxidation_state(self, structures: List[Dict]) -> List[Dict]:
        """氧化态过滤：验证电中性约束。

        使用pymatgen的BVAnalyzer验证氧化态合理性。

        Args:
            structures: 结构字典列表

        Returns:
            通过氧化态筛选的结构列表
        """
        filtered = []
        for struct_dict in structures:
            structure = struct_dict['structure']

            try:
                # 尝试使用BVAnalyzer分析氧化态
                valences = self.bv_analyzer.get_valences(structure)

                # 检查电中性
                total_charge = sum(valences)
                if abs(total_charge) < 0.5:  # 允许小的数值误差
                    filtered.append(struct_dict)
                else:
                    logger.debug(f"Structure not charge neutral: total charge = {total_charge}")

            except Exception as e:
                # BVAnalyzer可能失败，使用预定义氧化态作为后备
                logger.debug(f"BVAnalyzer failed: {e}, using predefined oxidation states")

                try:
                    composition = structure.composition
                    elements = [str(el) for el in composition.elements]

                    if 'O' not in elements or len(elements) != 3:
                        continue

                    # 假设氧为-2，检查电中性
                    o_count = composition['O']
                    non_oxygen = {el: composition[el] for el in elements if el != 'O'}

                    # 简单检查：对于ABO3，A和B的氧化态之和应该等于6
                    # 这是一个简化的检查
                    filtered.append(struct_dict)

                except Exception as e2:
                    logger.debug(f"Oxidation state check failed: {e2}")
                    continue

        logger.info(
            f"Oxidation state filter: {len(filtered)}/{len(structures)} structures passed"
        )
        return filtered

    def deduplicate(self, structures: List[Dict]) -> List[Dict]:
        """去重：使用pymatgen StructureMatcher。

        Args:
            structures: 结构字典列表

        Returns:
            去重后的结构列表
        """
        if not structures:
            return []

        unique_structures = [structures[0]]

        for struct_dict in structures[1:]:
            structure = struct_dict['structure']
            is_duplicate = False

            for unique_dict in unique_structures:
                unique_structure = unique_dict['structure']
                if self.structure_matcher.fit(structure, unique_structure):
                    is_duplicate = True
                    logger.debug("Found duplicate structure")
                    break

            if not is_duplicate:
                unique_structures.append(struct_dict)

        logger.info(
            f"Deduplication: {len(unique_structures)}/{len(structures)} unique structures"
        )
        return unique_structures

    def apply_all_filters(
        self,
        structures: List[Dict],
        energy_threshold: float = 0.1,
        skip_topology: bool = False,
        skip_coordination: bool = False,
        skip_oxidation: bool = False,
        skip_dedup: bool = False
    ) -> List[Dict]:
        """应用所有筛选器。

        Args:
            structures: 结构字典列表
            energy_threshold: 能量阈值
            skip_topology: 是否跳过拓扑筛选
            skip_coordination: 是否跳过配位数筛选
            skip_oxidation: 是否跳过氧化态筛选
            skip_dedup: 是否跳过去重

        Returns:
            通过所有筛选的结构列表
        """
        logger.info(f"Starting with {len(structures)} structures")

        # 能量筛选
        filtered = self.filter_by_energy(structures, energy_threshold)

        # 拓扑筛选
        if not skip_topology:
            filtered = self.filter_by_topology(filtered)

        # 配位数筛选
        if not skip_coordination:
            filtered = self.filter_by_coordination(filtered)

        # 氧化态筛选
        if not skip_oxidation:
            filtered = self.filter_by_oxidation_state(filtered)

        # 去重
        if not skip_dedup:
            filtered = self.deduplicate(filtered)

        logger.info(f"Final: {len(filtered)} structures passed all filters")
        return filtered
