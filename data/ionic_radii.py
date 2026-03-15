"""Shannon离子半径数据库查询接口。

使用pymatgen获取Shannon离子半径数据，用于计算Goldschmidt容忍因子和验证键长合理性。
"""
import logging
from typing import Optional, Dict
from pymatgen.core.periodic_table import Species

logger = logging.getLogger(__name__)


class IonicRadiiDatabase:
    """Shannon离子半径查询接口。

    提供元素离子半径查询功能，支持缓存和默认氧化态映射。
    """

    def __init__(self):
        """初始化离子半径数据库。"""
        self._cache: Dict[tuple, Optional[float]] = {}

        # 常见A位和B位元素的默认氧化态
        self._default_oxidation_states = {
            'A_site': {
                'Ba': 2, 'Sr': 2, 'Ca': 2, 'Pb': 2, 'La': 3,
                'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3, 'Eu': 3,
                'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3,
                'Tm': 3, 'Yb': 3, 'Lu': 3, 'Y': 3, 'Bi': 3,
                'K': 1, 'Na': 1, 'Rb': 1, 'Cs': 1
            },
            'B_site': {
                'Ti': 4, 'Zr': 4, 'Hf': 4, 'Sn': 4, 'Ge': 4,
                'Nb': 5, 'Ta': 5, 'V': 5, 'Sb': 5,
                'Fe': 3, 'Mn': 3, 'Cr': 3, 'Co': 3, 'Ni': 3,
                'Al': 3, 'Ga': 3, 'In': 3, 'Sc': 3,
                'W': 6, 'Mo': 6, 'Re': 6, 'Ru': 4, 'Rh': 3,
                'Ir': 4, 'Pt': 4, 'Cu': 2, 'Zn': 2, 'Mg': 2
            }
        }

        # 氧的默认氧化态
        self._oxygen_oxidation_state = -2

    def get_radius(
        self,
        element: str,
        oxidation_state: int,
        coordination: int = 6
    ) -> Optional[float]:
        """查询Shannon离子半径。

        Args:
            element: 元素符号（如 'Ba', 'Ti'）
            oxidation_state: 氧化态（如 +2, +4）
            coordination: 配位数（默认为6，对应八面体配位）

        Returns:
            离子半径（单位：Å），如果数据不存在则返回None
        """
        # 检查缓存
        cache_key = (element, oxidation_state, coordination)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 使用pymatgen Species获取Shannon半径
            species = Species(element, oxidation_state)
            radius = species.ionic_radius

            # pymatgen默认返回配位数为6的半径
            # 如果需要其他配位数，可能需要额外处理
            if radius is None:
                logger.warning(
                    f"Shannon radius not found for {element}{oxidation_state:+d} "
                    f"with coordination {coordination}"
                )

            # 缓存结果
            self._cache[cache_key] = radius
            return radius

        except Exception as e:
            logger.warning(
                f"Error querying radius for {element}{oxidation_state:+d}: {e}"
            )
            self._cache[cache_key] = None
            return None

    def get_default_oxidation_state(self, element: str, site: str) -> Optional[int]:
        """获取A/B位元素的默认氧化态。

        Args:
            element: 元素符号
            site: 'A_site' 或 'B_site'

        Returns:
            默认氧化态，如果元素不在默认列表中则返回None
        """
        if site not in ['A_site', 'B_site']:
            logger.warning(f"Invalid site: {site}. Must be 'A_site' or 'B_site'")
            return None

        return self._default_oxidation_states.get(site, {}).get(element)

    def get_oxygen_oxidation_state(self) -> int:
        """获取氧的氧化态。

        Returns:
            氧的氧化态（-2）
        """
        return self._oxygen_oxidation_state

    def compute_goldschmidt_tolerance(
        self,
        a_element: str,
        b_element: str,
        a_oxidation: Optional[int] = None,
        b_oxidation: Optional[int] = None
    ) -> Optional[float]:
        """计算Goldschmidt容忍因子。

        公式：t = (r_A + r_O) / [√2(r_B + r_O)]
        有效范围：[0.8, 1.0]

        Args:
            a_element: A位元素符号
            b_element: B位元素符号
            a_oxidation: A位氧化态（如果为None则使用默认值）
            b_oxidation: B位氧化态（如果为None则使用默认值）

        Returns:
            Goldschmidt容忍因子，如果无法计算则返回None
        """
        # 获取氧化态
        if a_oxidation is None:
            a_oxidation = self.get_default_oxidation_state(a_element, 'A_site')
        if b_oxidation is None:
            b_oxidation = self.get_default_oxidation_state(b_element, 'B_site')

        if a_oxidation is None or b_oxidation is None:
            logger.warning(
                f"Cannot determine oxidation states for {a_element}/{b_element}"
            )
            return None

        # 获取离子半径
        r_a = self.get_radius(a_element, a_oxidation, coordination=12)  # A位通常12配位
        r_b = self.get_radius(b_element, b_oxidation, coordination=6)   # B位6配位
        r_o = self.get_radius('O', self._oxygen_oxidation_state, coordination=6)

        if r_a is None or r_b is None or r_o is None:
            logger.warning(
                f"Cannot compute Goldschmidt factor: missing radius data for "
                f"{a_element}{a_oxidation:+d}, {b_element}{b_oxidation:+d}, or O"
            )
            return None

        # 计算容忍因子
        import math
        tolerance = (r_a + r_o) / (math.sqrt(2) * (r_b + r_o))

        return tolerance

    def clear_cache(self):
        """清空缓存。"""
        self._cache.clear()
