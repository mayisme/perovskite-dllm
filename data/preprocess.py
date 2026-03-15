"""数据预处理模块。

从Materials Project原始数据进行筛选、标准化和拆分，保存为HDF5格式。
"""
import h5py
import json
import numpy as np
import os
import logging
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import random

from data.filter import PerovskiteFilter
from data.ionic_radii import IonicRadiiDatabase

logger = logging.getLogger(__name__)


def composition_aware_split(
    structures: List[Dict],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按A-B元素组合分组，避免测试集泄漏。

    Args:
        structures: 结构字典列表
        ratios: (train, val, test) 比例
        seed: 随机种子

    Returns:
        (train_structures, val_structures, test_structures)
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按(A_element, B_element)分组
    composition_groups = defaultdict(list)

    for struct_dict in structures:
        structure = struct_dict['structure']
        composition = structure.composition
        elements = sorted([str(el) for el in composition.elements if str(el) != 'O'])

        if len(elements) == 2:
            key = tuple(elements)
            composition_groups[key].append(struct_dict)
        else:
            logger.warning(f"Unexpected composition: {composition}")

    # 随机打乱组
    groups = list(composition_groups.values())
    random.shuffle(groups)

    # 按比例分配
    total_groups = len(groups)
    train_end = int(total_groups * ratios[0])
    val_end = train_end + int(total_groups * ratios[1])

    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    # 展平
    train_structures = [s for g in train_groups for s in g]
    val_structures = [s for g in val_groups for s in g]
    test_structures = [s for g in test_groups for s in g]

    logger.info(
        f"Composition-aware split: "
        f"train={len(train_structures)}, "
        f"val={len(val_structures)}, "
        f"test={len(test_structures)}"
    )

    return train_structures, val_structures, test_structures


def extract_lattice_params(structure: Structure) -> np.ndarray:
    """提取晶格参数 (a, b, c, α, β, γ)。

    Args:
        structure: pymatgen Structure对象

    Returns:
        晶格参数数组 [a, b, c, alpha, beta, gamma]
    """
    lattice = structure.lattice
    return np.array([
        lattice.a,
        lattice.b,
        lattice.c,
        lattice.alpha,
        lattice.beta,
        lattice.gamma
    ], dtype=np.float32)


def preprocess_perovskites(
    raw_data_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> None:
    """预处理钙钛矿数据并保存为HDF5格式。

    流程：
    1. 加载原始JSON数据
    2. 应用筛选器链
    3. 标准化为primitive cell
    4. 提取晶格参数和分数坐标
    5. Composition-aware split
    6. 保存到HDF5

    Args:
        raw_data_path: 原始JSON数据路径
        output_path: 输出HDF5文件路径
        config: 配置字典
    """
    logger.info(f"Loading raw data from {raw_data_path}...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"{raw_data_path} not found")

    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} raw structures")

    # 转换为筛选器所需格式
    structures = []
    for item in tqdm(raw_data, desc="Parsing structures"):
        try:
            struct_dict = item.get("structure")
            if struct_dict is None:
                continue

            structure = Structure.from_dict(struct_dict)
            structures.append({
                'structure': structure,
                'energy_above_hull': item.get('energy_above_hull', 0.0),
                'band_gap': item.get('band_gap', 0.0),
                'formation_energy': item.get('formation_energy_per_atom', 0.0),
                'material_id': item.get('material_id', 'unknown'),
                'space_group': item.get('spacegroup', {}).get('number', 0)
            })
        except Exception as e:
            logger.warning(f"Failed to parse structure: {e}")

    logger.info(f"Successfully parsed {len(structures)} structures")

    # 应用筛选器
    ionic_radii_db = IonicRadiiDatabase()
    perovskite_filter = PerovskiteFilter(ionic_radii_db)

    filtered_structures = perovskite_filter.apply_all_filters(
        structures,
        energy_threshold=config.get('energy_threshold', 0.1),
        skip_topology=config.get('skip_topology', False),
        skip_coordination=config.get('skip_coordination', False),
        skip_oxidation=config.get('skip_oxidation', False),
        skip_dedup=config.get('skip_dedup', False)
    )

    if not filtered_structures:
        raise ValueError("No structures passed filtering")

    # 标准化为primitive cell
    logger.info("Standardizing to primitive cells...")
    standardized_structures = []
    max_atoms = config.get('max_atoms', 20)  # 最大原子数限制

    for struct_dict in tqdm(filtered_structures, desc="Standardizing"):
        try:
            structure = struct_dict['structure']
            analyzer = SpacegroupAnalyzer(structure)
            primitive = analyzer.get_primitive_standard_structure()

            # 保留原子数在合理范围内的primitive cell
            # 5原子: 简单ABO3
            # 10原子: 有序钙钛矿或倾斜结构
            # 15-20原子: 复杂有序结构
            n_atoms = len(primitive)
            if 5 <= n_atoms <= max_atoms:
                struct_dict['structure'] = primitive
                struct_dict['n_atoms'] = n_atoms
                standardized_structures.append(struct_dict)
            else:
                logger.debug(f"Skipping structure with {n_atoms} atoms (max={max_atoms})")
        except Exception as e:
            logger.warning(f"Failed to standardize structure: {e}")

    logger.info(f"Standardized {len(standardized_structures)} structures to primitive cells")

    # Composition-aware split
    split_ratios = config.get('split_ratios', (0.8, 0.1, 0.1))
    train_structs, val_structs, test_structs = composition_aware_split(
        standardized_structures,
        ratios=split_ratios,
        seed=config.get('seed', 42)
    )

    # 保存到HDF5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for split_name, split_structs in [
            ('train', train_structs),
            ('val', val_structs),
            ('test', test_structs)
        ]:
            if not split_structs:
                continue

            logger.info(f"Saving {split_name} split with {len(split_structs)} structures")
            group = f.create_group(split_name)

            n_samples = len(split_structs)

            # 找出最大原子数用于padding
            max_atoms_in_split = max(len(s['structure']) for s in split_structs)
            logger.info(f"  Max atoms in {split_name}: {max_atoms_in_split}")

            # 创建数据集（使用最大原子数）
            frac_coords_ds = group.create_dataset(
                "frac_coords", (n_samples, max_atoms_in_split, 3), dtype=np.float32
            )
            lattice_params_ds = group.create_dataset(
                "lattice_params", (n_samples, 6), dtype=np.float32
            )
            atom_types_ds = group.create_dataset(
                "atom_types", (n_samples, max_atoms_in_split), dtype=np.int32
            )
            n_atoms_ds = group.create_dataset(
                "n_atoms", (n_samples,), dtype=np.int32
            )
            band_gap_ds = group.create_dataset(
                "band_gap", (n_samples,), dtype=np.float32
            )
            formation_energy_ds = group.create_dataset(
                "formation_energy", (n_samples,), dtype=np.float32
            )
            space_group_ds = group.create_dataset(
                "space_group", (n_samples,), dtype=np.int32
            )

            # 填充数据
            for i, struct_dict in enumerate(tqdm(split_structs, desc=f"Saving {split_name}")):
                structure = struct_dict['structure']
                n_atoms = len(structure)

                # 分数坐标归一化到[0,1)，padding到max_atoms
                frac_coords = structure.frac_coords % 1.0
                frac_coords_ds[i, :n_atoms] = frac_coords
                # padding部分保持为0

                # 晶格参数
                lattice_params = extract_lattice_params(structure)
                lattice_params_ds[i] = lattice_params

                # 原子类型，padding用0表示
                atom_types = np.array([site.specie.Z for site in structure], dtype=np.int32)
                atom_types_ds[i, :n_atoms] = atom_types

                # 记录实际原子数
                n_atoms_ds[i] = n_atoms

                # 属性
                band_gap_ds[i] = struct_dict['band_gap']
                formation_energy_ds[i] = struct_dict['formation_energy']
                space_group_ds[i] = struct_dict['space_group']

    # 输出统计报告
    logger.info("\n" + "="*50)
    logger.info("Preprocessing Statistics:")
    logger.info(f"  Original structures: {len(raw_data)}")
    logger.info(f"  After filtering: {len(filtered_structures)}")
    logger.info(f"  After standardization: {len(standardized_structures)}")
    logger.info(f"  Train: {len(train_structs)}")
    logger.info(f"  Val: {len(val_structs)}")
    logger.info(f"  Test: {len(test_structs)}")
    logger.info("="*50)

    logger.info(f"Successfully saved preprocessed data to {output_path}")

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Preprocess perovskite structures")
    parser.add_argument("--input", default="data/raw/perovskites.json",
                        help="Input JSON file path")
    parser.add_argument("--output", default="data/processed/perovskites.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Configuration file path")
    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            config = config.get('data', {})
    else:
        config = {}

    preprocess_perovskites(args.input, args.output, config)
