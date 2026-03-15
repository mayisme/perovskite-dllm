"""检查HDF5数据文件内容。"""
import h5py
import numpy as np

h5_path = "data/processed/perovskites_v2.h5"

with h5py.File(h5_path, "r") as f:
    print("HDF5文件结构:")
    print(f"  分组: {list(f.keys())}")

    for split in ['train', 'val', 'test']:
        if split in f:
            group = f[split]
            print(f"\n{split}集:")
            print(f"  数据集: {list(group.keys())}")
            print(f"  样本数: {len(group['frac_coords'])}")
            print(f"  frac_coords shape: {group['frac_coords'].shape}")
            print(f"  lattice_params shape: {group['lattice_params'].shape}")
            print(f"  atom_types shape: {group['atom_types'].shape}")
            print(f"  n_atoms shape: {group['n_atoms'].shape}")

            # 检查原子数分布
            n_atoms = group['n_atoms'][:]
            print(f"  原子数范围: {n_atoms.min()} - {n_atoms.max()}")
            print(f"  原子数分布: {np.bincount(n_atoms)}")

            # 检查第一个样本
            print(f"  第一个样本:")
            print(f"    n_atoms: {n_atoms[0]}")
            print(f"    lattice_params: {group['lattice_params'][0]}")
            print(f"    atom_types[:n_atoms]: {group['atom_types'][0, :n_atoms[0]]}")
