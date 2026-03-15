"""PyTorch数据集类。

从HDF5文件加载钙钛矿结构数据，支持数据增强。
"""
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerovskiteDataset(Dataset):
    """钙钛矿结构数据集。

    从HDF5文件的指定分组（train/val/test）加载数据。
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        augment: bool = True,
        augment_std: float = 0.02
    ):
        """初始化数据集。

        Args:
            h5_path: HDF5文件路径
            split: 数据分组 ('train', 'val', 'test')
            augment: 是否应用数据增强
            augment_std: 增强扰动的标准差（Å）
        """
        self.h5_path = h5_path
        self.split = split
        self.augment = augment and (split == 'train')
        self.augment_std = augment_std

        # 获取数据集大小
        with h5py.File(h5_path, "r") as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not found in {h5_path}")
            self.n_samples = len(f[split]["frac_coords"])

        logger.info(f"Loaded {split} split with {self.n_samples} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本。

        Returns:
            包含以下键的字典：
            - frac_coords: (5, 3) 分数坐标
            - lattice_params: (6,) 晶格参数 (a,b,c,α,β,γ)
            - atom_types: (5,) 原子序数
            - band_gap: 标量
            - formation_energy: 标量
        """
        with h5py.File(self.h5_path, "r") as f:
            group = f[self.split]

            frac_coords = torch.from_numpy(group["frac_coords"][idx]).float()
            lattice_params = torch.from_numpy(group["lattice_params"][idx]).float()
            atom_types = torch.from_numpy(group["atom_types"][idx]).long()
            band_gap = torch.tensor(group["band_gap"][idx]).float()
            formation_energy = torch.tensor(group["formation_energy"][idx]).float()

        # 数据增强
        if self.augment:
            frac_coords, lattice_params = self.augment_structure(
                frac_coords, lattice_params
            )

        return {
            "frac_coords": frac_coords,
            "lattice_params": lattice_params,
            "atom_types": atom_types,
            "band_gap": band_gap,
            "formation_energy": formation_energy
        }

    def augment_structure(
        self,
        coords: torch.Tensor,
        lattice_params: torch.Tensor
    ) -> tuple:
        """数据增强：对称容许的小畸变。

        Args:
            coords: (N, 3) 分数坐标
            lattice_params: (6,) 晶格参数

        Returns:
            (增强后的coords, 增强后的lattice_params)
        """
        # 分数坐标小扰动
        noise = torch.randn_like(coords) * self.augment_std
        coords_aug = coords + noise
        coords_aug = coords_aug % 1.0  # 包裹到[0,1)

        # 晶格参数小扰动（保持角度相对稳定）
        lattice_noise = torch.randn_like(lattice_params) * self.augment_std
        lattice_noise[3:] *= 0.5  # 角度扰动更小
        lattice_aug = lattice_params + lattice_noise

        # 确保晶格参数在合理范围
        lattice_aug[:3] = torch.clamp(lattice_aug[:3], min=3.0, max=15.0)  # a,b,c
        lattice_aug[3:] = torch.clamp(lattice_aug[3:], min=60.0, max=120.0)  # α,β,γ

        return coords_aug, lattice_aug


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """批处理函数。

    Args:
        batch: 样本列表

    Returns:
        批处理后的字典
    """
    return {
        "frac_coords": torch.stack([item["frac_coords"] for item in batch]),
        "lattice_params": torch.stack([item["lattice_params"] for item in batch]),
        "atom_types": torch.stack([item["atom_types"] for item in batch]),
        "band_gap": torch.stack([item["band_gap"] for item in batch]),
        "formation_energy": torch.stack([item["formation_energy"] for item in batch])
    }


def get_dataloader(
    h5_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    split: str = "train",
    augment: bool = True
):
    """创建数据加载器。

    Args:
        h5_path: HDF5文件路径
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        split: 数据分组
        augment: 是否数据增强

    Returns:
        DataLoader对象
    """
    dataset = PerovskiteDataset(h5_path, split=split, augment=augment)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
