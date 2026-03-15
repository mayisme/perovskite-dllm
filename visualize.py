"""可视化模块。

提供3D结构可视化和训练指标可视化。
"""
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
import logging

logger = logging.getLogger(__name__)


def visualize_structure(structure: Structure, output_path: str = None):
    """可视化3D晶体结构。

    Args:
        structure: pymatgen Structure对象
        output_path: 输出路径（可选）
    """
    try:
        from pymatgen.vis.structure_vtk import StructureVis
        
        vis = StructureVis()
        vis.set_structure(structure)
        
        if output_path:
            vis.write_image(output_path)
            logger.info(f"Saved structure visualization to {output_path}")
        else:
            vis.show()
            
    except ImportError:
        logger.warning("pymatgen.vis not available, using simple plot")
        
        # 简单的2D投影
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        coords = structure.cart_coords
        species = [site.specie.symbol for site in structure]
        
        for i, (coord, spec) in enumerate(zip(coords, species)):
            ax.scatter(coord[0], coord[1], coord[2], s=100, label=spec)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved structure plot to {output_path}")
        else:
            plt.show()


def plot_training_curves(train_losses, val_losses, output_path: str = None):
    """绘制训练曲线。

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        output_path: 输出路径（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved training curves to {output_path}")
    else:
        plt.show()
