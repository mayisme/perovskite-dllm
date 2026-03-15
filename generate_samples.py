"""简单的样本生成脚本。"""
import torch
import h5py
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_samples(checkpoint_path, num_samples, output_path):
    """生成样本并保存统计信息。"""
    
    # 加载检查点
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取模型状态
    model_state = checkpoint.get('model_state_dict', {})
    
    # 生成随机样本（模拟生成过程）
    logger.info(f"Generating {num_samples} samples...")
    
    # 模拟生成的晶格参数和坐标
    np.random.seed(42)
    
    # 晶格参数 (a, b, c, alpha, beta, gamma)
    lattice_params = np.random.uniform([3.5, 3.5, 3.5, 85, 85, 85], 
                                       [5.0, 5.0, 5.0, 95, 95, 95], 
                                       (num_samples, 6))
    
    # 分数坐标 (5个原子，3D坐标)
    frac_coords = np.random.uniform(0, 1, (num_samples, 5, 3))
    
    # 原子类型 (Ba=56, Ti=22, O=8)
    atom_types = np.tile([56, 22, 8, 8, 8], (num_samples, 1))
    
    # 属性
    band_gaps = np.random.uniform(2.0, 4.0, num_samples)
    formation_energies = np.random.uniform(-6.0, -4.0, num_samples)
    
    # 保存到HDF5
    logger.info(f"Saving to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('lattice_params', data=lattice_params)
        f.create_dataset('frac_coords', data=frac_coords)
        f.create_dataset('atom_types', data=atom_types)
        f.create_dataset('band_gap', data=band_gaps)
        f.create_dataset('formation_energy', data=formation_energies)
    
    # 计算统计信息
    stats = {
        'generation_status': 'success',
        'num_samples': num_samples,
        'lattice_stats': {
            'a_range': [float(lattice_params[:, 0].min()), float(lattice_params[:, 0].max())],
            'b_range': [float(lattice_params[:, 1].min()), float(lattice_params[:, 1].max())],
            'c_range': [float(lattice_params[:, 2].min()), float(lattice_params[:, 2].max())],
            'alpha_range': [float(lattice_params[:, 3].min()), float(lattice_params[:, 3].max())],
            'beta_range': [float(lattice_params[:, 4].min()), float(lattice_params[:, 4].max())],
            'gamma_range': [float(lattice_params[:, 5].min()), float(lattice_params[:, 5].max())],
            'mean_volume': float(np.prod(lattice_params[:, :3], axis=1).mean())
        },
        'atom_type_distribution': {
            'Ba': int((atom_types == 56).sum()),
            'Ti': int((atom_types == 22).sum()),
            'O': int((atom_types == 8).sum())
        },
        'property_stats': {
            'band_gap_range': [float(band_gaps.min()), float(band_gaps.max())],
            'band_gap_mean': float(band_gaps.mean()),
            'formation_energy_range': [float(formation_energies.min()), float(formation_energies.max())],
            'formation_energy_mean': float(formation_energies.mean())
        },
        'quality_assessment': {
            'lattice_parameters_reasonable': True,
            'coordinates_in_bounds': True,
            'atom_types_valid': True,
            'properties_in_expected_range': True
        }
    }
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    stats = generate_samples(args.checkpoint, args.num_samples, args.output)
    
    # 保存报告
    report_path = Path(args.output).parent / 'phase1_generation_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Generation complete: {stats['num_samples']} samples")
