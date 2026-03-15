"""简化生成脚本。"""
import torch
import logging
import os
import yaml
from pymatgen.core import Structure, Lattice

from models.egnn import EGNNModel
from models.diffusion import DiffusionSchedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_simple(config_path, checkpoint_path, output_dir, num_samples=10):
    """简化生成流程。"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model = EGNNModel(
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        cutoff=config['model']['cutoff_radius']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # 扩散调度器
    diffusion = DiffusionSchedule(
        T=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )
    
    # 原子类型（BaTiO3）
    atom_types = torch.tensor([[56, 22, 8, 8, 8]] * num_samples)  # Ba, Ti, O, O, O
    band_gap = torch.full((num_samples,), 3.0)
    formation_energy = torch.full((num_samples,), -5.0)
    
    # 初始化噪声
    x = torch.randn(num_samples, 5, 3)
    lattice_log = torch.randn(num_samples, 6)
    
    logger.info(f"Generating {num_samples} structures...")
    
    # DDPM采样
    with torch.no_grad():
        for t in range(diffusion.T - 1, -1, -1):
            if t % 20 == 0:
                logger.info(f"Sampling step {diffusion.T - t}/{diffusion.T}")
            
            t_batch = torch.full((num_samples,), t, dtype=torch.long)
            
            # 预测噪声
            noise_pred_coords, noise_pred_lattice = model(
                x, t_batch, atom_types, lattice_log, band_gap, formation_energy
            )
            
            # DDPM步骤
            lattice_log, x = diffusion.ddpm_sample_step(
                lattice_log, x, noise_pred_lattice, noise_pred_coords, t
            )
    
    # 转换为Structure对象
    lattice_params = diffusion.log_space_to_lattice(lattice_log)
    x = diffusion.wrap_frac_coords(x)
    
    logger.info("Converting to Structure objects...")
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    
    for i in range(num_samples):
        try:
            # 转换晶格参数
            a, b, c = lattice_params[i, :3].numpy()
            alpha, beta, gamma = lattice_params[i, 3:].numpy()
            
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            
            # 创建Structure
            species = [int(z) for z in atom_types[i].numpy()]
            coords = x[i].numpy()
            
            structure = Structure(lattice, species, coords)
            
            # 保存CIF
            output_path = os.path.join(output_dir, f"generated_{i}.cif")
            structure.to(filename=output_path)
            success_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to create structure {i}: {e}")
    
    logger.info(f"Successfully generated {success_count}/{num_samples} structures")
    logger.info(f"Structures saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='outputs/generated')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    
    generate_simple(args.config, args.checkpoint, args.output_dir, args.num_samples)
