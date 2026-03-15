"""简化训练脚本（用于小数据集）。"""
import torch
import torch.nn as nn
import logging
import os
import yaml
from tqdm import tqdm

from data.dataset import get_dataloader
from data.ionic_radii import IonicRadiiDatabase
from models.egnn import EGNNModel
from models.diffusion import DiffusionSchedule
from models.physics_loss import PhysicsLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_simple(config_path, checkpoint_dir):
    """简化训练流程。"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    # 数据加载（只用训练集）
    train_loader = get_dataloader(
        config['data']['processed_data_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='train',
        augment=True
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    
    # 模型
    model = EGNNModel(
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        cutoff=config['model']['cutoff_radius']
    ).to(device)
    
    # 扩散调度器
    diffusion = DiffusionSchedule(
        T=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['schedule_type'],
        device=device
    )
    
    # 物理损失
    ionic_radii_db = IonicRadiiDatabase()
    physics_loss = PhysicsLoss(ionic_radii_db)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 训练循环
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0
        total_noise_loss = 0.0
        total_physics_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # 提取数据
            frac_coords = batch['frac_coords'].to(device)
            lattice_params = batch['lattice_params'].to(device)
            atom_types = batch['atom_types'].to(device)
            band_gap = batch['band_gap'].to(device)
            formation_energy = batch['formation_energy'].to(device)
            
            B = frac_coords.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, diffusion.T, (B,), device=device)
            
            # 联合加噪
            xt_lattice, xt_coords, noise_lattice, noise_coords = diffusion.add_noise(
                lattice_params, frac_coords, t
            )
            
            # 模型预测
            noise_pred_coords, noise_pred_lattice = model(
                xt_coords, t, atom_types, xt_lattice, band_gap, formation_energy
            )
            
            # 噪声预测损失
            loss_noise_coords = nn.functional.mse_loss(noise_pred_coords, noise_coords)
            loss_noise_lattice = nn.functional.mse_loss(noise_pred_lattice, noise_lattice)
            loss_noise = loss_noise_coords + loss_noise_lattice
            
            # 预测x0
            x0_pred_coords = diffusion.predict_x0_from_noise(xt_coords, noise_pred_coords, t)
            x0_pred_lattice = diffusion.predict_x0_from_noise(xt_lattice, noise_pred_lattice, t)
            
            # 转换回实空间
            x0_pred_lattice_real = diffusion.log_space_to_lattice(x0_pred_lattice)
            
            # 物理损失
            physics_weights = config['physics_loss']
            loss_phys = physics_loss.combined_loss(
                x0_pred_coords, x0_pred_lattice_real, atom_types, physics_weights
            )
            
            # 总损失
            physics_weight = config['training']['physics_loss_weight']
            loss = loss_noise + physics_weight * loss_phys
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_noise_loss += loss_noise.item()
            total_physics_loss += loss_phys.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'noise': f"{loss_noise.item():.4f}",
                'phys': f"{loss_phys.item():.4f}"
            })
        
        # Epoch统计
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_noise = total_noise_loss / n_batches
        avg_phys = total_physics_loss / n_batches
        
        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"noise={avg_noise:.4f}, phys={avg_phys:.4f}"
        )
        
        # 保存检查点
        if (epoch + 1) % config['training']['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, final_path)
    logger.info(f"Training completed! Final model saved to {final_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    args = parser.parse_args()
    
    train_simple(args.config, args.checkpoint_dir)
