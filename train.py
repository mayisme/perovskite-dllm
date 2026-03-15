"""训练框架。

实现扩散模型训练、检查点管理、验证和早停。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from typing import Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """扩散模型训练器。"""

    def __init__(
        self,
        model: nn.Module,
        diffusion,
        physics_loss,
        config: Dict,
        device: str = "cuda"
    ):
        """初始化训练器。"""
        self.model = model.to(device)
        self.diffusion = diffusion
        self.physics_loss = physics_loss
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 5e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 500)
        )

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 物理损失权重
        self.physics_weights = config.get('physics_loss', {
            'goldschmidt': 1.0,
            'coordination': 0.5,
            'bond_length': 0.3,
            'pauli': 1.0
        })

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练。"""
        self.model.train()
        self.optimizer.zero_grad()

        # 提取数据
        frac_coords = batch['frac_coords'].to(self.device)
        lattice_params = batch['lattice_params'].to(self.device)
        atom_types = batch['atom_types'].to(self.device)
        band_gap = batch['band_gap'].to(self.device)
        formation_energy = batch['formation_energy'].to(self.device)

        B = frac_coords.shape[0]

        # 随机采样时间步
        t = torch.randint(0, self.diffusion.T, (B,), device=self.device)

        # 联合加噪
        xt_lattice, xt_coords, noise_lattice, noise_coords = self.diffusion.add_noise(
            lattice_params, frac_coords, t
        )

        # 模型预测
        noise_pred_coords, noise_pred_lattice = self.model(
            xt_coords, t, atom_types, xt_lattice, band_gap, formation_energy
        )

        # 噪声预测损失
        loss_noise_coords = nn.functional.mse_loss(noise_pred_coords, noise_coords)
        loss_noise_lattice = nn.functional.mse_loss(noise_pred_lattice, noise_lattice)
        loss_noise = loss_noise_coords + loss_noise_lattice

        # 预测x0
        x0_pred_coords = self.diffusion.predict_x0_from_noise(
            xt_coords, noise_pred_coords, t
        )
        x0_pred_lattice = self.diffusion.predict_x0_from_noise(
            xt_lattice, noise_pred_lattice, t
        )

        # 转换回实空间
        x0_pred_lattice_real = self.diffusion.log_space_to_lattice(x0_pred_lattice)

        # 物理损失
        loss_physics = self.physics_loss.combined_loss(
            x0_pred_coords, x0_pred_lattice_real, atom_types, self.physics_weights
        )

        # 总损失
        physics_weight = self.config.get('physics_loss_weight', 0.1)
        loss = loss_noise + physics_weight * loss_physics

        # 反向传播
        loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('grad_clip', 1.0)
        )

        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_noise': loss_noise.item(),
            'loss_physics': loss_physics.item(),
            'grad_norm': grad_norm.item()
        }

    @torch.no_grad()
    def val_step(self, batch: Dict) -> Dict[str, float]:
        """验证步骤。"""
        self.model.eval()

        frac_coords = batch['frac_coords'].to(self.device)
        lattice_params = batch['lattice_params'].to(self.device)
        atom_types = batch['atom_types'].to(self.device)
        band_gap = batch['band_gap'].to(self.device)
        formation_energy = batch['formation_energy'].to(self.device)

        B = frac_coords.shape[0]
        t = torch.randint(0, self.diffusion.T, (B,), device=self.device)

        xt_lattice, xt_coords, noise_lattice, noise_coords = self.diffusion.add_noise(
            lattice_params, frac_coords, t
        )

        noise_pred_coords, noise_pred_lattice = self.model(
            xt_coords, t, atom_types, xt_lattice, band_gap, formation_energy
        )

        loss_noise_coords = nn.functional.mse_loss(noise_pred_coords, noise_coords)
        loss_noise_lattice = nn.functional.mse_loss(noise_pred_lattice, noise_lattice)
        loss_noise = loss_noise_coords + loss_noise_lattice

        return {'val_loss': loss_noise.item()}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch。"""
        total_loss = 0.0
        total_noise_loss = 0.0
        total_physics_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_noise_loss += metrics['loss_noise']
            total_physics_loss += metrics['loss_physics']

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'noise': f"{metrics['loss_noise']:.4f}",
                'physics': f"{metrics['loss_physics']:.4f}"
            })

        n_batches = len(train_loader)
        return {
            'train_loss': total_loss / n_batches,
            'train_noise_loss': total_noise_loss / n_batches,
            'train_physics_loss': total_physics_loss / n_batches
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证。"""
        total_val_loss = 0.0

        for batch in tqdm(val_loader, desc="Validation"):
            metrics = self.val_step(batch)
            total_val_loss += metrics['val_loss']

        return {'val_loss': total_val_loss / len(val_loader)}

    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点。"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, path: str):
        """加载检查点。"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str,
        use_wandb: bool = False
    ):
        """完整训练循环。"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            if epoch % self.config.get('val_frequency', 10) == 0:
                val_metrics = self.validate(val_loader)

                # 记录
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}"
                )

                if use_wandb:
                    import wandb
                    wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})

                # 早停
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0

                    # 保存最佳模型
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                    self.save_checkpoint(checkpoint_path, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # 定期保存
            if epoch % self.config.get('checkpoint_frequency', 20) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path)

            # 学习率调度
            self.scheduler.step()

        logger.info("Training completed")
