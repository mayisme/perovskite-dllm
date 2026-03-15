"""示例工作流程。

演示从数据预处理到生成的完整流程。
"""
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_workflow():
    """示例工作流程。"""
    
    logger.info("="*60)
    logger.info("ABO₃钙钛矿扩散生成系统 - 示例工作流程")
    logger.info("="*60)
    
    # 1. 测试离子半径数据库
    logger.info("\n1. 测试离子半径数据库...")
    from data.ionic_radii import IonicRadiiDatabase
    
    db = IonicRadiiDatabase()
    t = db.compute_goldschmidt_tolerance('Ba', 'Ti')
    logger.info(f"   BaTiO3 Goldschmidt容忍因子: {t:.4f}")
    
    # 2. 测试扩散调度器
    logger.info("\n2. 测试扩散调度器...")
    from models.diffusion import DiffusionSchedule
    
    diffusion = DiffusionSchedule(T=100, schedule_type='cosine', device='cpu')
    logger.info(f"   时间步数: {diffusion.T}")
    logger.info(f"   alpha_bar[0]: {diffusion.alpha_bar[0]:.4f}")
    logger.info(f"   alpha_bar[-1]: {diffusion.alpha_bar[-1]:.6f}")
    
    # 3. 测试EGNN模型
    logger.info("\n3. 测试EGNN模型...")
    from models.egnn import EGNNModel
    
    model = EGNNModel(hidden_dim=64, n_layers=2, cutoff=6.0)
    
    # 创建示例输入
    B, N = 2, 5
    x = torch.rand(B, N, 3)
    t = torch.randint(0, 100, (B,))
    atom_types = torch.tensor([[56, 22, 8, 8, 8]] * B)  # Ba, Ti, O, O, O
    lattice_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * B)
    band_gap = torch.tensor([3.0] * B)
    formation_energy = torch.tensor([-5.0] * B)
    
    noise_coords, noise_lattice = model(
        x, t, atom_types, lattice_params, band_gap, formation_energy
    )
    
    logger.info(f"   输入坐标形状: {x.shape}")
    logger.info(f"   输出噪声坐标形状: {noise_coords.shape}")
    logger.info(f"   输出噪声晶格形状: {noise_lattice.shape}")
    
    # 4. 测试物理损失
    logger.info("\n4. 测试物理损失...")
    from models.physics_loss import PhysicsLoss
    
    physics_loss = PhysicsLoss(db)
    
    coords = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0],
                            [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]])
    lattice_params_test = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]])
    
    loss_pauli = physics_loss.pauli_repulsion_loss(coords, lattice_params_test)
    logger.info(f"   Pauli排斥损失: {loss_pauli:.4f}")
    
    weights = {
        'goldschmidt': 1.0,
        'coordination': 0.5,
        'bond_length': 0.3,
        'pauli': 1.0
    }
    
    total_loss = physics_loss.combined_loss(
        coords, lattice_params_test, atom_types[:1], weights
    )
    logger.info(f"   组合物理损失: {total_loss:.4f}")
    
    # 5. 测试训练步骤（不实际训练）
    logger.info("\n5. 测试训练步骤...")
    from train import DiffusionTrainer
    
    config = {
        'lr': 5e-5,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'epochs': 500,
        'physics_loss_weight': 0.1,
        'physics_loss': weights
    }
    
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        physics_loss=physics_loss,
        config=config,
        device='cpu'
    )
    
    # 创建示例批次
    batch = {
        'frac_coords': x,
        'lattice_params': lattice_params,
        'atom_types': atom_types,
        'band_gap': band_gap,
        'formation_energy': formation_energy
    }
    
    metrics = trainer.train_step(batch)
    logger.info(f"   训练损失: {metrics['loss']:.4f}")
    logger.info(f"   噪声损失: {metrics['loss_noise']:.4f}")
    logger.info(f"   物理损失: {metrics['loss_physics']:.4f}")
    
    # 6. 测试生成（简化版）
    logger.info("\n6. 测试生成...")
    from generate import PerovskiteGenerator
    
    gen_config = {
        'sampling_steps': 50,
        'guidance_scale': 3.0
    }
    
    generator = PerovskiteGenerator(model, diffusion, gen_config, 'cpu')
    
    logger.info("   生成器已初始化")
    logger.info("   （实际生成需要训练好的模型）")
    
    logger.info("\n" + "="*60)
    logger.info("示例工作流程完成！")
    logger.info("="*60)
    
    logger.info("\n下一步：")
    logger.info("1. 准备数据：python main.py preprocess --input ... --output ...")
    logger.info("2. 训练模型：python main.py train --config configs/base.yaml")
    logger.info("3. 生成结构：python main.py generate --checkpoint ... --band-gap 3.0")
    logger.info("4. 验证结构：python main.py validate --input-dir outputs/generated")


if __name__ == '__main__':
    example_workflow()
