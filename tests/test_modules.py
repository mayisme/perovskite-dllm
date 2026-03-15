"""模块测试。"""
import sys
sys.path.insert(0, '/Users/xiaoyf/Documents/Python/dllm-perovskite')

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ionic_radii():
    """测试离子半径数据库。"""
    from data.ionic_radii import IonicRadiiDatabase
    
    db = IonicRadiiDatabase()
    
    # 测试查询
    r_ba = db.get_radius('Ba', 2, 12)
    r_ti = db.get_radius('Ti', 4, 6)
    r_o = db.get_radius('O', -2, 6)
    
    logger.info(f"Ba2+ radius: {r_ba}")
    logger.info(f"Ti4+ radius: {r_ti}")
    logger.info(f"O2- radius: {r_o}")
    
    # 测试Goldschmidt计算
    t = db.compute_goldschmidt_tolerance('Ba', 'Ti')
    logger.info(f"BaTiO3 Goldschmidt tolerance: {t}")
    
    assert r_ba is not None
    assert r_ti is not None
    assert r_o is not None
    assert 0.8 <= t <= 1.0
    
    logger.info("✓ IonicRadiiDatabase test passed")


def test_diffusion_schedule():
    """测试扩散调度器。"""
    from models.diffusion import DiffusionSchedule
    
    diffusion = DiffusionSchedule(T=100, schedule_type='cosine', device='cpu')
    
    # 测试alpha_bar单调递减
    for t in range(diffusion.T - 1):
        assert diffusion.alpha_bar[t] > diffusion.alpha_bar[t + 1]
    
    # 测试边界条件
    assert diffusion.alpha_bar[0] > 0.99
    assert diffusion.alpha_bar[-1] < 0.01
    
    # 测试对数空间转换
    lattice_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]])
    log_params = diffusion.lattice_to_log_space(lattice_params)
    recovered = diffusion.log_space_to_lattice(log_params)
    
    assert torch.allclose(lattice_params, recovered, atol=1e-5)
    
    # 测试加噪
    coords = torch.rand(2, 5, 3)
    lattice = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * 2)
    t = torch.tensor([10, 20])
    
    xt_lattice, xt_coords, noise_lattice, noise_coords = diffusion.add_noise(
        lattice, coords, t
    )
    
    assert xt_lattice.shape == (2, 6)
    assert xt_coords.shape == (2, 5, 3)
    
    logger.info("✓ DiffusionSchedule test passed")


def test_egnn():
    """测试EGNN模型。"""
    from models.egnn import EGNNModel
    
    model = EGNNModel(hidden_dim=64, n_layers=2, cutoff=6.0)
    
    # 测试前向传播
    B, N = 2, 5
    x = torch.rand(B, N, 3)
    t = torch.randint(0, 100, (B,))
    atom_types = torch.randint(1, 100, (B, N))
    lattice_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * B)
    band_gap = torch.rand(B)
    formation_energy = torch.rand(B)
    
    noise_coords, noise_lattice = model(
        x, t, atom_types, lattice_params, band_gap, formation_energy
    )
    
    assert noise_coords.shape == (B, N, 3)
    assert noise_lattice.shape == (B, 6)
    
    logger.info("✓ EGNNModel test passed")


def test_physics_loss():
    """测试物理损失。"""
    from models.physics_loss import PhysicsLoss
    from data.ionic_radii import IonicRadiiDatabase
    
    db = IonicRadiiDatabase()
    physics_loss = PhysicsLoss(db)
    
    # 测试Pauli排斥
    coords = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0],
                            [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]])
    lattice_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]])
    
    loss = physics_loss.pauli_repulsion_loss(coords, lattice_params, min_dist=1.5)
    
    assert loss >= 0
    
    logger.info("✓ PhysicsLoss test passed")


def test_dataset():
    """测试数据集（需要HDF5文件）。"""
    # 跳过，因为需要实际数据
    logger.info("⊘ Dataset test skipped (requires data)")


def run_all_tests():
    """运行所有测试。"""
    logger.info("="*50)
    logger.info("Running module tests...")
    logger.info("="*50)
    
    try:
        test_ionic_radii()
        test_diffusion_schedule()
        test_egnn()
        test_physics_loss()
        test_dataset()
        
        logger.info("="*50)
        logger.info("All tests passed! ✓")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    run_all_tests()
