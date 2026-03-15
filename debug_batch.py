"""测试完整batch训练。"""
import torch
import yaml
from torch.utils.data import DataLoader
from data.dataset import PerovskiteDataset
from models.egnn import EGNNModel
from models.diffusion import DiffusionSchedule
from models.physics_loss import PhysicsLoss

# 加载配置
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)

# 加载数据
dataset = PerovskiteDataset('data/processed/perovskites_20k.h5', split='train')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# 初始化模型
model = EGNNModel(
    hidden_dim=config['model']['hidden_dim'],
    n_layers=config['model']['n_layers'],
    cutoff=config['model']['cutoff_radius'],
    n_atom_types=config['model']['n_atom_types']
)

diffusion = DiffusionSchedule(
    T=config['diffusion']['timesteps'],
    schedule_type=config['diffusion']['schedule_type']
)

physics_loss = PhysicsLoss()

print("Testing first batch...")
batch = next(iter(dataloader))

print("Batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
        if torch.isnan(v).any():
            print(f"    WARNING: NaN detected in {k}!")
        if torch.isinf(v).any():
            print(f"    WARNING: Inf detected in {k}!")

frac_coords = batch['frac_coords']
lattice_params = batch['lattice_params']
atom_types = batch['atom_types']
band_gap = batch['band_gap']
formation_energy = batch['formation_energy']

B = frac_coords.shape[0]
t = torch.randint(0, diffusion.T, (B,))

print(f"\nSampled timesteps: {t}")

# 测试加噪
print("\nTesting noise addition...")
xt_lattice, xt_coords, noise_lattice, noise_coords = diffusion.add_noise(
    lattice_params, frac_coords, t
)

print(f"xt_lattice stats: min={xt_lattice.min():.4f}, max={xt_lattice.max():.4f}, mean={xt_lattice.mean():.4f}")
print(f"xt_coords stats: min={xt_coords.min():.4f}, max={xt_coords.max():.4f}, mean={xt_coords.mean():.4f}")

if torch.isnan(xt_lattice).any():
    print("ERROR: NaN in xt_lattice!")
    nan_mask = torch.isnan(xt_lattice)
    print(f"  NaN positions: {nan_mask.nonzero()}")
    print(f"  Original lattice_params at NaN positions:")
    for idx in nan_mask.nonzero():
        print(f"    Sample {idx[0]}, param {idx[1]}: {lattice_params[idx[0], idx[1]]}")

if torch.isnan(xt_coords).any():
    print("ERROR: NaN in xt_coords!")

# 测试模型前向传播
print("\nTesting model forward pass...")
try:
    noise_pred_coords, noise_pred_lattice = model(
        xt_coords, t, atom_types, xt_lattice, band_gap, formation_energy
    )
    
    print(f"noise_pred_coords stats: min={noise_pred_coords.min():.4f}, max={noise_pred_coords.max():.4f}")
    print(f"noise_pred_lattice stats: min={noise_pred_lattice.min():.4f}, max={noise_pred_lattice.max():.4f}")
    
    if torch.isnan(noise_pred_coords).any():
        print("ERROR: NaN in noise_pred_coords!")
    if torch.isnan(noise_pred_lattice).any():
        print("ERROR: NaN in noise_pred_lattice!")
        
    # 测试损失计算
    print("\nTesting loss computation...")
    loss_coords = torch.nn.functional.mse_loss(noise_pred_coords, noise_coords)
    loss_lattice = torch.nn.functional.mse_loss(noise_pred_lattice, noise_lattice)
    loss = loss_coords + loss_lattice
    
    print(f"loss_coords: {loss_coords.item():.6f}")
    print(f"loss_lattice: {loss_lattice.item():.6f}")
    print(f"total loss: {loss.item():.6f}")
    
    if torch.isnan(loss):
        print("ERROR: NaN in loss!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
