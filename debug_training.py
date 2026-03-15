"""调试训练中的NaN问题。"""
import torch
import yaml
from data.dataset import PerovskiteDataset
from models.egnn import EGNNModel
from models.diffusion import DiffusionSchedule
from models.physics_loss import PhysicsLoss

# 加载配置
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)

# 加载数据
dataset = PerovskiteDataset('data/processed/perovskites_20k.h5', split='train')
batch = dataset[0]

# 转换为batch
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].unsqueeze(0)

print("Batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")

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

print("\nTesting forward pass...")
frac_coords = batch['frac_coords']
lattice_params = batch['lattice_params']
atom_types = batch['atom_types']
band_gap = batch['band_gap']
formation_energy = batch['formation_energy']

print(f"Input lattice_params: {lattice_params}")
print(f"Input frac_coords range: [{frac_coords.min():.4f}, {frac_coords.max():.4f}]")

# 测试对数空间转换
log_lattice = diffusion.lattice_to_log_space(lattice_params)
print(f"Log lattice: {log_lattice}")
recovered_lattice = diffusion.log_space_to_lattice(log_lattice)
print(f"Recovered lattice: {recovered_lattice}")
print(f"Conversion error: {(lattice_params - recovered_lattice).abs().max():.6f}")

# 测试加噪
t = torch.tensor([100])
print(f"\nTesting noise addition at t={t.item()}...")
xt_lattice, xt_coords, noise_lattice, noise_coords = diffusion.add_noise(
    lattice_params, frac_coords, t
)

print(f"xt_lattice: {xt_lattice}")
print(f"noise_lattice: {noise_lattice}")
print(f"xt_coords range: [{xt_coords.min():.4f}, {xt_coords.max():.4f}]")
print(f"noise_coords range: [{noise_coords.min():.4f}, {noise_coords.max():.4f}]")

# 检查是否有NaN
if torch.isnan(xt_lattice).any():
    print("WARNING: NaN in xt_lattice!")
if torch.isnan(xt_coords).any():
    print("WARNING: NaN in xt_coords!")

# 测试模型前向传播
print("\nTesting model forward pass...")
try:
    noise_pred_coords, noise_pred_lattice = model(
        xt_coords, t, atom_types, xt_lattice, band_gap, formation_energy
    )
    print(f"noise_pred_coords range: [{noise_pred_coords.min():.4f}, {noise_pred_coords.max():.4f}]")
    print(f"noise_pred_lattice: {noise_pred_lattice}")
    
    if torch.isnan(noise_pred_coords).any():
        print("WARNING: NaN in noise_pred_coords!")
    if torch.isnan(noise_pred_lattice).any():
        print("WARNING: NaN in noise_pred_lattice!")
        
except Exception as e:
    print(f"ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
