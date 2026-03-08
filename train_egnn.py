"""Training with EGNN and physics constraints."""
import torch
import torch.nn.functional as F
import json
import numpy as np
from pymatgen.core import Structure
from models.diffusion import DiffusionSchedule
from models.egnn import EGNNModel
from models.physics_loss import combined_physics_loss

# Load data
with open("data/raw/perovskites.json") as f:
    data = json.load(f)

coords_list = []
bandgaps = []

for item in data[:500]:  # Use 500 structures
    struct = Structure.from_dict(item["structure"])
    if len(struct) < 5:
        continue
    coords = np.array([site.frac_coords for site in struct[:5]])
    coords_list.append(torch.tensor(coords, dtype=torch.float32))
    bandgaps.append(item["band_gap"] or 0.0)

coords_tensor = torch.stack(coords_list)
bandgaps_tensor = torch.tensor(bandgaps, dtype=torch.float32)

print(f"Training data: {len(coords_tensor)} structures")

# Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = EGNNModel(n_atoms=5, hidden_dim=128, n_layers=3).to(device)
schedule = DiffusionSchedule(T=100, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"Device: {device}")
print(f"Model: EGNN with {sum(p.numel() for p in model.parameters()):,} parameters")

# Training
epochs = 50  # Reduced for demo
batch_size = 16

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_noise_loss = 0
    total_physics_loss = 0
    
    indices = torch.randperm(len(coords_tensor))
    
    for i in range(0, len(coords_tensor), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_coords = coords_tensor[batch_idx].to(device)
        batch_bandgaps = bandgaps_tensor[batch_idx].to(device)
        
        # Random timestep
        t = torch.randint(0, schedule.T, (len(batch_coords),), device=device)
        
        # Add noise
        noise = torch.randn_like(batch_coords)
        noisy_coords, _ = schedule.add_noise(batch_coords, t, noise)
        
        # Predict noise
        noise_pred = model(noisy_coords, t, batch_bandgaps)
        
        # Noise prediction loss
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Physics loss (on denoised coords)
        with torch.no_grad():
            alpha_bar_t = schedule.alpha_bar[t].view(-1, 1, 1)
            pred_x0 = (noisy_coords - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        physics_loss = combined_physics_loss(pred_x0, batch_bandgaps, batch_bandgaps)
        
        # Combined loss
        loss = noise_loss + 0.1 * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_noise_loss += noise_loss.item()
        total_physics_loss += physics_loss.item()
    
    if (epoch + 1) % 20 == 0:
        n_batches = len(coords_tensor) // batch_size
        print(f"Epoch {epoch+1:3d}: "
              f"loss={total_loss/n_batches:.4f} "
              f"(noise={total_noise_loss/n_batches:.4f}, "
              f"physics={total_physics_loss/n_batches:.4f})")

# Save
torch.save(model.state_dict(), "checkpoints/egnn_model.pt")
print("\nEGNN model saved to checkpoints/egnn_model.pt")
