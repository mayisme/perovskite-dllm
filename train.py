"""Minimal training script."""
import torch
import torch.nn.functional as F
import json
import numpy as np
from pymatgen.core import Structure
from models.diffusion import DiffusionSchedule
from models.simple_model import SimpleDiffusionModel

# Load and preprocess data
with open("data/raw/perovskites.json") as f:
    data = json.load(f)

coords_list = []
bandgaps = []

for item in data[:50]:  # Use first 50 for quick training
    struct = Structure.from_dict(item["structure"])
    if len(struct) < 5:
        continue
    coords = torch.tensor([site.frac_coords for site in struct[:5]], dtype=torch.float32)
    coords_list.append(coords)
    bandgaps.append(item["band_gap"] or 0.0)

coords_tensor = torch.stack(coords_list)
bandgaps_tensor = torch.tensor(bandgaps, dtype=torch.float32)

print(f"Training data: {len(coords_tensor)} structures")
print(f"Coords shape: {coords_tensor.shape}")

# Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SimpleDiffusionModel(n_atoms=5).to(device)
schedule = DiffusionSchedule(T=100, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
epochs = 100
batch_size = 8

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i in range(0, len(coords_tensor), batch_size):
        batch_coords = coords_tensor[i:i+batch_size].to(device)
        batch_bandgaps = bandgaps_tensor[i:i+batch_size].to(device)
        
        # Random timestep
        t = torch.randint(0, schedule.T, (len(batch_coords),), device=device)
        
        # Add noise
        noise = torch.randn_like(batch_coords)
        noisy_coords, _ = schedule.add_noise(batch_coords, t, noise)
        
        # Predict noise
        noise_pred = model(noisy_coords, t, batch_bandgaps)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / (len(coords_tensor) // batch_size)
        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), "checkpoints/model.pt")
print("\nModel saved to checkpoints/model.pt")
