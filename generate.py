"""Generate perovskite structures using diffusion model."""
import torch
from models.diffusion import DiffusionSchedule
from models.simple_model import SimpleDiffusionModel

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model (untrained for now, just demo the generation process)
model = SimpleDiffusionModel(n_atoms=5).to(device)
schedule = DiffusionSchedule(T=100)

@torch.no_grad()
def generate(target_bandgap=2.0, num_samples=5, steps=100):
    """Generate structures from noise."""
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(num_samples, 5, 3).to(device)
    bandgap = torch.full((num_samples,), target_bandgap).to(device)
    
    print(f"Generating {num_samples} structures with bandgap ~{target_bandgap} eV")
    print("Denoising steps:")
    
    # Reverse diffusion
    for t in reversed(range(steps)):
        t_tensor = torch.full((num_samples,), t, device=device)
        
        # Predict noise
        noise_pred = model(x_t, t_tensor, bandgap)
        
        # Denoise step
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bar[t]
        
        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        
        # Add noise (except last step)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt((1 - alpha_t) * (1 - alpha_bar_t) / (1 - schedule.alpha_bar[t-1]))
            x_t += sigma_t * noise
        
        if t % 20 == 0:
            print(f"  Step {t:3d}: mean={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")
    
    return x_t.cpu()

if __name__ == "__main__":
    structures = generate(target_bandgap=2.0, num_samples=3, steps=100)
    
    print(f"\nGenerated {len(structures)} structures:")
    for i, coords in enumerate(structures):
        print(f"  Structure {i}: shape={coords.shape}, range=[{coords.min():.2f}, {coords.max():.2f}]")
