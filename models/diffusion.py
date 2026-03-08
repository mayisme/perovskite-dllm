"""Diffusion schedule for DDPM."""
import torch
import numpy as np


class DiffusionSchedule:
    """Cosine schedule for diffusion process."""
    
    def __init__(self, T=500, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        
        # Cosine schedule (more stable than linear)
        betas = self._cosine_beta_schedule(T)
        alphas = 1 - betas
        
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
    
    def _cosine_beta_schedule(self, T, s=0.008):
        """Cosine schedule as proposed in Improved DDPM."""
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise
