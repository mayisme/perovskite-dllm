"""Simple MLP-based diffusion model for perovskite generation."""
import torch
import torch.nn as nn


class SimpleDiffusionModel(nn.Module):
    """MLP-based model for coordinate denoising."""
    
    def __init__(self, n_atoms=5, hidden_dim=256, time_dim=64):
        super().__init__()
        self.n_atoms = n_atoms
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Property embedding (bandgap)
        self.prop_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU()
        )
        
        # Main network
        coord_dim = n_atoms * 3
        self.net = nn.Sequential(
            nn.Linear(coord_dim + time_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coord_dim)
        )
    
    def forward(self, x, t, bandgap):
        """
        x: (B, N, 3) noisy coordinates
        t: (B,) timestep
        bandgap: (B,) target bandgap
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        p_emb = self.prop_mlp(bandgap.unsqueeze(-1).float())
        
        h = torch.cat([x_flat, t_emb, p_emb], dim=-1)
        noise_pred = self.net(h)
        
        return noise_pred.reshape(B, self.n_atoms, 3)
