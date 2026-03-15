"""Hybrid model: EGNN + Equivariant Transformer for crystal diffusion.

Combines:
1. Fast EGNN layers for local geometric feature extraction
2. Equivariant Transformer for global dependency modeling
"""
import torch
import torch.nn as nn
from models.fast_egnn import FastEGNNLayer
from models.equivariant_transformer import EquivariantTransformer


class HybridEGNNTransformer(nn.Module):
    """Hybrid architecture combining EGNN and Equivariant Transformer.
    
    Architecture:
        Input → EGNN layers (local) → Equivariant Transformer (global) → Output
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_egnn_layers: int = 3,
        n_transformer_layers: int = 2,
        n_atom_types: int = 100,
        cutoff: float = 6.0,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_egnn_layers = n_egnn_layers
        self.n_transformer_layers = n_transformer_layers
        
        # Embedding layers
        self.node_emb = nn.Embedding(n_atom_types, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bandgap_emb = nn.Linear(1, hidden_dim)
        self.formation_energy_emb = nn.Linear(1, hidden_dim)
        
        # EGNN layers for local geometry
        self.egnn_layers = nn.ModuleList([
            FastEGNNLayer(hidden_dim, cutoff) for _ in range(n_egnn_layers)
        ])
        
        # Equivariant Transformer for global dependencies
        self.transformer = EquivariantTransformer(
            hidden_dim=hidden_dim,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            ff_dim=hidden_dim * 4,
            dropout=dropout
        )
        
        # Output layers
        self.out_coords = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Lattice parameter prediction (6维：a,b,c,α,β,γ)
        self.out_lattice = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        lattice_params: torch.Tensor,
        band_gap: torch.Tensor,
        formation_energy: torch.Tensor
    ):
        """Forward pass.
        
        Args:
            x: (B, N, 3) fractional coordinates
            t: (B,) timesteps
            atom_types: (B, N) atom types
            lattice_params: (B, 6) lattice parameters
            band_gap: (B,) band gap
            formation_energy: (B,) formation energy
            
        Returns:
            noise_pred_coords: (B, N, 3) predicted noise for coordinates
            noise_pred_lattice: (B, 6) predicted noise for lattice
        """
        B, N = x.shape[:2]
        device = x.device
        
        # Embeddings
        h = self.node_emb(atom_types)  # (B, N, hidden_dim)
        t_emb = self.time_emb(t.view(B, 1, 1).float() / 1000.0)
        bg_emb = self.bandgap_emb(band_gap.view(B, 1, 1))
        fe_emb = self.formation_energy_emb(formation_energy.view(B, 1, 1))
        
        h = h + t_emb + bg_emb + fe_emb
        
        # Flatten batch for EGNN
        h_flat = h.reshape(B * N, -1)
        x_flat = x.reshape(B * N, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)
        
        # Convert fractional to Cartesian coordinates
        lattice_matrices = self._params_to_matrix(lattice_params)
        x_cart_flat = torch.zeros_like(x_flat)
        for b in range(B):
            mask = batch_idx == b
            x_cart_flat[mask] = torch.matmul(x_flat[mask], lattice_matrices[b])
        
        # Build edges for EGNN
        # 优先使用 torch_cluster，否则使用向量化实现
        try:
            edge_index, edge_dist = self.egnn_layers[0].build_edges_fast(
                x_cart_flat, batch_idx, max_neighbors=32
            )
        except RuntimeError:
            # Fallback to vectorized implementation
            edge_index, edge_dist = self.egnn_layers[0].build_edges_vectorized(
                x_cart_flat, batch_idx, max_neighbors=32
            )
        
        # EGNN layers (local geometric features)
        for layer in self.egnn_layers:
            h_flat, x_cart_flat = layer(h_flat, x_cart_flat, edge_index, edge_dist)
        
        # Reshape back to batch format for Transformer
        h = h_flat.reshape(B, N, -1)
        x_cart = x_cart_flat.reshape(B, N, 3)
        
        # Create attention mask (all atoms are valid)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # Equivariant Transformer (global dependencies)
        h, x_cart = self.transformer(h, x_cart, mask)
        
        # Flatten again for output
        h_flat = h.reshape(B * N, -1)
        
        # Global pooling for lattice prediction
        h_global = h.mean(dim=1)  # (B, hidden_dim)
        
        # Predict noise
        pred_noise_coords = self.out_coords(h_flat).reshape(B, N, 3)
        pred_noise_lattice = self.out_lattice(h_global)
        
        return pred_noise_coords, pred_noise_lattice
    
    def _params_to_matrix(self, lattice_params: torch.Tensor):
        """Convert lattice parameters to matrix form.
        
        Args:
            lattice_params: (B, 6) [a, b, c, alpha, beta, gamma]
            
        Returns:
            lattice_matrices: (B, 3, 3)
        """
        a = lattice_params[:, 0]
        b = lattice_params[:, 1]
        c = lattice_params[:, 2]
        alpha = lattice_params[:, 3] * torch.pi / 180.0
        beta = lattice_params[:, 4] * torch.pi / 180.0
        gamma = lattice_params[:, 5] * torch.pi / 180.0
        
        # Compute lattice vectors
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        
        # 数值稳定性：避免除以接近0的sin_gamma
        sin_gamma = torch.clamp(sin_gamma, min=1e-6)
        
        # Volume factor
        vol_factor = torch.sqrt(
            torch.clamp(
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                + 2 * cos_alpha * cos_beta * cos_gamma,
                min=1e-6  # 避免负数或0
            )
        )
        
        # Lattice matrix (fractional to Cartesian)
        B = lattice_params.shape[0]
        lattice_matrices = torch.zeros(B, 3, 3, device=lattice_params.device)
        
        lattice_matrices[:, 0, 0] = a
        lattice_matrices[:, 0, 1] = b * cos_gamma
        lattice_matrices[:, 0, 2] = c * cos_beta
        lattice_matrices[:, 1, 1] = b * sin_gamma
        lattice_matrices[:, 1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        lattice_matrices[:, 2, 2] = c * vol_factor / sin_gamma
        
        return lattice_matrices
