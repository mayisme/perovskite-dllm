"""Equivariant Transformer for crystal structure modeling.

Based on TransVAE-CSP: Equivariant dot-product attention that preserves
geometric equivariance while capturing long-range dependencies.
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class EquivariantDotProductAttention(nn.Module):
    """Equivariant dot-product attention mechanism.
    
    Maintains E(3) equivariance by:
    1. Using scalar features for attention weights
    2. Applying attention to both scalar and vector features separately
    3. Preserving geometric structure through coordinate transformations
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for scalar features
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Coordinate attention (for equivariant features)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        h: torch.Tensor,  # (B, N, hidden_dim) scalar features
        coords: torch.Tensor,  # (B, N, 3) coordinates
        mask: Optional[torch.Tensor] = None  # (B, N) attention mask
    ):
        """Forward pass with equivariant attention.
        
        Args:
            h: Scalar node features (B, N, hidden_dim)
            coords: Node coordinates (B, N, 3)
            mask: Optional attention mask (B, N)
            
        Returns:
            h_out: Updated scalar features (B, N, hidden_dim)
            coords_out: Updated coordinates (B, N, 3)
        """
        B, N, _ = h.shape
        
        # Project to Q, K, V
        Q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, N, N)
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to scalar features
        h_attn = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        h_attn = h_attn.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        h_out = self.out_proj(h_attn)
        
        # Equivariant coordinate update
        # Compute coordinate attention weights (scalar, so equivariant)
        coord_weights = self.coord_mlp(h)  # (B, N, 1)
        coord_weights = torch.softmax(coord_weights, dim=1)  # Normalize over nodes
        
        # Weighted average of relative positions (maintains equivariance)
        # For each node, compute attention-weighted displacement
        coords_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        
        # Average attention weights across heads
        attn_weights_mean = attn_weights.mean(dim=1)  # (B, N, N)
        
        # Apply attention to coordinate differences
        coords_update = torch.einsum('bnm,bnmd->bnd', attn_weights_mean, coords_diff)
        coords_update = coords_update * coord_weights  # Scale by learned weights
        
        coords_out = coords + coords_update
        
        return h_out, coords_out


class EquivariantTransformerLayer(nn.Module):
    """Single layer of Equivariant Transformer.
    
    Combines:
    1. Equivariant dot-product attention
    2. Feed-forward network
    3. Layer normalization
    4. Residual connections
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim or hidden_dim * 4
        
        # Equivariant attention
        self.attention = EquivariantDotProductAttention(
            hidden_dim, num_heads, dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """Forward pass.
        
        Args:
            h: Scalar features (B, N, hidden_dim)
            coords: Coordinates (B, N, 3)
            mask: Attention mask (B, N)
            
        Returns:
            h_out: Updated features (B, N, hidden_dim)
            coords_out: Updated coordinates (B, N, 3)
        """
        # Attention with residual
        h_attn, coords_attn = self.attention(h, coords, mask)
        h = self.norm1(h + h_attn)
        coords = coords_attn  # Coordinates don't use residual (to avoid drift)
        
        # Feed-forward with residual
        h_ff = self.ff(h)
        h = self.norm2(h + h_ff)
        
        return h, coords


class EquivariantTransformer(nn.Module):
    """Stack of Equivariant Transformer layers.
    
    Designed to be used after EGNN layers for global dependency modeling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(
                hidden_dim, num_heads, ff_dim, dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """Forward pass through all layers.
        
        Args:
            h: Scalar features (B, N, hidden_dim)
            coords: Coordinates (B, N, 3)
            mask: Attention mask (B, N)
            
        Returns:
            h_out: Updated features (B, N, hidden_dim)
            coords_out: Updated coordinates (B, N, 3)
        """
        for layer in self.layers:
            h, coords = layer(h, coords, mask)
        
        return h, coords
