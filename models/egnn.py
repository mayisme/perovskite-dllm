"""E(n)-Equivariant Graph Neural Network layer."""
import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant message passing layer."""
    
    def __init__(self, hidden_dim=64, edge_dim=0):
        super().__init__()
        
        # Edge model: h_i + h_j + dist
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node model: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate model: updates positions
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, h, x, edge_index):
        """
        h: (N, hidden_dim) node features
        x: (N, 3) coordinates
        edge_index: (2, E) edge connectivity
        """
        row, col = edge_index
        
        # Compute edge features
        rel_pos = x[row] - x[col]  # (E, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (E, 1)
        
        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        edge_msg = self.edge_mlp(edge_feat)  # (E, hidden_dim)
        
        # Aggregate messages
        agg_msg = torch.zeros_like(h)
        agg_msg.index_add_(0, col, edge_msg)
        
        # Update node features
        h_new = self.node_mlp(torch.cat([h, agg_msg], dim=-1))
        h_new = h + h_new  # Residual
        
        # Update coordinates (equivariant)
        coord_weight = self.coord_mlp(edge_msg)  # (E, 1)
        coord_diff = rel_pos * coord_weight  # (E, 3)
        
        x_new = x.clone()
        x_new.index_add_(0, col, coord_diff)
        
        return h_new, x_new


class EGNNModel(nn.Module):
    """Full EGNN model for diffusion."""
    
    def __init__(self, n_atoms=5, hidden_dim=128, n_layers=3):
        super().__init__()
        self.n_atoms = n_atoms
        
        # Initial embedding
        self.node_emb = nn.Linear(1, hidden_dim)  # Atom type (dummy for now)
        self.time_emb = nn.Linear(1, hidden_dim)
        self.prop_emb = nn.Linear(1, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output: predict noise on coordinates
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x, t, bandgap):
        """
        x: (B, N, 3) coordinates
        t: (B,) timestep
        bandgap: (B,) property
        """
        B, N = x.shape[0], x.shape[1]
        
        # Flatten batch
        x_flat = x.reshape(B * N, 3)
        
        # Node features: time + property
        h = torch.ones(B * N, 1, device=x.device)
        h = self.node_emb(h)
        
        t_emb = self.time_emb(t.unsqueeze(-1).float())
        p_emb = self.prop_emb(bandgap.unsqueeze(-1).float())
        
        # Broadcast to all nodes
        global_emb = (t_emb + p_emb).repeat_interleave(N, dim=0)
        h = h + global_emb
        
        # Fully connected graph within each structure
        edge_index = self._get_edges(B, N, x.device)
        
        # Message passing
        for layer in self.layers:
            h, x_flat = layer(h, x_flat, edge_index)
        
        # Predict noise
        noise = self.out_mlp(h)
        
        return noise.reshape(B, N, 3)
    
    def _get_edges(self, B, N, device):
        """Create fully connected graph for each structure."""
        edges = []
        for b in range(B):
            offset = b * N
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edges.append([offset + i, offset + j])
        
        return torch.tensor(edges, device=device).t()
