"""优化的EGNN边构建。"""
import torch
import torch.nn as nn

try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


class FastEGNNLayer(nn.Module):
    """优化的E(n)-等变消息传递层。
    
    优先使用 torch_cluster.radius_graph，否则使用向量化实现。
    """

    def __init__(self, hidden_dim: int = 128, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff

        # 边模型
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 节点更新
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 坐标更新
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def build_edges_fast(
        self,
        cart_coords: torch.Tensor,
        batch_idx: torch.Tensor,
        max_neighbors: int = 32
    ) -> tuple:
        """使用 torch_cluster.radius_graph 快速构建边。
        
        Args:
            cart_coords: (N, 3) 笛卡尔坐标
            batch_idx: (N,) 批次索引
            max_neighbors: 最大邻居数
            
        Returns:
            (edge_index, edge_dist)
        """
        if not HAS_TORCH_CLUSTER:
            raise RuntimeError("torch_cluster not installed. Use build_edges_vectorized instead.")
        
        # 使用 torch_cluster 的 radius_graph
        edge_index = radius_graph(
            cart_coords,
            r=self.cutoff,
            batch=batch_idx,
            loop=False,
            max_num_neighbors=max_neighbors
        )
        
        # 计算边距离
        row, col = edge_index
        edge_vec = cart_coords[row] - cart_coords[col]
        edge_dist = torch.norm(edge_vec, dim=-1)
        
        return edge_index, edge_dist

    def build_edges_vectorized(
        self,
        cart_coords: torch.Tensor,
        batch_idx: torch.Tensor,
        max_neighbors: int = 32
    ) -> tuple:
        """向量化边构建（fallback，不依赖 torch_cluster）。
        
        Args:
            cart_coords: (N, 3) 笛卡尔坐标
            batch_idx: (N,) 批次索引
            max_neighbors: 最大邻居数
            
        Returns:
            (edge_index, edge_dist)
        """
        N = cart_coords.shape[0]
        device = cart_coords.device
        
        edges = []
        dists = []
        
        # 分batch处理
        for b in batch_idx.unique():
            mask = batch_idx == b
            coords_b = cart_coords[mask]
            indices_b = torch.where(mask)[0]
            
            Nb = coords_b.shape[0]
            
            # 计算距离矩阵
            diff = coords_b.unsqueeze(0) - coords_b.unsqueeze(1)
            dist_matrix = torch.norm(diff, dim=-1)
            
            # 找到cutoff内的边（排除自环）
            mask_cutoff = (dist_matrix < self.cutoff) & (dist_matrix > 0)
            
            # 限制每个节点的邻居数
            for i in range(Nb):
                neighbors = torch.where(mask_cutoff[i])[0]
                if len(neighbors) > max_neighbors:
                    _, top_k_idx = torch.topk(dist_matrix[i, neighbors], max_neighbors, largest=False)
                    neighbors = neighbors[top_k_idx]
                
                for j in neighbors:
                    edges.append([indices_b[i].item(), indices_b[j].item()])
                    dists.append(dist_matrix[i, j].item())
        
        if not edges:
            # 如果没有边，创建自环
            edges = [[i, i] for i in range(N)]
            dists = [0.0] * N
        
        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t()
        edge_dist = torch.tensor(dists, device=device, dtype=torch.float32)
        
        return edge_index, edge_dist

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor
    ) -> tuple:
        """前向传播。"""
        row, col = edge_index

        # 计算边特征
        rel_pos = x[row] - x[col]
        dist = edge_dist.unsqueeze(-1)

        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        edge_msg = self.edge_mlp(edge_feat)

        # 注意力权重
        attn_weights = self.attention(edge_msg)
        edge_msg = edge_msg * attn_weights

        # 聚合消息
        agg_msg = torch.zeros_like(h)
        agg_msg.index_add_(0, col, edge_msg)

        # 更新节点特征
        h_new = self.node_mlp(torch.cat([h, agg_msg], dim=-1))
        h_new = self.norm(h + h_new)

        # 更新坐标（等变）
        coord_weights = self.coord_mlp(edge_msg)
        coord_diff = rel_pos * coord_weights
        
        agg_coord = torch.zeros_like(x)
        agg_coord.index_add_(0, col, coord_diff)
        
        x_new = x + agg_coord

        return h_new, x_new
