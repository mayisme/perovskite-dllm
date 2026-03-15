"""E(n)-等变图神经网络，支持PBC边构建。"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class EGNNLayer(nn.Module):
    """E(n)-等变消息传递层，支持PBC和注意力机制。"""

    def __init__(self, hidden_dim: int = 128, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff

        # 边模型：计算消息
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

        # 坐标更新（等变）
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def build_edges_pbc(
        self,
        frac_coords: torch.Tensor,
        lattice_params: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> tuple:
        """构建PBC-aware边（minimum-image convention）。

        Args:
            frac_coords: (N, 3) 分数坐标
            lattice_params: (B, 6) 晶格参数
            batch_idx: (N,) 批次索引

        Returns:
            (edge_index, edge_dist)
        """
        device = frac_coords.device
        N = frac_coords.shape[0]

        # 转换晶格参数为矩阵
        lattice_matrices = self._params_to_matrix(lattice_params)  # (B, 3, 3)

        edges = []
        dists = []

        for i in range(N):
            batch_i = batch_idx[i]
            lattice_i = lattice_matrices[batch_i]

            for j in range(N):
                if i == j or batch_idx[j] != batch_i:
                    continue

                # Minimum-image convention
                min_dist = float('inf')
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            offset = torch.tensor([dx, dy, dz], device=device, dtype=torch.float32)
                            frac_j_image = frac_coords[j] + offset

                            cart_i = torch.matmul(frac_coords[i], lattice_i)
                            cart_j = torch.matmul(frac_j_image, lattice_i)

                            dist = torch.norm(cart_i - cart_j).item()
                            min_dist = min(min_dist, dist)

                if min_dist < self.cutoff:
                    edges.append([i, j])
                    dists.append(min_dist)

        if not edges:
            # 如果没有边，创建自环
            edges = [[i, i] for i in range(N)]
            dists = [0.0] * N

        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t()
        edge_dist = torch.tensor(dists, device=device, dtype=torch.float32)

        return edge_index, edge_dist

    def _params_to_matrix(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """将晶格参数转换为矩阵。

        Args:
            lattice_params: (B, 6) - (a, b, c, α, β, γ)

        Returns:
            (B, 3, 3) 晶格矩阵
        """
        a, b, c = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2]
        alpha, beta, gamma = lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]

        # 转换为弧度
        alpha_rad = alpha * torch.pi / 180.0
        beta_rad = beta * torch.pi / 180.0
        gamma_rad = gamma * torch.pi / 180.0

        # 构建晶格矩阵
        cos_alpha = torch.cos(alpha_rad)
        cos_beta = torch.cos(beta_rad)
        cos_gamma = torch.cos(gamma_rad)
        sin_gamma = torch.sin(gamma_rad)

        vol = torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 +
                        2 * cos_alpha * cos_beta * cos_gamma)

        matrices = torch.zeros(lattice_params.shape[0], 3, 3, device=lattice_params.device)
        matrices[:, 0, 0] = a
        matrices[:, 0, 1] = b * cos_gamma
        matrices[:, 0, 2] = c * cos_beta
        matrices[:, 1, 1] = b * sin_gamma
        matrices[:, 1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        matrices[:, 2, 2] = c * vol / sin_gamma

        return matrices

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor
    ) -> tuple:
        """前向传播。

        Args:
            h: (N, hidden_dim) 节点特征
            x: (N, 3) 笛卡尔坐标
            edge_index: (2, E) 边索引
            edge_dist: (E,) 边距离

        Returns:
            (h_new, x_new)
        """
        row, col = edge_index

        # 计算边特征
        rel_pos = x[row] - x[col]  # (E, 3)
        dist = edge_dist.unsqueeze(-1)  # (E, 1)

        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        edge_msg = self.edge_mlp(edge_feat)  # (E, hidden_dim)

        # 注意力权重
        attn_weights = self.attention(edge_msg)  # (E, 1)
        edge_msg = edge_msg * attn_weights

        # 聚合消息
        agg_msg = torch.zeros_like(h)
        agg_msg.index_add_(0, col, edge_msg)

        # 更新节点特征（带残差连接）
        h_new = self.node_mlp(torch.cat([h, agg_msg], dim=-1))
        h_new = self.norm(h + h_new)

        # 更新坐标（等变）
        coord_weight = self.coord_mlp(edge_msg)  # (E, 1)
        coord_diff = rel_pos * coord_weight  # (E, 3)

        x_new = x.clone()
        x_new.index_add_(0, col, coord_diff)

        return h_new, x_new


class EGNNModel(nn.Module):
    """改进的EGNN用于晶体扩散生成。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_atom_types: int = 100,
        cutoff: float = 6.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 嵌入层
        self.node_emb = nn.Embedding(n_atom_types, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bandgap_emb = nn.Linear(1, hidden_dim)
        self.formation_energy_emb = nn.Linear(1, hidden_dim)

        # EGNN层
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, cutoff) for _ in range(n_layers)
        ])

        # 输出层
        self.out_coords = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

        # 晶格参数预测（6维：a,b,c,α,β,γ）
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
    ) -> tuple:
        """前向传播。

        Args:
            x: (B, N, 3) 分数坐标
            t: (B,) 时间步
            atom_types: (B, N) 原子类型
            lattice_params: (B, 6) 晶格参数
            band_gap: (B,) 带隙
            formation_energy: (B,) 形成能

        Returns:
            (noise_pred_coords, noise_pred_lattice)
        """
        B, N = x.shape[:2]
        device = x.device

        # 嵌入
        h = self.node_emb(atom_types)  # (B, N, hidden_dim)
        t_emb = self.time_emb(t.view(B, 1, 1).float() / 1000.0)
        bg_emb = self.bandgap_emb(band_gap.view(B, 1, 1))
        fe_emb = self.formation_energy_emb(formation_energy.view(B, 1, 1))

        h = h + t_emb + bg_emb + fe_emb

        # 展平批次
        h_flat = h.reshape(B * N, -1)
        x_flat = x.reshape(B * N, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        # 转换分数坐标为笛卡尔坐标
        lattice_matrices = self.layers[0]._params_to_matrix(lattice_params)
        x_cart_flat = torch.zeros_like(x_flat)
        for b in range(B):
            mask = batch_idx == b
            x_cart_flat[mask] = torch.matmul(x_flat[mask], lattice_matrices[b])

        # 构建边
        edge_index, edge_dist = self.layers[0].build_edges_pbc(
            x_flat, lattice_params, batch_idx
        )

        # EGNN层
        for layer in self.layers:
            h_flat, x_cart_flat = layer(h_flat, x_cart_flat, edge_index, edge_dist)

        # 全局池化用于晶格预测
        h_global = h_flat.reshape(B, N, -1).mean(dim=1)

        # 预测噪声
        pred_noise_coords = self.out_coords(h_flat).reshape(B, N, 3)
        pred_noise_lattice = self.out_lattice(h_global)

        return pred_noise_coords, pred_noise_lattice
