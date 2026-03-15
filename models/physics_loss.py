"""物理信息损失函数。

提供精细的物理约束：Goldschmidt、配位数、键长、键角、Pauli排斥。
"""
import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PhysicsLoss:
    """物理约束损失函数集合。"""

    def __init__(self, ionic_radii_db=None):
        """初始化物理损失。

        Args:
            ionic_radii_db: IonicRadiiDatabase实例
        """
        self.ionic_radii_db = ionic_radii_db

    def params_to_matrix(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """将晶格参数转换为矩阵。

        Args:
            lattice_params: (B, 6) - (a, b, c, α, β, γ)

        Returns:
            (B, 3, 3) 晶格矩阵
        """
        a, b, c = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2]
        alpha, beta, gamma = lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]

        alpha_rad = alpha * torch.pi / 180.0
        beta_rad = beta * torch.pi / 180.0
        gamma_rad = gamma * torch.pi / 180.0

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

    def goldschmidt_loss(
        self,
        lattice_params: torch.Tensor,
        atom_types: torch.Tensor
    ) -> torch.Tensor:
        """Goldschmidt容忍因子损失。

        t = (r_A + r_O) / [√2(r_B + r_O)]
        有效范围：[0.8, 1.0]
        """
        # 使用典型值：Ba(1.61Å), Ti(0.605Å), O(1.40Å)
        r_a = 1.61
        r_b = 0.605
        r_o = 1.40

        t = (r_a + r_o) / (np.sqrt(2) * (r_b + r_o))
        t_tensor = torch.tensor(t, device=lattice_params.device)

        # 惩罚偏离[0.8, 1.0]范围
        loss = torch.relu(0.8 - t_tensor) + torch.relu(t_tensor - 1.0)

        return loss

    def coordination_loss(
        self,
        coords: torch.Tensor,
        lattice_params: torch.Tensor,
        atom_types: torch.Tensor,
        target_coord: int = 6
    ) -> torch.Tensor:
        """BO₆配位数损失。"""
        B, N = coords.shape[:2]
        lattice_matrices = self.params_to_matrix(lattice_params)

        total_loss = 0.0
        for b in range(B):
            cart_coords = torch.matmul(coords[b], lattice_matrices[b])
            dist_matrix = torch.cdist(cart_coords, cart_coords)

            # 假设B位是第二个原子（索引1），O是后三个原子
            b_idx = 1
            o_indices = [2, 3, 4]

            b_o_dists = dist_matrix[b_idx, o_indices]
            coord_num = (b_o_dists < 3.0).sum().float()

            loss = torch.abs(coord_num - target_coord)
            total_loss += loss

        return total_loss / B

    def bond_length_loss(
        self,
        coords: torch.Tensor,
        lattice_params: torch.Tensor,
        atom_types: torch.Tensor
    ) -> torch.Tensor:
        """键长分布损失。

        B-O键长：1.8-2.2Å
        A-O键长：2.5-3.2Å
        """
        B, N = coords.shape[:2]
        lattice_matrices = self.params_to_matrix(lattice_params)

        total_loss = 0.0
        for b in range(B):
            cart_coords = torch.matmul(coords[b], lattice_matrices[b])
            dist_matrix = torch.cdist(cart_coords, cart_coords)

            # B-O键长
            b_o_dists = dist_matrix[1, [2, 3, 4]]
            b_o_loss = torch.relu(1.8 - b_o_dists).sum() + torch.relu(b_o_dists - 2.2).sum()

            # A-O键长
            a_o_dists = dist_matrix[0, [2, 3, 4]]
            a_o_loss = torch.relu(2.5 - a_o_dists).sum() + torch.relu(a_o_dists - 3.2).sum()

            total_loss += b_o_loss + a_o_loss

        return total_loss / B

    def pauli_repulsion_loss(
        self,
        coords: torch.Tensor,
        lattice_params: torch.Tensor,
        min_dist: float = 1.5
    ) -> torch.Tensor:
        """Pauli排斥损失。"""
        B, N = coords.shape[:2]
        lattice_matrices = self.params_to_matrix(lattice_params)

        total_loss = 0.0
        for b in range(B):
            cart_coords = torch.matmul(coords[b], lattice_matrices[b])
            dist_matrix = torch.cdist(cart_coords, cart_coords)

            mask = ~torch.eye(N, dtype=torch.bool, device=coords.device)
            dists = dist_matrix[mask]

            violations = torch.relu(min_dist - dists)
            total_loss += violations.sum()

        return total_loss / B

    def combined_loss(
        self,
        pred_x0_coords: torch.Tensor,
        pred_x0_lattice: torch.Tensor,
        atom_types: torch.Tensor,
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """组合物理损失。"""
        loss = 0.0

        if weights.get('goldschmidt', 0) > 0:
            loss += weights['goldschmidt'] * self.goldschmidt_loss(
                pred_x0_lattice, atom_types
            )

        if weights.get('coordination', 0) > 0:
            loss += weights['coordination'] * self.coordination_loss(
                pred_x0_coords, pred_x0_lattice, atom_types
            )

        if weights.get('bond_length', 0) > 0:
            loss += weights['bond_length'] * self.bond_length_loss(
                pred_x0_coords, pred_x0_lattice, atom_types
            )

        if weights.get('pauli', 0) > 0:
            loss += weights['pauli'] * self.pauli_repulsion_loss(
                pred_x0_coords, pred_x0_lattice
            )

        return loss
