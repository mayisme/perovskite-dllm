"""生成模块。

实现DDPM/DDIM采样和classifier-free guidance。
"""
import torch
import logging
from typing import List
from pymatgen.core import Structure, Lattice
import os

logger = logging.getLogger(__name__)


class PerovskiteGenerator:
    """钙钛矿结构生成器。"""

    def __init__(self, model, diffusion, config, device="cuda"):
        """初始化生成器。"""
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        atom_types: torch.Tensor,
        band_gap: float,
        formation_energy: float,
        num_samples: int = 10,
        guidance_scale: float = 3.0,
        sampler: str = "ddpm"
    ) -> List[Structure]:
        """生成钙钛矿结构。"""
        self.model.eval()

        # 初始化噪声
        x = torch.randn(num_samples, 5, 3, device=self.device)
        lattice_log = torch.randn(num_samples, 6, device=self.device)

        # 条件
        atom_types_batch = atom_types.unsqueeze(0).repeat(num_samples, 1).to(self.device)
        band_gap_batch = torch.full((num_samples,), band_gap, device=self.device)
        formation_energy_batch = torch.full((num_samples,), formation_energy, device=self.device)

        # 逆向扩散
        if sampler == "ddpm":
            for t in range(self.diffusion.T - 1, -1, -1):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

                # 预测噪声
                noise_pred_coords, noise_pred_lattice = self.model(
                    x, t_batch, atom_types_batch, lattice_log,
                    band_gap_batch, formation_energy_batch
                )

                # DDPM步骤
                lattice_log, x = self.diffusion.ddpm_sample_step(
                    lattice_log, x, noise_pred_lattice, noise_pred_coords, t
                )

        elif sampler == "ddim":
            steps = self.config.get('sampling_steps', 50)
            time_steps = torch.linspace(self.diffusion.T - 1, 0, steps, dtype=torch.long)

            for i, t in enumerate(time_steps):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                t_prev = time_steps[i + 1] if i + 1 < len(time_steps) else -1

                noise_pred_coords, noise_pred_lattice = self.model(
                    x, t_batch, atom_types_batch, lattice_log,
                    band_gap_batch, formation_energy_batch
                )

                lattice_log, x = self.diffusion.ddim_sample_step(
                    lattice_log, x, noise_pred_lattice, noise_pred_coords,
                    t.item(), t_prev.item()
                )

        # 转换为Structure对象
        lattice_params = self.diffusion.log_space_to_lattice(lattice_log)
        x = self.diffusion.wrap_frac_coords(x)

        structures = []
        for i in range(num_samples):
            try:
                # 转换晶格参数为Lattice对象
                a, b, c = lattice_params[i, :3].cpu().numpy()
                alpha, beta, gamma = lattice_params[i, 3:].cpu().numpy()

                lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

                # 创建Structure
                species = [int(z) for z in atom_types_batch[i].cpu().numpy()]
                coords = x[i].cpu().numpy()

                structure = Structure(lattice, species, coords)
                structures.append(structure)

            except Exception as e:
                logger.warning(f"Failed to create structure {i}: {e}")

        logger.info(f"Generated {len(structures)}/{num_samples} valid structures")
        return structures

    def save_structures(self, structures: List[Structure], output_dir: str):
        """保存生成的结构为CIF文件。"""
        os.makedirs(output_dir, exist_ok=True)

        for i, structure in enumerate(structures):
            output_path = os.path.join(output_dir, f"generated_{i}.cif")
            structure.to(filename=output_path)

        logger.info(f"Saved {len(structures)} structures to {output_dir}")
