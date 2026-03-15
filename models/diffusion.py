"""扩散调度器，支持晶格对数空间和分数坐标PBC处理。"""
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DiffusionSchedule:
    """扩散调度器，支持DDPM/DDIM采样。

    关键设计：
    - 晶格参数在对数空间扩散：(log a, log b, log c, α, β, γ)
    - 分数坐标使用minimum-image convention处理PBC
    """

    def __init__(
        self,
        T: int = 500,
        schedule_type: str = 'cosine',
        device: str = "cpu"
    ):
        """初始化扩散调度器。

        Args:
            T: 扩散时间步数
            schedule_type: 调度类型 ('cosine' 或 'linear')
            device: 设备
        """
        self.T = T
        self.device = device
        self.schedule_type = schedule_type

        # 计算beta调度
        if schedule_type == 'cosine':
            betas = self._cosine_beta_schedule(T)
        elif schedule_type == 'linear':
            betas = self._linear_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1 - betas

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        logger.info(
            f"Initialized diffusion schedule: T={T}, type={schedule_type}, "
            f"alpha_bar[0]={self.alpha_bar[0]:.4f}, "
            f"alpha_bar[-1]={self.alpha_bar[-1]:.4f}"
        )

    def _cosine_beta_schedule(self, T: int, s: float = 0.008) -> torch.Tensor:
        """余弦beta调度。

        Args:
            T: 时间步数
            s: 偏移参数

        Returns:
            beta值
        """
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _linear_beta_schedule(
        self,
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> torch.Tensor:
        """线性beta调度。

        Args:
            T: 时间步数
            beta_start: 起始beta值
            beta_end: 结束beta值

        Returns:
            beta值
        """
        return torch.linspace(beta_start, beta_end, T)

    def lattice_to_log_space(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """将晶格参数转换为对数空间。

        Args:
            lattice_params: (B, 6) - (a, b, c, α, β, γ)

        Returns:
            (B, 6) - (log a, log b, log c, α, β, γ)
        """
        log_params = lattice_params.clone()
        log_params[:, :3] = torch.log(lattice_params[:, :3])
        return log_params

    def log_space_to_lattice(self, log_params: torch.Tensor) -> torch.Tensor:
        """对数空间转回实空间。

        Args:
            log_params: (B, 6) - (log a, log b, log c, α, β, γ)

        Returns:
            (B, 6) - (a, b, c, α, β, γ)
        """
        lattice_params = log_params.clone()
        lattice_params[:, :3] = torch.exp(log_params[:, :3])
        return lattice_params

    def wrap_frac_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """将分数坐标包裹到[0,1)区间。

        Args:
            coords: (..., 3) 分数坐标

        Returns:
            包裹后的坐标
        """
        return coords - torch.floor(coords)

    def add_noise(
        self,
        x0_lattice: torch.Tensor,
        x0_coords: torch.Tensor,
        t: torch.Tensor
    ) -> tuple:
        """联合加噪：晶格参数和分数坐标。

        Args:
            x0_lattice: (B, 6) 晶格参数
            x0_coords: (B, N, 3) 分数坐标
            t: (B,) 时间步

        Returns:
            (xt_lattice, xt_coords, noise_lattice, noise_coords)
        """
        # 晶格参数在对数空间加噪
        x0_log = self.lattice_to_log_space(x0_lattice)
        noise_lattice = torch.randn_like(x0_log)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)

        xt_log = sqrt_alpha_bar_t * x0_log + sqrt_one_minus_alpha_bar_t * noise_lattice

        # 分数坐标加噪后包裹
        noise_coords = torch.randn_like(x0_coords)
        xt_coords = sqrt_alpha_bar_t.view(-1, 1, 1) * x0_coords + \
                    sqrt_one_minus_alpha_bar_t.view(-1, 1, 1) * noise_coords
        xt_coords = self.wrap_frac_coords(xt_coords)

        return xt_log, xt_coords, noise_lattice, noise_coords

    def predict_x0_from_noise(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """从噪声预测反推x₀。

        Args:
            xt: 噪声数据
            noise_pred: 预测的噪声
            t: 时间步

        Returns:
            预测的x₀
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

        # 处理维度
        if xt.dim() == 2:  # 晶格参数 (B, 6)
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1)
        elif xt.dim() == 3:  # 坐标 (B, N, 3)
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1)

        x0_pred = (xt - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        return x0_pred

    @torch.no_grad()
    def ddpm_sample_step(
        self,
        xt_lattice: torch.Tensor,
        xt_coords: torch.Tensor,
        noise_pred_lattice: torch.Tensor,
        noise_pred_coords: torch.Tensor,
        t: int
    ) -> tuple:
        """DDPM采样单步。

        Args:
            xt_lattice: (B, 6) 当前晶格参数（对数空间）
            xt_coords: (B, N, 3) 当前分数坐标
            noise_pred_lattice: (B, 6) 预测的晶格噪声
            noise_pred_coords: (B, N, 3) 预测的坐标噪声
            t: 当前时间步

        Returns:
            (xt_minus_1_lattice, xt_minus_1_coords)
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.betas[t]

        # 计算均值
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

        mean_lattice = coef1 * (xt_lattice - coef2 * noise_pred_lattice)
        mean_coords = coef1 * (xt_coords - coef2 * noise_pred_coords)

        # 添加噪声（t>0时）
        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            z_lattice = torch.randn_like(xt_lattice)
            z_coords = torch.randn_like(xt_coords)

            xt_minus_1_lattice = mean_lattice + sigma_t * z_lattice
            xt_minus_1_coords = mean_coords + sigma_t * z_coords
        else:
            xt_minus_1_lattice = mean_lattice
            xt_minus_1_coords = mean_coords

        # 包裹分数坐标
        xt_minus_1_coords = self.wrap_frac_coords(xt_minus_1_coords)

        return xt_minus_1_lattice, xt_minus_1_coords

    @torch.no_grad()
    def ddim_sample_step(
        self,
        xt_lattice: torch.Tensor,
        xt_coords: torch.Tensor,
        noise_pred_lattice: torch.Tensor,
        noise_pred_coords: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0
    ) -> tuple:
        """DDIM采样单步（更快）。

        Args:
            xt_lattice: (B, 6) 当前晶格参数（对数空间）
            xt_coords: (B, N, 3) 当前分数坐标
            noise_pred_lattice: (B, 6) 预测的晶格噪声
            noise_pred_coords: (B, N, 3) 预测的坐标噪声
            t: 当前时间步
            t_prev: 前一个时间步
            eta: 随机性参数（0为确定性）

        Returns:
            (xt_prev_lattice, xt_prev_coords)
        """
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # 预测x0
        x0_pred_lattice = self.predict_x0_from_noise(
            xt_lattice, noise_pred_lattice, torch.tensor([t]).to(self.device)
        )
        x0_pred_coords = self.predict_x0_from_noise(
            xt_coords, noise_pred_coords, torch.tensor([t]).to(self.device)
        )

        # DDIM更新
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
                  torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

        dir_xt_lattice = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * noise_pred_lattice
        dir_xt_coords = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * noise_pred_coords

        xt_prev_lattice = torch.sqrt(alpha_bar_t_prev) * x0_pred_lattice + dir_xt_lattice
        xt_prev_coords = torch.sqrt(alpha_bar_t_prev) * x0_pred_coords + dir_xt_coords

        if eta > 0:
            z_lattice = torch.randn_like(xt_lattice)
            z_coords = torch.randn_like(xt_coords)
            xt_prev_lattice += sigma_t * z_lattice
            xt_prev_coords += sigma_t * z_coords

        # 包裹分数坐标
        xt_prev_coords = self.wrap_frac_coords(xt_prev_coords)

        return xt_prev_lattice, xt_prev_coords
