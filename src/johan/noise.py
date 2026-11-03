from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float
from .tensor_dataclass import TensorDataclass
class CosineNoiseScheduleConstants(TensorDataclass):
    alpha_t: Float[Tensor, "T"]
    r"""$1 - \beta_t$"""

    alpha_bar_t: Float[Tensor, "T+1"]
    r"""$\Prod_{j=1}^t (1 - \beta_j)$"""

    @staticmethod
    def compute(timesteps: int, s: float = 0.008) -> CosineNoiseScheduleConstants:
        steps = timesteps + 1
        x = torch.linspace(0, 1, steps, dtype=torch.float64)

        def get_betas():
            alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        alpha_t = 1.0 - get_betas()
        assert len(alpha_t.shape) == 1
        alpha_cumprod_t = torch.cat(
            [torch.ones((1,)), torch.cumprod(alpha_t, dim=0)],
            dim=0,
        )
        return CosineNoiseScheduleConstants(
            alpha_t=alpha_t, alpha_bar_t=alpha_cumprod_t
        )

