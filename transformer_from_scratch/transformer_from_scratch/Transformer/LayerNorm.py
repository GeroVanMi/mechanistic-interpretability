import torch
import torch.nn as nn
from jaxtyping import Float, Int
from ModelConfig import Config
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(dim=-1, keepdim=True)
        std = residual.std(dim=-1, keepdim=True, unbiased=False)

        # Standardization
        residual = (residual - mean) / std

        # Scale (*) with learned weights and translate (+) with learned bias
        return residual * self.w + self.b
