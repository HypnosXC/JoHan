from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np
from torch import Tensor, nn
from einops import rearrange
from jaxtyping import Float, Bool
from rotary_embedding_torch import RotaryEmbedding
def make_positional_encoding(
    d_latent: int, length: int, dtype: torch.dtype
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


def fourier_encode(
    x: Float[Tensor, "*#batch channels"], freqs: int
) -> Float[Tensor, "*#batch channels+2*freqs*channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs))
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )




@dataclass(frozen=True)
class TransformerBlockConfig:
    d_latent: int
    d_noise_emb: int
    d_feedforward: int
    n_heads: int
    dropout_p: float
    activation: Literal["gelu", "relu"]
    include_xattn: bool
    use_rope_embedding: bool
    use_film_noise_conditioning: bool
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"]


class TransformerBlock(nn.Module):
    """An even-tempered Transformer block."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.sattn_qkv_proj = nn.Linear(
            config.d_latent, config.d_latent * 3, bias=False
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(config.d_latent // config.n_heads)
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent, config.d_latent * 2, bias=False
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent, config.d_latent, bias=False
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward, elementwise_affine=False, bias=False
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False)
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    def forward(
        self,
        x: Float[Tensor, "batch tokens d_latent"],
        attn_mask: Bool[Tensor, "batch 1 tokens tokens"] | None,
        noise_emb: Float[Tensor, "batch d_noise_emb"],
        cond: Float[Tensor, "batch tokens d_latent"] | None = None,
    ) -> Float[Tensor, "batch tokens d_latent"]:
        config = self.config
        (batch, time, d_latent) = x.shape

        # Self-attention.
        # We put layer normalization after the residual connection.
        x = self.layernorm1(x + self._sattn(x, attn_mask))

        # Include conditioning.
        if config.include_xattn:
            assert cond is not None
            x = self.xattn_layernorm(x + self._xattn(x, attn_mask, cond=cond))

        mlp_out = x
        mlp_out = self.mlp0(mlp_out)
        mlp_out = self.activation(mlp_out)

        # FiLM-style conditioning.
        if self.mlp_film_cond_proj is not None:
            scale, shift = torch.chunk(
                self.mlp_film_cond_proj(noise_emb), chunks=2, dim=-1
            )
            assert scale.shape == shift.shape == (batch, config.d_feedforward)
            mlp_out = (
                self.norm_no_learnable(mlp_out) * (1.0 + scale[:, None, :])
                + shift[:, None, :]
            )

        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp1(mlp_out)

        x = self.layernorm2(x + mlp_out)
        assert x.shape == (batch, time, d_latent)
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Multi-head self-attention."""
        config = self.config
        q, k, v = rearrange(
            self.sattn_qkv_proj(x),
            "b t (qkv nh dh) -> qkv b nh t dh",
            qkv=3,
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = self.sattn_out_proj(x)
        return x

    def _xattn(self, x: Tensor, attn_mask: Tensor | None, cond: Tensor) -> Tensor:
        """Multi-head cross-attention."""
        config = self.config
        k, v = rearrange(
            self.xattn_kv_proj(
                {
                    "kv_from_cond_q_from_x": cond,
                    "kv_from_x_q_from_cond": x,
                }[self.config.xattn_mode]
            ),
            "b t (qk nh dh) -> qk b nh t dh",
            qk=2,
            nh=config.n_heads,
        )
        q = rearrange(
            self.xattn_q_proj(
                {
                    "kv_from_cond_q_from_x": x,
                    "kv_from_x_q_from_cond": cond,
                }[self.config.xattn_mode]
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

