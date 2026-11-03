"""2D pose decoder + 3D temporal attention blocks for the hand denoiser.

Module layout:
  - LatentCNNEncoder     : DINO patch features → spatial tokens + global feat
  - _film_ffn / _sdpa    : tiny reusable building blocks
  - TemporalCrossAttnBlock: 3D decoder cross-attn (Q=noise, KV=cond), RoPE+FiLM
  - TemporalSelfAttnBlock : 3D decoder self-attn  (Q=K=V=noise),     RoPE+FiLM
  - Pose2DBlock          : 2D decoder = spatial cross-attn + temporal self-attn
  - Pose2DTransformerDecoder: full 2D pose decoder
  - GlobalPoseHead       : optional global translation/orientation predictor

All blocks expose the same call signature
    block(queries, memory, batch, time_q, noise_emb=..., time_kv=...)
so they can be interleaved in a single ModuleList.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from jaxtyping import Float
from rotary_embedding_torch import RotaryEmbedding


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def _film_ffn(x, norm, mlp0, act, drop, mlp1, film, noise_emb):
    """Pre-norm FFN with FiLM(noise_emb) modulation on the activation."""
    h = act(mlp0(norm(x)))
    if noise_emb is not None:
        film_params = film(noise_emb)
        if film_params.dim() == 2 and h.dim() == 3:
            film_params = film_params.unsqueeze(1)
        scale, shift = film_params.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
    return drop(mlp1(drop(h)))


def _sdpa(q, k, v, dropout_p: float):
    """Wrapper around scaled_dot_product_attention. Uses Flash Attention when available."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


# ---------------------------------------------------------------------------
# CNN encoder for image patch features
# ---------------------------------------------------------------------------


class LatentCNNEncoder(nn.Module):
    """DINO patch features (768x16x16) → spatial tokens + global feature.

    With downsample=True the spatial grid is reduced from 16x16 (256 tokens) to
    8x8 (64 tokens) via a stride-2 conv, which makes the 2D decoder's spatial
    cross-attention 4x cheaper.
    """

    def __init__(
        self,
        patch_feat_channels: int = 768,
        d_out: int = 64,
        global_feat_dim: int = 128,
        activation: str = "gelu",
        downsample: bool = False,
        patch_grid: int = 16,
    ):
        super().__init__()
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.conv_layers = nn.Sequential(
            nn.Conv2d(patch_feat_channels, 128, kernel_size=1),         # 768 → 128 (1x1)
            nn.GroupNorm(8, 128),
            Activation(),
            nn.Conv2d(128, d_out, kernel_size=3, stride=1, padding=1),  # 128 → d_out (3x3)
            nn.GroupNorm(4, d_out),
            Activation(),
        )

        self.downsample = downsample
        if downsample:
            self.spatial_downsample = nn.Sequential(
                nn.Conv2d(d_out, d_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, d_out),
                Activation(),
            )
            # DINO emits image_size/16 patches per side (16 @256, 32 @512); the
            # stride-2 conv halves that grid.
            self.num_tokens = (patch_grid // 2) ** 2  # 64 @256, 256 @512
        else:
            self.num_tokens = patch_grid**2  # 256 @256, 1024 @512

        self.d_out = d_out
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, d_out) * 0.02
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(nn.Linear(d_out, global_feat_dim))


# ---------------------------------------------------------------------------
# Temporal blocks for the 3D decoder
# ---------------------------------------------------------------------------


class TemporalCrossAttnBlock(nn.Module):
    """3D-decoder cross-attn block.

    Q  = noise_input  (B, T,   D)
    KV = encoded_cond (B, T/k, D)   ← asymmetric T vs T/k

    RoPE on Q/K, FiLM(noise_emb) on the FFN. Used as the "cross" half of the
    interleaved cross/self decoder stack.
    """

    def __init__(self, d_latent=512, d_noise_emb=1024, num_heads=4,
                 d_feedforward=2048, dropout_p=0.1, activation="gelu", use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_latent // num_heads
        self.dropout_p = dropout_p
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.rotary_emb = RotaryEmbedding(self.d_head) if use_rope else None

        # Cross-attention projections.
        self.norm_q = nn.LayerNorm(d_latent)
        self.norm_kv = nn.LayerNorm(d_latent)
        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.kv_proj = nn.Linear(d_latent, d_latent * 2, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent, bias=False)

        # FFN with FiLM(noise_emb).
        self.ffn_norm = nn.LayerNorm(d_latent)
        self.ffn_mlp0 = nn.Linear(d_latent, d_feedforward)
        self.ffn_act = Activation()
        self.ffn_drop = nn.Dropout(dropout_p)
        self.ffn_mlp1 = nn.Linear(d_feedforward, d_latent)
        self.film = nn.Linear(d_noise_emb, d_feedforward * 2, bias=False)
        nn.init.zeros_(self.film.weight)

    def forward(self, queries, memory, batch=None, time_q=None, noise_emb=None, time_kv=None):
        """Q=(B, T, D), KV=(B, T_kv, D). Returns (B, T, D)."""
        nh = self.num_heads
        q = rearrange(self.q_proj(self.norm_q(queries)),
                      'b t (nh dh) -> b nh t dh', nh=nh)
        k, v = rearrange(self.kv_proj(self.norm_kv(memory)),
                         'b t (two nh dh) -> two b nh t dh', two=2, nh=nh)
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        attn_out = _sdpa(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b nh t dh -> b t (nh dh)')
        queries = queries + self.out_proj(attn_out)
        queries = queries + _film_ffn(
            queries, self.ffn_norm, self.ffn_mlp0, self.ffn_act,
            self.ffn_drop, self.ffn_mlp1, self.film, noise_emb,
        )
        return queries


class TemporalSelfAttnBlock(nn.Module):
    """3D-decoder self-attn block.

    Q = K = V = noise_input (B, T, D). Same call signature as
    TemporalCrossAttnBlock so the two can be interleaved in a single ModuleList
    (memory and time_kv are accepted but ignored).
    """

    def __init__(self, d_latent=512, d_noise_emb=1024, num_heads=4,
                 d_feedforward=2048, dropout_p=0.1, activation="gelu", use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_latent // num_heads
        self.dropout_p = dropout_p
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.rotary_emb = RotaryEmbedding(self.d_head) if use_rope else None

        self.norm = nn.LayerNorm(d_latent)
        self.qkv_proj = nn.Linear(d_latent, d_latent * 3, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent, bias=False)

        self.ffn_norm = nn.LayerNorm(d_latent)
        self.ffn_mlp0 = nn.Linear(d_latent, d_feedforward)
        self.ffn_act = Activation()
        self.ffn_drop = nn.Dropout(dropout_p)
        self.ffn_mlp1 = nn.Linear(d_feedforward, d_latent)
        self.film = nn.Linear(d_noise_emb, d_feedforward * 2, bias=False)
        nn.init.zeros_(self.film.weight)

    def forward(self, queries, memory=None, batch=None, time_q=None, noise_emb=None, time_kv=None):
        nh = self.num_heads
        qkv = self.qkv_proj(self.norm(queries))
        q, k, v = rearrange(qkv, 'b t (three nh dh) -> three b nh t dh', three=3, nh=nh)
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        attn_out = _sdpa(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b nh t dh -> b t (nh dh)')
        queries = queries + self.out_proj(attn_out)
        queries = queries + _film_ffn(
            queries, self.ffn_norm, self.ffn_mlp0, self.ffn_act,
            self.ffn_drop, self.ffn_mlp1, self.film, noise_emb,
        )
        return queries


# ---------------------------------------------------------------------------
# 2D decoder block (spatial cross-attn + temporal self-attn per joint)
# ---------------------------------------------------------------------------


class Pose2DBlock(nn.Module):
    """2D-decoder block. Two attention layers per call:

    1) Spatial cross-attn (per timestep):
         Q  = joint queries  (B*T, J,    D)
         KV = image tokens   (B*T, S_kv, D)
       FiLM(noise_emb) on the FFN.

    2) Temporal self-attn (per joint, with RoPE):
         Q = K = V = (B*J, T, D)
       FiLM(noise_emb) on the FFN.
    """

    def __init__(self, d_latent=64, d_noise_emb=1024, num_heads=4,
                 d_feedforward=256, dropout_p=0.1, activation="gelu", use_rope=True,
                 temporal_stride: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_latent // num_heads
        self.dropout_p = dropout_p
        # Stride for asymmetric temporal self-attn:
        #   stride=1 → standard self-attn Q=K=V over T (uses temporal_qkv_proj).
        #   stride>1 → Q@T × KV@T/stride (uses separate temporal_q_proj / temporal_kv_proj).
        # KV-side compression happens *before* the KV projection, so both the
        # projection AND the attention matmul are k× cheaper.
        self.temporal_stride = temporal_stride
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.rotary_emb = RotaryEmbedding(self.d_head) if use_rope else None

        # ---- spatial cross-attn ----
        self.spatial_norm_q = nn.LayerNorm(d_latent)
        self.spatial_norm_kv = nn.LayerNorm(d_latent)
        self.spatial_q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.spatial_kv_proj = nn.Linear(d_latent, d_latent * 2, bias=False)
        self.spatial_out_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.spatial_ffn_norm = nn.LayerNorm(d_latent)
        self.spatial_ffn_mlp0 = nn.Linear(d_latent, d_feedforward)
        self.spatial_ffn_act = Activation()
        self.spatial_ffn_drop = nn.Dropout(dropout_p)
        self.spatial_ffn_mlp1 = nn.Linear(d_feedforward, d_latent)
        self.spatial_film = nn.Linear(d_noise_emb, d_feedforward * 2, bias=False)
        nn.init.zeros_(self.spatial_film.weight)

        # ---- temporal self-attn ----
        self.temporal_norm = nn.LayerNorm(d_latent)
        if temporal_stride > 1:
            self.temporal_q_proj = nn.Linear(d_latent, d_latent, bias=False)
            self.temporal_kv_proj = nn.Linear(d_latent, d_latent * 2, bias=False)
        else:
            self.temporal_qkv_proj = nn.Linear(d_latent, d_latent * 3, bias=False)
        self.temporal_out_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.temporal_ffn_norm = nn.LayerNorm(d_latent)
        self.temporal_ffn_mlp0 = nn.Linear(d_latent, d_feedforward)
        self.temporal_ffn_act = Activation()
        self.temporal_ffn_drop = nn.Dropout(dropout_p)
        self.temporal_ffn_mlp1 = nn.Linear(d_feedforward, d_latent)
        self.temporal_film = nn.Linear(d_noise_emb, d_feedforward * 2, bias=False)
        nn.init.zeros_(self.temporal_film.weight)

    def forward(self, queries, memory, batch, time_q, noise_emb=None, time_kv=None):
        """Q=(B*T, J, D), KV=(B*T, S_kv, D). Returns (B*T, J, D).

        Note: the 2D decoder runs at full T (no temporal compression), so
        time_kv is always equal to time_q. The previous time_kv != time_q
        expansion branch was removed.
        """
        J = queries.shape[1]
        nh = self.num_heads
        BT = batch * time_q

        # ---- spatial cross-attn ----
        spatial_noise = None
        if noise_emb is not None:
            spatial_noise = noise_emb[:, None, :].expand(batch, time_q, -1).reshape(BT, -1)

        q = rearrange(self.spatial_q_proj(self.spatial_norm_q(queries)),
                      'b s (nh dh) -> b nh s dh', nh=nh)
        k, v = rearrange(self.spatial_kv_proj(self.spatial_norm_kv(memory)),
                         'b s (two nh dh) -> two b nh s dh', two=2, nh=nh)
        attn_out = _sdpa(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b nh s dh -> b s (nh dh)')
        queries = queries + self.spatial_out_proj(attn_out)
        queries = queries + _film_ffn(
            queries, self.spatial_ffn_norm, self.spatial_ffn_mlp0, self.spatial_ffn_act,
            self.spatial_ffn_drop, self.spatial_ffn_mlp1, self.spatial_film, spatial_noise,
        )

        # ---- temporal self-attn (per joint) ----
        temporal_noise = None
        if noise_emb is not None:
            temporal_noise = noise_emb[:, None, :].expand(batch, J, -1).reshape(batch * J, -1)

        # (B*T, J, D) → (B, T, J, D) → (B, J, T, D) → (B*J, T, D)
        q_bt = queries.reshape(batch, time_q, J, -1).permute(0, 2, 1, 3).reshape(batch * J, time_q, -1)
        norm_full = self.temporal_norm(q_bt)
        if self.temporal_stride > 1:
            # Asymmetric: Q@T × KV@(T/stride). Stride before KV projection so
            # both projection and attention matmul are stride× cheaper.
            s = self.temporal_stride
            q = rearrange(self.temporal_q_proj(norm_full),
                          'b t (nh dh) -> b nh t dh', nh=nh)
            norm_strided = norm_full[:, ::s, :]  # view, no copy
            kv = self.temporal_kv_proj(norm_strided)
            k, v = rearrange(kv, 'b t (two nh dh) -> two b nh t dh', two=2, nh=nh)
            # RoPE positions: Q at full T, K at strided positions [0, s, 2s, ...].
            # We disable RoPE here when strided (different lengths break the
            # standard rotate_queries_or_keys contract); position info still
            # comes from the spatial cross-attn and surrounding decoder blocks.
        else:
            qkv = self.temporal_qkv_proj(norm_full)
            q, k, v = rearrange(qkv, 'b t (three nh dh) -> three b nh t dh', three=3, nh=nh)
            if self.rotary_emb is not None:
                q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
                k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        attn_out = _sdpa(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b nh t dh -> b t (nh dh)')
        q_bt = q_bt + self.temporal_out_proj(attn_out)
        q_bt = q_bt + _film_ffn(
            q_bt, self.temporal_ffn_norm, self.temporal_ffn_mlp0, self.temporal_ffn_act,
            self.temporal_ffn_drop, self.temporal_ffn_mlp1, self.temporal_film, temporal_noise,
        )
        # back to (B*T, J, D)
        queries = q_bt.reshape(batch, J, time_q, -1).permute(0, 2, 1, 3).reshape(BT, J, -1)
        return queries


# ---------------------------------------------------------------------------
# Pose2DTransformerDecoder
# ---------------------------------------------------------------------------


class Pose2DTransformerDecoder(nn.Module):
    """2D pose decoder — operates at full T with 8x8 (64) spatial tokens.

    Pipeline:
      1. encode_image(): patch features → 64 spatial tokens (cacheable across
         use_optimal_2d's noisy + clean passes).
      2. Pose2DBlock × N: spatial cross-attn (joints → image) + temporal
         self-attn per joint, with FiLM(noise_emb) modulation.
      3. pose_head / confidence_head: per-joint (x, y) and confidence.

    Two query modes:
      - regression_mode=True : learnable joint queries (no flow matching).
      - regression_mode=False: queries from noisy 2D joints (flow matching).
    """

    def __init__(
        self,
        patch_feat_channels: int = 768,
        d_latent: int = 256,
        num_joints: int = 21,
        num_layers: int = 4,
        num_heads: int = 4,
        d_feedforward: int = 1024,
        dropout_p: float = 0.1,
        global_feat_dim: int = 64,
        activation: str = "gelu",
        d_cond: int = 7,
        d_noise_emb: int = 1024,
        regression_mode: bool = False,
        grad_checkpoint: bool = False,
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.d_latent = d_latent
        self.global_feat_dim = global_feat_dim
        self.regression_mode = regression_mode
        # Gradient checkpointing trades ~20% compute for ~60% activation memory.
        # Only worth it when pose2d_d_latent is large (e.g., 1024).
        self.grad_checkpoint = grad_checkpoint
        self.temporal_stride = temporal_stride

        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        # ---- Image encoder (with 16x16 → 8x8 spatial downsample) ----
        self.latent_cnn = LatentCNNEncoder(
            patch_feat_channels=patch_feat_channels,
            d_out=d_latent,
            global_feat_dim=global_feat_dim,
            activation=activation,
            downsample=True,
        )

        # ---- Decoder blocks ----
        num_blocks = max(1, num_layers // 2)
        self.blocks = nn.ModuleList([
            Pose2DBlock(
                d_latent=d_latent,
                d_noise_emb=d_noise_emb,
                num_heads=num_heads,
                d_feedforward=d_feedforward,
                dropout_p=dropout_p,
                activation=activation,
                temporal_stride=temporal_stride,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_latent)

        # ---- Output heads ----
        self.pose_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 1),
            nn.Sigmoid(),
        )

        # ---- Query source ----
        if regression_mode:
            # Pure direct-regression: learnable per-joint queries.
            self.learnable_joint_queries = nn.Parameter(
                torch.randn(1, num_joints, d_latent) * 0.02
            )
        else:
            # Flow matching: project (x, y) of noisy joints to query space.
            self.noisy_joint_proj = nn.Linear(2, d_latent)

        # Per-joint identity embedding — gives each of J queries a unique
        # signature so spatial cross-attn can specialise per joint. Tiny init
        # so existing checkpoints (which lack this param) still work via
        # strict=False loading; the model can then learn to use it.
        self.joint_identity_embed = nn.Parameter(
            torch.randn(1, num_joints, d_latent) * 0.02
        )

        # Global feat → query bias (also keeps global_proj in the wrapped
        # forward for DDP grad-hook tracking).
        self.global_feat_to_query = nn.Linear(global_feat_dim, d_latent)
        nn.init.zeros_(self.global_feat_to_query.weight)
        nn.init.zeros_(self.global_feat_to_query.bias)

        # make_cond → image-token bias (added to memory).
        self.cond_to_tokens = nn.Sequential(
            nn.Linear(d_cond, d_latent),
            Activation(),
            nn.Linear(d_latent, d_latent),
        )

        # bbox mask embedding: (B*T, 1, H, W) → (B*T, 768, 16, 16) additive.
        mask_embed_dim = 8
        self.mask_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(1, mask_embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, mask_embed_dim),
            Activation(),
            nn.Conv2d(mask_embed_dim, patch_feat_channels, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # CNN feature extraction (cacheable)
    # ------------------------------------------------------------------

    def encode_image(
        self,
        img_patch_feat: Float[Tensor, "batch time 768 16 16"],
        bbox_mask: Float[Tensor, "batch 1 h w"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract image features. Cacheable across noisy/clean passes.

        Returns:
            tokens     : (B*T, 64, d_latent)  — spatial tokens after downsample.
            global_feat: (B,   T, global_feat_dim) — pooled global feature.
        """
        batch, time = img_patch_feat.shape[:2]
        BT = batch * time
        x = img_patch_feat.reshape(BT, *img_patch_feat.shape[2:])

        # bbox mask additive bias on the patch features.
        if bbox_mask is None:
            bbox_mask = torch.zeros(batch, 1, 16, 16, device=x.device)
        
        # Handle both per-frame (B, T, 1, H, W) and global (B, 1, H, W) bbox_mask
        if bbox_mask.dim() == 5:
            # Per-frame bbox: (B, T, 1, H, W) -> (BT, 1, H, W)
            bbox_exp = bbox_mask.reshape(BT, 1, bbox_mask.shape[3], bbox_mask.shape[4])
        else:
            # Global bbox: (B, 1, H, W) -> expand to (B, T, 1, H, W) -> (BT, 1, H, W)
            h, w = bbox_mask.shape[2], bbox_mask.shape[3]
            bbox_exp = bbox_mask[:, None].expand(batch, time, 1, h, w).reshape(BT, 1, h, w)
        x = x + self.mask_embed(bbox_exp)

        # CNN + global pool + spatial downsample.
        cnn_feat = self.latent_cnn.conv_layers(x)                   # (BT, d_latent, 16, 16)
        global_feat = self.latent_cnn.global_pool(cnn_feat).squeeze(-1).squeeze(-1)
        global_feat = self.latent_cnn.global_proj(global_feat)
        global_feat = global_feat.reshape(batch, time, self.global_feat_dim)

        ds_feat = self.latent_cnn.spatial_downsample(cnn_feat)       # (BT, d_latent, 8, 8)
        tokens = ds_feat.flatten(2).transpose(1, 2)                  # (BT, 64, d_latent)
        tokens = tokens + self.latent_cnn.spatial_pos_embed
        return tokens, global_feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img_patch_feat: Float[Tensor, "batch time 768 16 16"],
        noisy_joint_2d: Float[Tensor, "batch time num_joints 2"] | None = None,
        cond: Float[Tensor, "batch_time d_cond"] | None = None,
        bbox_mask: Float[Tensor, "batch 1 h w"] | None = None,
        noise_emb: Float[Tensor, "batch d_noise_emb"] | None = None,
        cached_tokens: Tensor | None = None,
        cached_global_feat: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns (pose_2d, confidence, joint_feat, global_feat, tokens)."""
        batch, time = img_patch_feat.shape[:2]
        BT = batch * time

        # 1. Image features (use cache if available).
        if cached_tokens is not None:
            tokens, global_feat = cached_tokens, cached_global_feat
        else:
            tokens, global_feat = self.encode_image(img_patch_feat, bbox_mask)

        # 2. Add make_cond bias to image tokens.
        if cond is not None:
            tokens = tokens + self.cond_to_tokens(cond)[:, None, :]

        # 3. Build joint queries.
        if self.regression_mode:
            queries = self.learnable_joint_queries.expand(BT, -1, -1).contiguous()
        else:
            if noisy_joint_2d is None:
                noisy_joint_2d = torch.zeros(batch, time, self.num_joints, 2, device=tokens.device)
            queries = self.noisy_joint_proj(noisy_joint_2d.reshape(BT, self.num_joints, 2))

        # Add per-joint identity embedding (unique signature per joint).
        queries = queries + self.joint_identity_embed

        # Add global feat as a per-frame context bias.
        gf_bias = self.global_feat_to_query(global_feat.reshape(BT, self.global_feat_dim))
        queries = queries + gf_bias[:, None, :]

        # 4. Pose2DBlocks (spatial cross + temporal self), optional grad checkpointing.
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                queries = checkpoint(block, queries, tokens, batch, time, noise_emb,
                                     use_reentrant=False)
            else:
                queries = block(queries, tokens, batch, time, noise_emb=noise_emb)

        queries = self.final_norm(queries)

        # 5. Decode outputs.
        pose_2d = self.pose_head(queries).reshape(batch, time, self.num_joints, 2)
        confidence = self.confidence_head(queries).reshape(batch, time, self.num_joints, 1)
        joint_feat = queries.reshape(batch, time, self.num_joints, self.d_latent)
        return pose_2d, confidence, joint_feat, global_feat, tokens


# ---------------------------------------------------------------------------
# GlobalPoseHead (optional)
# ---------------------------------------------------------------------------


class GlobalPoseHead(nn.Module):
    """Predict global translation + 3x3 orientation from shared spatial tokens.

    Uses a single learnable global query that cross-attends to the image tokens
    via Pose2DBlock (J=1), then heads predict k frames per compressed timestep
    so the total temporal output is at full T.
    """

    def __init__(
        self,
        d_latent: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        d_feedforward: int = 1024,
        dropout_p: float = 0.1,
        d_cond: int = 7,
        d_noise_emb: int = 1024,
        activation: str = "gelu",
        temporal_k: int = 4,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.temporal_k = temporal_k
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.global_query = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)

        num_blocks = max(1, num_layers // 2)
        self.blocks = nn.ModuleList([
            Pose2DBlock(
                d_latent=d_latent,
                d_noise_emb=d_noise_emb,
                num_heads=num_heads,
                d_feedforward=d_feedforward,
                dropout_p=dropout_p,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_latent)

        # Each head predicts k frames per query → reshape gives full T.
        self.translation_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 3 * temporal_k),
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 9 * temporal_k),
        )
        self.feat_head = nn.Linear(d_latent, d_latent * temporal_k)

        # Project noisy translation/orientation into query space (for flow matching).
        self.noisy_trans_proj = nn.Linear(3, d_latent)
        self.noisy_ori_proj = nn.Linear(9, d_latent)

        # make_cond → query bias.
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_latent),
            Activation(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        tokens: Float[Tensor, "batch_time num_tokens d_latent"],
        batch: int,
        time: int,
        cond: Float[Tensor, "batch_time d_cond"] | None = None,
        noisy_translation: Float[Tensor, "batch time 3"] | None = None,
        noisy_ori_mat: Float[Tensor, "batch time 9"] | None = None,
        noise_emb: Float[Tensor, "batch d_noise_emb"] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (pred_trans, pred_ori, global_pose_feat) at full T (=time*k)."""
        BT = batch * time
        query = self.global_query.expand(BT, -1, -1)  # (BT, 1, D)

        # Condition on noisy inputs.
        if noisy_translation is None:
            noisy_translation = torch.zeros(batch, time, 3, device=query.device)
        if noisy_ori_mat is None:
            noisy_ori_mat = torch.zeros(batch, time, 9, device=query.device)
        query = (
            query
            + self.noisy_trans_proj(noisy_translation.reshape(BT, 3))[:, None, :]
            + self.noisy_ori_proj(noisy_ori_mat.reshape(BT, 9))[:, None, :]
        )

        if cond is not None:
            query = query + self.cond_proj(cond)[:, None, :]

        for block in self.blocks:
            query = block(query, tokens, batch, time, noise_emb=noise_emb)

        query = self.final_norm(query).squeeze(1)  # (BT, D)

        k = self.temporal_k
        pred_trans = self.translation_head(query).reshape(batch, time * k, 3)
        pred_ori = self.orientation_head(query).reshape(batch, time * k, 9)
        global_pose_feat = self.feat_head(query).reshape(batch, time * k, self.d_latent)
        return pred_trans, pred_ori, global_pose_feat
