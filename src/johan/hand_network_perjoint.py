"""Per-joint tokenized 3D hand denoiser with triple-attention blocks.

Per-frame layout:
  token 0      = mano_betas               (identity / shape, with Fourier-encoded input)
  tokens 1..15 = per-joint rotation       (pose)

Conditioning routing:
  * Cross-attention to 8x8 image-feature patch tokens — every joint
    and the betas token query the visual KV in each trunk round.
  * 6 fingertip 2D feats (sum-pooled) → added to the betas token.
  * Base cond (palm pose / side / intrinsics) → added to the betas
    token.
  * Per-joint non-aligned 2D feat → added to each joint token.
  * Per-joint aligned 2D feat → concatenated with rotation at the
    joint input encoder.

Confidence handling: 2D-pose confidence values below 0.3 are forced
to zero before they multiply the per-joint 2D features (acts as a
hard occlusion gate).

Trunk: trunk_rounds × TripleAttentionBlock = (
  1 spatial cross-attn with image patches,
  1 spatial self-attn over the 16 token axis,
  1 temporal self-attn over T.
).
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from .hand_network import (
    HandDenoiseTraj,
    HandDenoiserConfig,
    fourier_encode,
)
from .network import TransformerBlock, TransformerBlockConfig, zero_module


class TimestepEmbedder(nn.Module):
    """DiT-style timestep embedding: sinusoidal frequency features of the integer
    timestep followed by a 2-layer MLP. Called with the same 0-based index as the
    legacy nn.Embedding table (``t - 1``) so it is a drop-in replacement."""

    def __init__(self, d_out: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d_out), nn.SiLU(), nn.Linear(d_out, d_out))

    def forward(self, t_index: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(self.max_period)
                          * torch.arange(half, device=t_index.device, dtype=torch.float32) / half)
        args = t_index.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


def make_time_embed(kind: str, max_t: int, d_noise_emb: int) -> nn.Module:
    if kind == "table":
        return nn.Embedding(max_t, d_noise_emb)
    if kind == "sinusoidal":
        return TimestepEmbedder(d_noise_emb)
    raise ValueError(f"unknown time_embed {kind!r}; expected 'table' or 'sinusoidal'")


@dataclass(frozen=True)
class HandDenoiserPerJointConfig(HandDenoiserConfig):
    """Same as HandDenoiserConfig, plus per-joint trunk knobs.

    Supports two 2D-to-3D conditioning modes:
      * **Regress mode** (pose2d_regression=True): 2D decoder predicts 2D
        joints deterministically via learned queries. Single-stage inference.
      * **Opt_2d mode** (use_optimal_2d=True, pose2d_regression=False): 2D
        decoder uses flow-matching to denoise 2D joints iteratively, then the
        clean 2D is used as 3D conditioning. Two-stage inference: first call
        `denoise_2d_step` in a loop, then call forward with `clean_joint_2d`.
    """
    trunk_rounds: int = 6
    """How many TripleAttentionBlock rounds in the trunk."""
    betas_fourier_freqs: int = 4
    """Fourier-frequency count for betas encoding. Per-coord dim becomes
    1 + 2*F; betas input dim = 10 * (1 + 2*F)."""
    image_patch_dim: int = 64
    """Patches per frame from encode_image (8x8=64)."""
    conf_threshold: float = 0.3
    """Per-joint 2D-confidence values below this are zeroed before
    multiplying the joint 2D features."""
    use_perjoint_pose2d: bool = True
    """Use the new triple-attention 2D decoder (Pose2DPerJointDecoder)
    instead of the original Pose2DTransformerDecoder."""
    pose2d_full_res_patches: bool = False
    """Option 8: keep the full GxG image patch tokens (G = image_size//16, so
    16x16=256 at 256px) for the 2D decoder cross-attention instead of
    downsampling to (G/2)x(G/2). Higher spatial resolution for localization."""
    image_size: int = 256
    """Input image resolution fed to DINO (square). The 512 WebDataset copies
    can be used at native resolution by setting this to 512, which makes the
    DINO patch grid 32x32 instead of 16x16. NOTE: this changes the shape of the
    learned spatial_pos_embed, so it is NOT resume-compatible across values —
    a change requires training from scratch."""
    crop_feat_for_3d: bool = False
    """Reversed dual-image routing: the CROPPED-hand features feed the 3D
    trunk (patch cross-attention + global-feat cond token) while the 2D
    decoder uses ONLY the full-image features (no crop gf_bias). Requires
    the training loop to pass img_feat_crop (crop_global_feat: true)."""
    pose2d_self_conditioning: bool = False
    """Option 10: self-conditioning for the flow-matching 2D decoder. The
    decoder additionally consumes its own previous x0 estimate; during
    training half the steps do a no-grad pre-pass to produce it. No-op in
    regression mode."""
    pose2d_heatmap_head: bool = False
    """Option 1: replace the direct (x,y) regression head with a soft-argmax
    over a fixed GxG grid in the output [-1,1]^2 space. Smoother gradients,
    no Tanh saturation, multi-modal-capable."""
    pose2d_heatmap_grid: int = 32
    """Grid resolution G for the heatmap head (G x G anchors)."""
    use_mmdit: bool = False
    """v3 architecture: replace the factored TripleAttentionBlock trunk (in BOTH
    the 2D decoder and the 3D denoiser) with MM-DiT blocks — two streams
    (pose tokens + image-patch tokens) in joint attention, per-stream AdaLN-Zero
    + gated MLP, and 3D axial RoPE over (time, row, col). See mmdit_block.py."""
    adaln_bounded: bool = False
    """Bounded AdaLN modulation (tanh on scale/gate, normalised pass-through)
    for use_adaln_cond; same parameters, same identity at zero modulation.
    See TripleAttentionBlock.__init__ for the instability it fixes."""
    time_embed: str = "table"
    """Timestep embedding (both the 3D denoiser and the 2D decoder):
    "table"      -- learned nn.Embedding(max_t, d_noise_emb), one free row per integer
                    timestep (legacy; rows are uncorrelated, and on ViT-L adalnzero_32g a
                    few learned extreme AdaLN modulations that caused every gradient spike).
    "sinusoidal" -- conventional DiT TimestepEmbedder: sinusoidal frequency features of t
                    -> Linear -> SiLU -> Linear, so nearby timesteps embed nearby.
    The hand side stays a 2-entry learned embedding (DiT's LabelEmbedder convention for a
    class label; a Linear on a 0/1 input spans exactly the same two vectors)."""
    adaln_zero: bool = False
    """DiT AdaLN-Zero (pre-norm gated residual sub-layers, zero gates, final
    modulated LayerNorm before the heads) for use_adaln_cond, in BOTH the 3D
    trunk and the 2D decoder. Different architecture: train from scratch.
    Takes precedence over adaln_bounded."""
    use_adaln_cond: bool = False
    """AdaLN-Zero conditioning on the TripleAttention trunk for BOTH the
    timestep and the hand side.

    Baseline (`noise_conditioning: token`) delivers the timestep as ONE additive
    bias on every pose token before the trunk, and the hand side only as an
    additive term on the betas token -- no per-layer, multiplicative signal.
    With this flag a conditioning vector

        c = SiLU( Linear(noise_emb) + SideEmbedding(mano_side) )

    drives per-block (shift, scale, gate) for all four sub-layers of every
    TripleAttentionBlock (cross-attn, cross-FFN, spatial, temporal).

    The modulation projection is ZERO-INITIALISED and the residuals use
    (1 + gate), so at step 0 the network is numerically IDENTICAL to the
    token-additive baseline. That keeps warm-starting from an existing
    per-joint checkpoint exact: the extra parameters are additive-only and
    start as no-ops. The additive noise token is kept, not replaced."""
    zero_bbox_mask: bool = False
    """Ablation: force the bbox_mask conditioning to all-zeros (the 'no box'
    mask_embed) in BOTH training and inference, so the model must localize from
    the image alone. Tests whether removing the bbox cue at train time yields a
    less bbox-sensitive (more deployment-robust) model. Gated in
    Pose2DPerJointDecoder.encode_image."""
    bbox_cfg_mask_prob: float = 0.0
    """CFG-style PER-FRAME bbox-condition dropout probability (train-time only):
    each frame of a sample independently drops its bbox cue (frame t can be
    masked while t+1 is visible); image/video features are never masked. 0
    disables. Requires zero_bbox_mask false to have any effect."""
    coord_range: float = 1.0
    """OUTPUT RANGE of the Tanh 2D coord head (user, 2026-09-10). The head is
    hard-bounded to [-1,1], so the model can never place a joint outside the box
    it is given (measured: 0.0% of predictions escape the box, max |uv_norm|
    0.988). Under cfgbox the normalisation box is jittered
    (bbox_jitter_affects_norm) and no longer contains every joint, so targets
    legitimately exceed 1 and saturate the Tanh into vanishing gradients. Setting
    k>1 scales the head to [-k,k] so those targets stay learnable. 1.0 = legacy."""
    pure_3d: bool = False
    flow_time_convention: str = "diffusion"
    """Flow matching time orientation for the integer noise-embedding index.
    'diffusion' (default, back-compat): index = t_frac*max_t (high index = noise).
    'flow': index = (1-t_frac)*max_t = data-fraction s, so the time axis runs
    t=0 noise -> t=max_t data (standard rectified-flow). noise_emb_table is a
    learned nn.Embedding so the flip only relabels rows; training (training_loss)
    and inference (inference_hand) read this flag and apply the SAME mapping so
    the round trip stays consistent."""
    pass_2d_joint_tok: bool = False
    """Pass the 2D decoder's post-trunk per-joint feature tokens (joint_tok,
    dim=pose2d_d_latent) into the 3D denoiser as a rich 2D-pose feature, ADDED to
    the existing coordinate-derived joint_2d_feat (per_joint_2d_encoder(fourier(
    base_2d))). Gives the 3D trunk the 2D decoder's learned per-joint features
    (which already attended to the image) on top of the bare 2D coordinates.
    A nn.Linear(pose2d_d_latent -> d_2d_feat) projects them. Factored trunk only."""

    @property
    def image_patch_dim_eff(self) -> int:
        """Patches per frame actually produced by encode_image."""
        return (self.image_size // 16) ** 2 if self.pose2d_full_res_patches else self.image_patch_dim

    @property
    def is_opt_2d_mode(self) -> bool:
        """True if configured for two-stage opt_2d inference."""
        return self.use_optimal_2d and not self.pose2d_regression

    @property
    def is_regress_mode(self) -> bool:
        """True if configured for single-stage regression inference."""
        return self.pose2d_regression


def _activation(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU()


# ---------------------------------------------------------------------------
# Triple-attention block: cross-attn(visual) → spatial self-attn → temporal.
# ---------------------------------------------------------------------------
class TripleAttentionBlock(nn.Module):
    """One trunk round on tokens shaped (B, T, S, D).

    Step 1: per-frame cross-attention against image-feature patches.
    Step 2: per-frame self-attention across the S=16 token axis.
    Step 3: per-token self-attention across the T axis.

    Steps 2 and 3 reuse the existing TransformerBlock (RoPE-capable).
    Step 1 is a thin nn.MultiheadAttention + residual + LayerNorm + FFN,
    so it can attend a fixed P=64 patches.
    """

    def __init__(self, d_latent: int, d_feedforward: int, n_heads: int,
                 dropout_p: float, activation: str, d_noise_emb: int,
                 use_rope: bool, use_adaln: bool = False,
                 adaln_bounded: bool = False, adaln_zero: bool = False):
        super().__init__()
        # adaln_zero (2026-09-22, user: "implement the real adaln-zero"): DiT's
        # AdaLN-Zero. Six PRE-NORM residual sub-layers (cross-attn, cross-FFN,
        # spatial attn, spatial MLP, temporal attn, temporal MLP), each
        #     x = x + gate * f(LN(x) * (1 + scale) + shift)
        # with (shift, scale, gate) = Linear(SiLU(c)) zero-initialised, so every
        # gate starts at 0: each block is the IDENTITY MAP at init and the
        # residual stream is never multiplied, only added to. This is a
        # different architecture from the post-norm baseline (the base
        # TransformerBlock's attention/MLP/LayerNorm parameters are reused in
        # pre-norm order), so it trains from scratch. Compare the original
        # "identity to the un-modulated baseline" formulation below.
        self.adaln_zero = bool(adaln_zero) and bool(use_adaln)
        # Original: 4 sub-layers x (shift, scale, gate). Zero-init so every
        # shift/scale is 0 and every gate enters as (1 + 0) = 1, i.e. exactly
        # the unmodulated residual at step 0.
        self.adaln_mod = (
            zero_module(nn.Linear(d_latent, (18 if self.adaln_zero else 12) * d_latent, bias=True))
            if use_adaln else None
        )
        # adaln_bounded (2026-09-22): the original formulation multiplies the
        # UN-NORMALISED residual stream by (1 + scale) and adds
        # (1 + gate) * (Block(x_in) - x_in), whose -gate * x_in term passes the raw
        # stream through un-normalised. The trunk blocks are post-norm, so
        # nothing renormalises it: once the learned modulation for a band of
        # timesteps exceeds |1|, the stream grows multiplicatively through the
        # spatial -> temporal sub-layers and across blocks. Replaying the
        # adaln_32g run's captured error batches showed exactly that: at
        # t_frac = 0.9 the activations went from ~50 to 3e5 by trunk.1 and the
        # attention backward returned NaN (fp32 too), with ~17% of optimizer
        # steps skipped on non-finite gradients. Bounded mode keeps the same
        # parameters and the same identity at zero modulation, but squashes
        # every scale and gate through tanh and normalises the pass-through
        # (parameter-free LayerNorm), so the stream can only grow additively.
        self.adaln_bounded = bool(adaln_bounded)
        # --- Step 1: cross-attn ---
        self.cross_ln = nn.LayerNorm(d_latent)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_latent, num_heads=n_heads, dropout=dropout_p,
            batch_first=True,
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_feedforward),
            _activation(activation),
            nn.Linear(d_feedforward, d_latent),
            nn.Dropout(dropout_p),
        )

        def _block() -> TransformerBlock:
            return TransformerBlock(TransformerBlockConfig(
                d_latent=d_latent, d_noise_emb=d_noise_emb,
                d_feedforward=d_feedforward, n_heads=n_heads,
                dropout_p=dropout_p, activation=activation,
                include_xattn=False,
                use_rope_embedding=use_rope,
                use_film_noise_conditioning=False,
                xattn_mode="kv_from_cond_q_from_x",
            ))

        # Step 2: spatial self-attn over S=16.
        self.spatial_block = _block()
        # Step 3: temporal self-attn over T.
        self.temporal_block = _block()

    def forward(
        self,
        x: Float[Tensor, "batch time S d"],
        img_patches: Float[Tensor, "batch time P d"],
        noise_emb: Float[Tensor, "batch d_noise"],
        cond: Float[Tensor, "batch d"] | None = None,
    ) -> Float[Tensor, "batch time S d"]:
        B, T, S, D = x.shape
        P = img_patches.shape[2]

        if self.adaln_zero and cond is not None:
            return self._forward_adaln_zero(x, img_patches, cond)

        if self.adaln_mod is not None and cond is not None:
            m = self.adaln_mod(cond)                      # (B, 12D)
            (sh_a, sc_a, g_a, sh_f, sc_f, g_f,
             sh_s, sc_s, g_s, sh_t, sc_t, g_t) = m.chunk(12, dim=-1)
            if self.adaln_bounded:
                # |1 + scale| <= 2 and |gate| <= 1 (tanh); shifts stay free.
                sc_a, g_a, sc_f, g_f, sc_s, g_s, sc_t, g_t = (
                    torch.tanh(v) for v in (sc_a, g_a, sc_f, g_f, sc_s, g_s, sc_t, g_t))
        else:
            sh_a = sc_a = g_a = sh_f = sc_f = g_f = None
            sh_s = sc_s = g_s = sh_t = sc_t = g_t = None

        # --- Step 1: cross-attn ---
        q = self.cross_ln(x)
        if sh_a is not None:
            q = q * (1 + sc_a[:, None, None, :]) + sh_a[:, None, None, :]
        q = q.reshape(B * T, S, D)
        kv = img_patches.reshape(B * T, P, D)
        a, _ = self.cross_attn(q, kv, kv)
        a = a.reshape(B, T, S, D)
        x = x + (a if g_a is None else (1 + g_a[:, None, None, :]) * a)

        # cross-FFN: modulate AFTER its internal LayerNorm (index 0) so the
        # norm cannot wash the shift/scale away; parameter names are untouched.
        h = self.cross_ffn[0](x)
        if sh_f is not None:
            h = h * (1 + sc_f[:, None, None, :]) + sh_f[:, None, None, :]
        for _m in self.cross_ffn[1:]:
            h = _m(h)
        x = x + (h if g_f is None else (1 + g_f[:, None, None, :]) * h)

        # --- Step 2: spatial self-attn over S (per-frame independent) ---
        xs = x.reshape(B * T, S, D)
        # TransformerBlock(noise_emb) expects (batch, d_noise) — broadcast
        # the per-clip noise level to (B*T, d_noise).
        noise_bt = noise_emb.repeat_interleave(T, dim=0)
        if sh_s is not None:
            # TransformerBlock is post-norm, so modulate its INPUT and gate the
            # block's net contribution: identity when scale=shift=gate=0.
            sh_bt = sh_s.repeat_interleave(T, dim=0)[:, None, :]
            sc_bt = sc_s.repeat_interleave(T, dim=0)[:, None, :]
            g_bt = g_s.repeat_interleave(T, dim=0)[:, None, :]
            xs_in = xs * (1 + sc_bt) + sh_bt
            out = self.spatial_block(xs_in, None, noise_emb=noise_bt)
            if self.adaln_bounded:
                # out is the block's post-norm output (unit scale); the
                # pass-through is normalised too, so the gate can only mix two
                # unit-scale tensors instead of leaking -gate * x_in.
                xs = out + g_bt * (out - F.layer_norm(xs_in, (D,)))
            else:
                xs = xs_in + (1 + g_bt) * (out - xs_in)
        else:
            xs = self.spatial_block(xs, None, noise_emb=noise_bt)
        x = xs.reshape(B, T, S, D)

        # --- Step 3: temporal self-attn over T (per-token independent) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        noise_bs = noise_emb.repeat_interleave(S, dim=0)
        if sh_t is not None:
            sh_bs = sh_t.repeat_interleave(S, dim=0)[:, None, :]
            sc_bs = sc_t.repeat_interleave(S, dim=0)[:, None, :]
            g_bs = g_t.repeat_interleave(S, dim=0)[:, None, :]
            xt_in = xt * (1 + sc_bs) + sh_bs
            out = self.temporal_block(xt_in, None, noise_emb=noise_bs)
            if self.adaln_bounded:
                xt = out + g_bs * (out - F.layer_norm(xt_in, (D,)))
            else:
                xt = xt_in + (1 + g_bs) * (out - xt_in)
        else:
            xt = self.temporal_block(xt, None, noise_emb=noise_bs)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)

        return x


    @staticmethod
    def _prenorm_sublayers(blk: TransformerBlock, x: Tensor, mods) -> Tensor:
        """DiT-style pre-norm attention + MLP residual sub-layers built from the
        parameters of a (post-norm) TransformerBlock. mods = per-row
        (shift1, scale1, gate1, shift2, scale2, gate2), each (rows, 1, D)."""
        b1, g1, a1, b2, g2, a2 = mods
        h = blk.layernorm1(x) * (1 + g1) + b1
        x = x + a1 * blk._sattn(h, None)
        h = blk.layernorm2(x) * (1 + g2) + b2
        h = blk.mlp1(blk.dropout(blk.activation(blk.mlp0(h))))
        return x + a2 * h

    def _forward_adaln_zero(self, x: Tensor, img_patches: Tensor, cond: Tensor) -> Tensor:
        B, T, S, D = x.shape
        P = img_patches.shape[2]
        mod = self.adaln_mod(cond)
        # HAND_ADALN_CLAMP=c (diagnostic, 2026-10-10): clip every scale and gate to [-c, c].
        # A few learned timestep rows of the ViT-L 3D trunk reach |scale| 267 / |gate| 324
        # (median 7 / 13) and cause all attributed gradient spikes; this tests whether
        # capping only those outliers is harmless. 0 = off (default, unchanged behaviour).
        _clamp = float(os.environ.get("HAND_ADALN_CLAMP", "0") or 0)
        if _clamp > 0:
            m3 = mod.unflatten(-1, (6, 3, D))  # (shift, scale, gate) x 6 sub-layers
            mod = torch.cat([m3[..., :1, :], m3[..., 1:, :].clamp(-_clamp, _clamp)], dim=-2).flatten(-3)
        (b_a, g_a, a_a, b_f, g_f, a_f,
         b_s1, g_s1, a_s1, b_s2, g_s2, a_s2,
         b_t1, g_t1, a_t1, b_t2, g_t2, a_t2) = mod.chunk(18, dim=-1)

        def e4(v):
            return v[:, None, None, :]

        # 1. cross-attention onto the image patches (pre-norm, gated)
        q = self.cross_ln(x) * (1 + e4(g_a)) + e4(b_a)
        kv = img_patches.reshape(B * T, P, D)
        a, _ = self.cross_attn(q.reshape(B * T, S, D), kv, kv)
        x = x + e4(a_a) * a.reshape(B, T, S, D)
        # 2. cross-FFN
        h = self.cross_ffn[0](x) * (1 + e4(g_f)) + e4(b_f)
        for _m in self.cross_ffn[1:]:
            h = _m(h)
        x = x + e4(a_f) * h
        # 3. spatial self-attention over the S tokens of each frame
        xs = x.reshape(B * T, S, D)
        xs = self._prenorm_sublayers(
            self.spatial_block, xs,
            [v.repeat_interleave(T, dim=0)[:, None, :] for v in (b_s1, g_s1, a_s1, b_s2, g_s2, a_s2)])
        x = xs.reshape(B, T, S, D)
        # 4. temporal self-attention over T for each token
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt = self._prenorm_sublayers(
            self.temporal_block, xt,
            [v.repeat_interleave(S, dim=0)[:, None, :] for v in (b_t1, g_t1, a_t1, b_t2, g_t2, a_t2)])
        return xt.reshape(B, S, T, D).permute(0, 2, 1, 3)


class HandDenoiserPerJoint(nn.Module):
    """16-token-per-frame factored-attention denoiser."""

    def __init__(self, config: HandDenoiserPerJointConfig) -> None:
        super().__init__()
        self.config = config
        D = config.d_latent

        # --- Fourier-encoded betas input ----------------------------
        self.betas_fourier_freqs = config.betas_fourier_freqs
        betas_in_dim = 10 * (1 + 2 * config.betas_fourier_freqs)
        self.betas_in = nn.Sequential(
            nn.Linear(betas_in_dim, D), _activation(config.activation),
            nn.Linear(D, D), _activation(config.activation),
            nn.Linear(D, D),
        )

        d_2d_feat = config.d_2d_feat
        self.d_2d_feat = d_2d_feat
        rot_dim = 9 if config.using_mat else 3
        # Joint token input: rotation (rot_dim) + aligned 2D feat (d_2d_feat)
        joint_in_dim = rot_dim + (d_2d_feat if config.using_img_feat and not config.pure_3d else 0)
        self.joint_in_dim = joint_in_dim
        self.joint_in = nn.Sequential(
            nn.Linear(joint_in_dim, D), _activation(config.activation),
            nn.Linear(D, D), _activation(config.activation),
            nn.Linear(D, D),
        )

        # Per-token id embedding (16 = 1 betas + 15 joints).
        self.token_id_emb = nn.Parameter(torch.randn(16, D) * 0.02)

        # --- conditioning projections (additive into specific tokens) ---
        # No-rot fingertip features (sum-pooled) → betas token.
        if not config.pure_3d:
            self.cond_norot_to_betas = nn.Linear(d_2d_feat, D)
        # Base cond (mano_side / palm pose / intrinsics) → betas token.
        self.cond_base_to_betas = nn.Linear(config.d_cond_base, D)
        # Non-aligned per-joint 2D feat → each joint token.
        if not config.pure_3d:
            self.cond_2d_to_joint = nn.Linear(d_2d_feat, D)
        # Global pooled image feature (from pose2d_decoder.encode_image) →
        # per-frame bias on every token. Mirrors how the original
        # HandDenoiser concatenates `cond_global_feat` into `cond_c` before
        # the trunk. Zero-init so loading older checkpoints starts from the
        # same behaviour; respects cfg.zero_img_feat_in_3d_cond at forward.
        self.global_feat_to_token = nn.Linear(config.pose2d_global_feat_dim, D)
        nn.init.zeros_(self.global_feat_to_token.weight)
        nn.init.zeros_(self.global_feat_to_token.bias)

        # --- image patch projection (P=64, pose2d_d_latent → D) -------
        if config.using_img_feat and config.pose2d_d_latent != D:
            self.patch_proj = nn.Linear(config.pose2d_d_latent, D)
        else:
            self.patch_proj = nn.Identity()

        # --- 2D head reuse -----------------------------------------------
        self._build_pose2d_stack(config)

        # --- AdaLN-Zero conditioning source (timestep + hand side) -------
        # Built only when enabled so existing checkpoints keep identical
        # state_dict keys. The per-block projection is the zero-init part;
        # these two just produce c, so they use ordinary init.
        if getattr(config, "use_adaln_cond", False):
            self.adaln_noise_proj = nn.Linear(config.d_noise_emb, D)
            self.adaln_side_emb = nn.Embedding(2, D)
            nn.init.normal_(self.adaln_side_emb.weight, std=0.02)
        else:
            self.adaln_noise_proj = None
            self.adaln_side_emb = None
        # AdaLN-Zero: the residual stream leaves the trunk un-normalised, so
        # (as in DiT) a final modulated LayerNorm precedes the heads.
        self.adaln_zero = bool(getattr(config, "adaln_zero", False)) and bool(getattr(config, "use_adaln_cond", False))
        if self.adaln_zero:
            self.final_norm = nn.LayerNorm(D, elementwise_affine=False)
            self.final_mod = zero_module(nn.Linear(D, 2 * D, bias=True))

        # --- noise embedding --------------------------------------------
        self.noise_emb = make_time_embed(getattr(config, "time_embed", "table"), config.max_t, config.d_noise_emb)
        self.noise_emb_to_token = nn.Linear(config.d_noise_emb, D, bias=False)

        # --- trunk: triple-attention OR MM-DiT (v3) rounds --------------
        self.use_mmdit = getattr(config, "use_mmdit", False)
        if self.use_mmdit:
            from .mmdit_block import MMDiTBlock
            self.trunk = nn.ModuleList([
                MMDiTBlock(
                    d_latent=D, d_feedforward=config.d_feedforward,
                    n_heads=config.num_heads, dropout_p=config.dropout_p,
                    activation=config.activation,
                    d_noise_emb=config.d_noise_emb,
                    use_rope=config.positional_encoding == "rope",
                )
                for _ in range(config.trunk_rounds)
            ])
        else:
            self.trunk = nn.ModuleList([
                TripleAttentionBlock(
                    d_latent=D, d_feedforward=config.d_feedforward,
                    n_heads=config.num_heads, dropout_p=config.dropout_p,
                    activation=config.activation,
                    d_noise_emb=config.d_noise_emb,
                    use_rope=config.positional_encoding == "rope",
                    use_adaln=getattr(config, "use_adaln_cond", False),
                    adaln_bounded=getattr(config, "adaln_bounded", False),
                    adaln_zero=getattr(config, "adaln_zero", False),
                )
                for _ in range(config.trunk_rounds)
            ])

        # --- output heads -----------------------------------------------
        self.betas_out = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), _activation(config.activation),
            nn.Linear(D, D), _activation(config.activation),
            nn.Linear(D, 10),
        )
        self.joint_out = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), _activation(config.activation),
            nn.Linear(D, D), _activation(config.activation),
            nn.Linear(D, rot_dim),
        )

    # ------------------------------------------------------------------
    # Pose2D stack — mirrors HandDenoiser construction.
    # ------------------------------------------------------------------
    def _build_pose2d_stack(self, config: HandDenoiserConfig) -> None:
        if config.pure_3d:
            if config.predict_2Dpose or config.use_optimal_2d or config.pose2d_state_only:
                raise ValueError("pure_3d forbids 2D generation and 2D conditioning modes")
            if config.using_img_feat:
                from .hand2d_perjoint_net import HandImageEncoder
                self.image_encoder = HandImageEncoder(config)
            return
        if config.using_img_feat:
            if getattr(config, "use_perjoint_pose2d", True):
                from .hand2d_perjoint_net import Pose2DPerJointDecoder
                self.pose2d_decoder = Pose2DPerJointDecoder(
                    patch_feat_channels=config.pose2d_patch_feat_channels,
                    d_latent=config.pose2d_d_latent,
                    num_joints=config.pose2d_num_joints,
                    num_layers=config.pose2d_num_layers,
                    num_heads=config.pose2d_num_heads,
                    d_feedforward=config.pose2d_d_feedforward,
                    dropout_p=config.dropout_p,
                    global_feat_dim=config.pose2d_global_feat_dim,
                    activation=config.activation,
                    d_cond=config.d_cond_base,
                    d_noise_emb=config.d_noise_emb,
                    regression_mode=config.pose2d_regression,
                    max_t=config.max_t,
                    time_embed=getattr(config, "time_embed", "table"),
                    # use_adaln_cond drives BOTH generative branches: the 3D
                    # trunk below and this 2D decoder, each off its own noise
                    # level. Gating them on one flag keeps the ablation honest.
                    use_adaln=getattr(config, "use_adaln_cond", False),
                    adaln_bounded=getattr(config, "adaln_bounded", False),
                    adaln_zero=getattr(config, "adaln_zero", False),
                    full_res_patches=getattr(config, "pose2d_full_res_patches", False),
                    self_conditioning=getattr(config, "pose2d_self_conditioning", False),
                    heatmap_head=getattr(config, "pose2d_heatmap_head", False),
                    heatmap_grid=getattr(config, "pose2d_heatmap_grid", 32),
                    patch_grid=getattr(config, "image_size", 256) // 16,
                    conf_in_state=getattr(config, "pose2d_conf_in_state", False),
                    use_mmdit=getattr(config, "use_mmdit", False),
                    zero_bbox_mask=getattr(config, "zero_bbox_mask", False),
                    bbox_cfg_mask_prob=getattr(config, "bbox_cfg_mask_prob", 0.0),
                    coord_range=getattr(config, "coord_range", 1.0),
                )
            else:
                from .hand2d_net import Pose2DTransformerDecoder
                self.pose2d_decoder = Pose2DTransformerDecoder(
                    patch_feat_channels=config.pose2d_patch_feat_channels,
                    d_latent=config.pose2d_d_latent,
                    num_joints=config.pose2d_num_joints,
                    num_layers=config.pose2d_num_layers,
                    num_heads=config.pose2d_num_heads,
                    d_feedforward=config.pose2d_d_feedforward,
                    dropout_p=config.dropout_p,
                    global_feat_dim=config.pose2d_global_feat_dim,
                    activation=config.activation,
                    d_cond=config.d_cond_base,
                    d_noise_emb=config.d_noise_emb,
                    regression_mode=config.pose2d_regression,
                    grad_checkpoint=config.pose2d_grad_checkpoint,
                    temporal_stride=config.pose2d_temporal_stride,
                )
        fourier_2d_dim = 2 * (1 + 2 * config.pose2d_fourier_freqs)
        self.fourier_2d_dim = fourier_2d_dim
        self.pose2d_fourier_freqs = config.pose2d_fourier_freqs
        self.per_joint_2d_encoder = nn.Sequential(
            nn.Linear(fourier_2d_dim, config.d_2d_feat),
            _activation(config.activation),
            nn.Linear(config.d_2d_feat, config.d_2d_feat),
        )
        # pass_2d_joint_tok: project the 2D decoder's post-trunk per-joint feature
        # (dim pose2d_d_latent) to d_2d_feat so it can be added to joint_2d_feat.
        if getattr(config, "pass_2d_joint_tok", False):
            self.joint_tok_2d_proj = nn.Linear(config.pose2d_d_latent, config.d_2d_feat)
        self.register_buffer('rotation_to_output_joint', torch.tensor([
            1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19
        ], dtype=torch.long))
        self.no_rot_joint_idx = torch.tensor([0, 4, 8, 12, 16, 20], dtype=torch.long)

        if config.use_xattn_rot_to_2d:
            self.rot_queries_xattn = nn.Parameter(torch.randn(15, config.d_2d_feat) * 0.02)
            self.xattn_rot_to_2d = nn.MultiheadAttention(
                embed_dim=config.d_2d_feat,
                num_heads=config.xattn_rot_num_heads,
                dropout=config.dropout_p, batch_first=True,
            )
            self.xattn_rot_norm = nn.LayerNorm(config.d_2d_feat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def get_d_state(self) -> int:
        if getattr(self.config, "pose2d_state_only", False):
            # pure-2D: the flow state IS the per-joint 2D pose (+ conf channel(s)).
            # coord_in_ch = 2 + conf_ch (1 BCE / 0 none) on the decoder.
            return self.config.pose2d_num_joints * self.pose2d_decoder.coord_in_ch
        return HandDenoiseTraj.get_packed_dim(self.config.using_mat)

    def denoise_2d_step(
        self,
        noisy_joint_2d: Float[Tensor, "batch time 21 2"],
        t: Float[Tensor, "batch"],
        img_feat: Float[Tensor, "batch time ..."],
        conds,
        rel_palm_pose,
        bbox_mask=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Single denoising step for 2D joints (opt_2d mode Stage 1).

        Returns (pred_2d, confidence, cached_tokens, cached_global_feat).
        Use this in a loop for iterative 2D denoising before calling the
        full forward with clean_joint_2d for Stage 2.
        """
        cfg = self.config
        B, T = noisy_joint_2d.shape[:2]
        cond_base = cfg.make_cond(rel_palm_pose=rel_palm_pose, conds=conds)
        pose2d_cond = cond_base.reshape(B * T, -1)

        cached_tokens, cached_global_feat = self.pose2d_decoder.encode_image(
            img_feat.float(), bbox_mask=bbox_mask,
        )
        pose_2d, confidence, _ = self.pose2d_decoder(
            img_feat.float(),
            noisy_joint_2d=noisy_joint_2d,
            cond=pose2d_cond,
            bbox_mask=bbox_mask,
            cached_tokens=cached_tokens,
            cached_global_feat=cached_global_feat,
            t=t,
        )
        return pose_2d, confidence, cached_tokens, cached_global_feat

    def forward(
        self,
        x_t_packed,
        t,
        *,
        rel_palm_pose,
        project_output_rotmats: bool,
        img_feat=None,
        mask=None,
        cond_dropout_keep_mask=None,
        conds,
        noisy_joint_2d=None,
        clean_joint_2d=None,
        bbox_mask=None,
        noisy_translation=None,
        noisy_ori_mat=None,
        cond_confidence=None,
        cached_tokens=None,
        cached_global_feat=None,
        img_feat_crop=None,
    ):
        cfg = self.config
        D = cfg.d_latent

        _state2d = getattr(cfg, "pose2d_state_only", False)
        if _state2d:
            # pure-2D: x_t_packed IS the noised per-joint 2D state [x, y, (conf)];
            # bypass the MANO unpack/trunk and let the 2D decoder denoise it.
            B, T = x_t_packed.shape[:2]
            x_t = None
            noisy_joint_2d = x_t_packed.reshape(B, T, cfg.pose2d_num_joints,
                                                self.pose2d_decoder.coord_in_ch)
        else:
            x_t = HandDenoiseTraj.unpack(x_t_packed, using_mat=cfg.using_mat, mano_side=conds.mano_side)
            B, T, _, _ = x_t.mano_poses.shape

        # ---- base cond (B, T, d_cond_base) ---------------------------
        cond_base = cfg.make_cond(rel_palm_pose=rel_palm_pose, conds=conds)

        # ---- image patches + optional 2D head ------------------------
        pose_2d = None
        confidence = None
        pred_translation = None
        pred_ori_mat = None
        joint_tok_2d = None   # 2D decoder's post-trunk per-joint feature (pass_2d_joint_tok)
        if cfg.using_img_feat and img_feat is not None:
            image_encoder = self.image_encoder if cfg.pure_3d else self.pose2d_decoder
            pose2d_cond = cond_base.reshape(B * T, -1)
            if cached_tokens is None or cached_global_feat is None:
                cached_tokens, cached_global_feat = image_encoder.encode_image(
                    img_feat.float(), bbox_mask=bbox_mask,
                )
            # Dual-image mode: derive the 2D decoder's global-feature query bias
            # (gf_bias) from the CROPPED hand image, while patches + the 3D cond
            # below keep using the full-image cached_global_feat. No bbox_mask for
            # the crop (the whole crop is the hand region).
            # crop_feat_for_3d REVERSES the routing: crop tokens+gf feed the 3D
            # trunk below, and the 2D decoder sees only full-image features.
            crop_global_feat = None
            crop_tokens = None
            if img_feat_crop is not None:
                crop_tokens, crop_global_feat = image_encoder.encode_image(
                    img_feat_crop.float(), bbox_mask=None,
                )
            crop_for_3d = getattr(cfg, "crop_feat_for_3d", False)
            gf_bias_2d = None if crop_for_3d else crop_global_feat
            # Always run the 2D decoder when predicting 2D. In opt_2d mode the
            # decoder MUST be trained on the (noisy) 2D denoising task — even
            # though clean_joint_2d (GT) is what conditions the 3D trunk — or
            # its flow-matching weights get zero gradient (j2d loss == 0) and
            # Stage-1 inference produces garbage. clean_joint_2d only governs
            # the 3D conditioning below, not whether we predict 2D here.
            if cfg.predict_2Dpose:
                # Only the new Pose2DPerJointDecoder accepts `t` (used in
                # its flow-matching path). The original
                # Pose2DTransformerDecoder doesn't have this kwarg.
                is_perjoint2d = getattr(cfg, "use_perjoint_pose2d", True)
                extra = ({"t": t, "mano_side": conds.mano_side}
                         if is_perjoint2d else {})
                # Option 10: self-conditioning (flow-matching 2D only). Do a
                # no-grad pre-pass to produce an x0 estimate, then feed it back
                # as the self-conditioning signal. Applied every call so train
                # and inference stay consistent (the model always sees a real
                # sc signal rather than zeros).
                sc = None
                if (is_perjoint2d and getattr(cfg, "pose2d_self_conditioning", False)
                        and not cfg.pose2d_regression):
                    with torch.no_grad():
                        sc, _, _ = self.pose2d_decoder(
                            img_feat.float(), noisy_joint_2d=noisy_joint_2d,
                            cond=pose2d_cond, bbox_mask=bbox_mask,
                            cached_tokens=cached_tokens,
                            cached_global_feat=cached_global_feat,
                            sc_joint_2d=None, gf_global_feat=gf_bias_2d, **extra,
                        )
                    sc = sc.detach()
                if is_perjoint2d:
                    extra["sc_joint_2d"] = sc
                _want_jtok = getattr(cfg, "pass_2d_joint_tok", False) and is_perjoint2d
                _dec_out = self.pose2d_decoder(
                    img_feat.float(), noisy_joint_2d=noisy_joint_2d, cond=pose2d_cond,
                    bbox_mask=bbox_mask,
                    cached_tokens=cached_tokens, cached_global_feat=cached_global_feat,
                    gf_global_feat=gf_bias_2d,
                    return_joint_tok=_want_jtok,
                    **extra,
                )
                if _want_jtok:
                    pose_2d, confidence, _g, joint_tok_2d = _dec_out
                else:
                    pose_2d, confidence, _g = _dec_out

                if _state2d:
                    # pure-2D: the decoder's predicted x0 IS the 2D state
                    # [x, y, (conf)] — reassemble it as packed_output and skip the
                    # entire MANO trunk below. conf channels match the GT
                    # gt_2d_state=cat([joint_2d, vis]) built in the loss.
                    packed_output = (
                        torch.cat([pose_2d, confidence], dim=-1)
                        if confidence is not None else pose_2d
                    ).reshape(B, T, -1)
                    return packed_output, pose_2d, confidence, None, None

        # Patch tokens (B, T, P, D) for cross-attention. P is 64 (8x8) or
        # 256 (16x16) depending on pose2d_full_res_patches.
        # crop_feat_for_3d: the 3D trunk consumes CROP tokens + crop global
        # feat (the 2D decoder above already used the full-image features).
        p_dim = cfg.image_patch_dim_eff
        patch_tokens_3d = cached_tokens
        gf_3d = cached_global_feat
        if (cfg.using_img_feat and img_feat is not None
                and getattr(cfg, "crop_feat_for_3d", False) and crop_tokens is not None):
            patch_tokens_3d = crop_tokens
            gf_3d = crop_global_feat
        if patch_tokens_3d is not None:
            patches = patch_tokens_3d.reshape(B, T, p_dim, -1)
            patches = self.patch_proj(patches)
        else:
            # Fallback: zero patches so cross-attn is a no-op.
            patches = torch.zeros(B, T, p_dim, D, device=x_t_packed.device)

        # ---- 2D-pose conditioning ------------------------------------
        if clean_joint_2d is not None:
            base_2d = clean_joint_2d
        elif pose_2d is not None:
            base_2d = pose_2d
        else:
            base_2d = None
        # Ablation: 3D conditioned ONLY on image features — drop all 2D cond.
        # base_2d=None -> joint_2d_feat=None -> aligned/non_aligned/no_rot_2d
        # all None, which the joint-token assembly already handles by zero-pad,
        # so the trunk sees rotation + image patches + global feat only.
        if cfg.pure_3d or getattr(cfg, "ignore_2d_cond", False):
            base_2d = None

        joint_2d_feat = None
        if base_2d is not None:
            # pose2d_absolute: base_2d is in ABSOLUTE pixels, whose magnitude
            # depends on image size and where the hand happens to be. Fourier-
            # encoding that directly would make the 3D branch's conditioning
            # translation- and scale-dependent, so normalise into the box implied
            # by the 2D itself first. Detached and side-clamped for the same
            # reason as in the loss: a collapsed box makes 1/box explode.
            if getattr(cfg, "pose2d_absolute", False):
                _mn = base_2d.min(dim=-2, keepdim=True).values
                _mx = base_2d.max(dim=-2, keepdim=True).values
                _mg = (_mx - _mn) * 0.05
                _b0 = (_mn - _mg).detach()
                _wh = ((_mx + _mg) - (_mn - _mg)).clamp(
                    min=float(getattr(cfg, "pose2d_absolute_min_wh", 0.0625))).detach()
                base_2d = (base_2d - _b0) / _wh * 2.0 - 1.0
            fourier_2d = fourier_encode(
                base_2d.reshape(B * T * 21, 2),
                freqs=self.pose2d_fourier_freqs,
            ).reshape(B, T, 21, -1)
            joint_2d_feat = self.per_joint_2d_encoder(fourier_2d)
            # pass_2d_joint_tok: add the 2D decoder's rich per-joint feature
            # (already attended to the image) on top of the coordinate feature.
            if getattr(cfg, "pass_2d_joint_tok", False) and joint_tok_2d is not None:
                joint_2d_feat = joint_2d_feat + self.joint_tok_2d_proj(joint_tok_2d)
            if getattr(cfg, "pose2d_conf_in_state", False) and confidence is not None:
                # conf_in_state: weight the 2D condition by the predicted confidence
                # (a soft reliability weight, detached; the sigmoid [0,1] value).
                joint_2d_feat = joint_2d_feat * confidence.detach()
            elif cond_confidence is not None:
                # Hard threshold at conf_threshold: anything below → 0.
                conf = cond_confidence
                conf = torch.where(conf >= cfg.conf_threshold, conf, torch.zeros_like(conf))
                joint_2d_feat = joint_2d_feat * conf

        # Aligned / non-aligned 2D feats.
        aligned_2d = None
        non_aligned_2d = None
        no_rot_2d = None
        if joint_2d_feat is not None:
            rot_idx = self.rotation_to_output_joint
            if cfg.use_xattn_rot_to_2d:
                bt = B * T
                kv = joint_2d_feat.reshape(bt, 21, -1)
                q = self.rot_queries_xattn.unsqueeze(0).expand(bt, -1, -1)
                attended, _ = self.xattn_rot_to_2d(q, kv, kv)
                aligned_2d = self.xattn_rot_norm(attended).reshape(B, T, 15, -1)
            else:
                aligned_2d = joint_2d_feat[:, :, rot_idx, :]
            non_aligned_2d = joint_2d_feat[:, :, rot_idx, :]
            no_rot_2d = joint_2d_feat[:, :, self.no_rot_joint_idx.to(joint_2d_feat.device), :].sum(dim=2)

        # ---- input encoding ------------------------------------------
        # Fourier-encode betas before the input MLP.
        betas_raw = x_t.mano_betas.reshape(B * T, 10)
        betas_fourier = fourier_encode(betas_raw, freqs=self.betas_fourier_freqs)
        betas_fourier = betas_fourier.reshape(B, T, -1)
        betas_tok = self.betas_in(betas_fourier)  # (B, T, D)

        pose_input = (x_t.mano_poses_mat if cfg.using_mat else x_t.mano_poses)
        rot_dim = 9 if cfg.using_mat else 3
        joint_raw = pose_input.reshape(B, T, 15, rot_dim)
        if aligned_2d is not None:
            joint_input = torch.cat([joint_raw, aligned_2d], dim=-1)
        else:
            joint_input = F.pad(joint_raw, (0, self.joint_in_dim - rot_dim))
        joint_tok = self.joint_in(joint_input)  # (B, T, 15, D)

        # ---- cond injection ------------------------------------------
        betas_tok = betas_tok + self.cond_base_to_betas(cond_base)
        if no_rot_2d is not None:
            betas_tok = betas_tok + self.cond_norot_to_betas(no_rot_2d)
        if non_aligned_2d is not None:
            joint_tok = joint_tok + self.cond_2d_to_joint(non_aligned_2d)

        # cond dropout regulariser.
        if cond_dropout_keep_mask is not None:
            keep = cond_dropout_keep_mask[:, None, None].float()
            betas_tok = betas_tok * keep
            joint_tok = joint_tok * keep[:, :, :, None]

        # Assemble (B, T, 16, D) + id emb + noise emb.
        x = torch.cat([betas_tok.unsqueeze(2), joint_tok], dim=2)
        x = x + self.token_id_emb[None, None, :, :]
        noise_emb = self.noise_emb(t - 1)
        x = x + self.noise_emb_to_token(noise_emb)[:, None, None, :]

        # Global pooled image feature → per-frame bias on every token
        # (matches the original HandDenoiser which concatenated it into
        # cond_c before the trunk). Honor zero_img_feat_in_3d_cond.
        if gf_3d is not None and not cfg.zero_img_feat_in_3d_cond:
            gf_bias = self.global_feat_to_token(gf_3d)  # (B, T, D)
            x = x + gf_bias[:, :, None, :]

        # ---- AdaLN conditioning vector c = f(timestep, hand side) -----
        adaln_c = None
        if self.adaln_noise_proj is not None:
            # mano_side arrives as (B, T, 1) or (B, 1); it is constant per clip,
            # so collapse to one index per sample. 1 = right, 0 = left.
            _side = conds.mano_side.reshape(x.shape[0], -1)[:, 0]
            _side_idx = (_side > 0.5).long()
            adaln_c = torch.nn.functional.silu(
                self.adaln_noise_proj(noise_emb) + self.adaln_side_emb(_side_idx)
            )

        # ---- trunk (TripleAttention or MM-DiT v3) --------------------
        if self.use_mmdit:
            # MM-DiT threads BOTH streams (pose tokens + image tokens).
            img_stream = patches
            for block in self.trunk:
                x, img_stream = block(x, img_stream, noise_emb)
        else:
            for block in self.trunk:
                x = block(x, patches, noise_emb, cond=adaln_c)
            if getattr(self, "adaln_zero", False) and adaln_c is not None:
                f_shift, f_scale = self.final_mod(adaln_c).chunk(2, dim=-1)
                x = self.final_norm(x) * (1 + f_scale[:, None, None, :]) + f_shift[:, None, None, :]

        # ---- decode --------------------------------------------------
        betas_pred = self.betas_out(x[:, :, 0, :])
        joint_pred = self.joint_out(x[:, :, 1:, :])
        packed_output = torch.cat(
            [betas_pred, joint_pred.reshape(B, T, 15 * rot_dim)],
            dim=-1,
        )
        return packed_output, pose_2d, confidence, pred_translation, pred_ori_mat
