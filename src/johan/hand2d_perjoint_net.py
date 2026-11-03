"""Companion 2D regression decoder for HandDenoiserPerJoint.

Same input/output interface as Pose2DTransformerDecoder in hand2d_net.py
so HandDenoiserPerJoint can swap to this one via a config flag. The
internal trunk uses the same TripleAttentionBlock layout as the 3D
denoiser:
    each round  =  spatial cross-attn(patches)
                  → spatial self-attn over 21 joint tokens
                  → temporal self-attn over T frames.

Regression-only: 21 learnable joint queries replace the noisy-2D input
path, so this decoder predicts deterministic 2D coordinates + per-joint
confidence in one shot.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch import Tensor

from .hand2d_net import LatentCNNEncoder
from .hand_network import fourier_encode
from .hand_network_perjoint import TripleAttentionBlock
from .network import zero_module


def _activation(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU()


class HandImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_grid = config.image_size // 16
        self.global_feat_dim = config.pose2d_global_feat_dim
        self.full_res_patches = config.pose2d_full_res_patches
        self.zero_bbox_mask = config.zero_bbox_mask
        self.bbox_cfg_mask_prob = config.bbox_cfg_mask_prob
        self.latent_cnn = LatentCNNEncoder(
            patch_feat_channels=config.pose2d_patch_feat_channels,
            d_out=config.pose2d_d_latent,
            global_feat_dim=self.global_feat_dim,
            activation=config.activation,
            downsample=not self.full_res_patches,
            patch_grid=self.patch_grid,
        )
        self.mask_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.patch_grid),
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GroupNorm(4, 8),
            _activation(config.activation),
            nn.Conv2d(8, config.pose2d_patch_feat_channels, kernel_size=1),
        )

    def encode_image(self, img_patch_feat, bbox_mask=None):
        return Pose2DPerJointDecoder.encode_image(self, img_patch_feat, bbox_mask)


class Pose2DPerJointDecoder(nn.Module):
    """Regression-mode 2D decoder using TripleAttentionBlock trunk."""

    def __init__(
        self,
        patch_feat_channels: int = 768,
        d_latent: int = 384,
        num_joints: int = 21,
        num_layers: int = 4,
        num_heads: int = 4,
        d_feedforward: int = 1024,
        dropout_p: float = 0.1,
        global_feat_dim: int = 64,
        activation: str = "gelu",
        d_cond: int = 7,
        d_noise_emb: int = 1024,
        regression_mode: bool = True,
        full_res_patches: bool = False,
        self_conditioning: bool = False,
        heatmap_head: bool = False,
        heatmap_grid: int = 32,
        patch_grid: int = 16,
        conf_in_state: bool = False,
        use_mmdit: bool = False,
        use_adaln: bool = False,
        adaln_bounded: bool = False,
        adaln_zero: bool = False,
        zero_bbox_mask: bool = False,
        bbox_cfg_mask_prob: float = 0.0,
        coord_range: float = 1.0,
        # The kwargs below are accepted for interface compatibility with
        # Pose2DTransformerDecoder but are unused here.
        grad_checkpoint: bool = False,
        temporal_stride: int = 1,
        max_t: int = 1000,
        coord_fourier_freqs: int = 4,
        time_embed: str = "table",
    ):
        super().__init__()
        self.d_latent = d_latent
        self.num_joints = num_joints
        self.global_feat_dim = global_feat_dim
        self.regression_mode = regression_mode
        self.coord_fourier_freqs = coord_fourier_freqs
        self.full_res_patches = full_res_patches
        self.self_conditioning = self_conditioning and not regression_mode
        self.heatmap_head = heatmap_head
        self.heatmap_grid = heatmap_grid
        # DINO patch grid per side = image_size // 16 (16 @256px, 32 @512px). Every
        # fixed-16 assumption below is derived from this so 512 input works.
        self.patch_grid = patch_grid
        # conf_in_state: make confidence a DENOISING target by feeding the noised
        # confidence as a 3rd INPUT channel [x, y, conf] (flow-matching mode only).
        # The trunk thus conditions on noised conf; the OUTPUT keeps the separate
        # sigmoid conf_head + BCE (the head reads the trunk that saw noised conf,
        # so it learns to predict clean conf from noisy conf). Input is 3-channel,
        # coordinate output stays 2-channel.
        self.conf_in_state = conf_in_state and not regression_mode
        self.conf_ch = 1 if self.conf_in_state else 0
        self.coord_in_ch = 2 + self.conf_ch
        # Ablation: always feed an all-zero ('no box') bbox_mask to encode_image.
        self.zero_bbox_mask = zero_bbox_mask
        # CFG-style PER-FRAME bbox-condition dropout (train-time only): each frame
        # of a length-T sample independently drops its box with this probability
        # (frame t masked while t+1 visible). The image/video features are NEVER
        # masked -- only the bbox cue -- so the model learns both a
        # box-conditioned and a box-free operating point per frame.
        self.bbox_cfg_mask_prob = bbox_cfg_mask_prob
        # COORD RANGE (user, 2026-09-10): the Tanh coord head below is hard-bounded
        # to [-1,1], i.e. the model can NEVER place a joint outside the box it is
        # given (measured: max |uv_norm| 0.988, 0.0% of predictions escape the box).
        # With cfgbox (bbox_jitter_affects_norm) the normalisation box is jittered
        # and no longer contains every joint, so targets legitimately exceed 1 and
        # those joints saturate into vanishing gradients -- a silent soft-ignore.
        # coord_range k scales the head to [-k, k] so such targets stay learnable.
        self.coord_range = float(coord_range)

        # Image encoder. With full_res_patches the 16x16=256 grid is kept
        # (downsample=False) for finer spatial localization; otherwise it is
        # downsampled to 8x8=64 (the original behaviour).
        self.latent_cnn = LatentCNNEncoder(
            patch_feat_channels=patch_feat_channels,
            d_out=d_latent,
            global_feat_dim=global_feat_dim,
            activation=activation,
            downsample=not full_res_patches,
            patch_grid=patch_grid,
        )
        # bbox mask embedding: (B*T, 1, H, W) → (B*T, patch_feat_channels, G, G)
        # where G = patch_grid. AdaptiveAvgPool2d(G) FIRST so the
        # 1→patch_feat_channels expansion happens on the patch grid — without this
        # the conv2d output is (BT, 768, H, W) which blows up to 100+ GiB at H=W=256.
        mask_embed_dim = 8
        self.mask_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(patch_grid),
            nn.Conv2d(1, mask_embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, mask_embed_dim),
            _activation(activation),
            nn.Conv2d(mask_embed_dim, patch_feat_channels, kernel_size=1),
        )

        # Per-joint slot embedding (so the trunk can tell joints apart in
        # both regression and denoise modes).
        self.joint_id_emb = nn.Parameter(torch.randn(num_joints, d_latent) * 0.02)

        if regression_mode:
            # 21 learnable joint queries replace the noisy 2D input path.
            self.joint_queries = nn.Parameter(torch.randn(num_joints, d_latent) * 0.02)
        else:
            # Flow-matching path: encode noisy 2D coords (Fourier) and use
            # those as the per-joint input instead of fixed queries. With
            # conf_in_state the noised INPUT state is 3-channel [x, y, conf].
            coord_fourier_dim = self.coord_in_ch * (1 + 2 * coord_fourier_freqs)
            self.coord_in = nn.Sequential(
                nn.Linear(coord_fourier_dim, d_latent),
                _activation(activation),
                nn.Linear(d_latent, d_latent),
            )
            # Flow-matching noise level embedding (per-clip).
            # "table" (legacy learned row per timestep) or "sinusoidal" (DiT TimestepEmbedder);
            # see HandDenoiserPerJointConfig.time_embed. Name kept for checkpoint compatibility.
            from .hand_network_perjoint import make_time_embed
            self.noise_emb_table = make_time_embed(time_embed, max_t, d_noise_emb)
            self.noise_emb_to_token = nn.Linear(d_noise_emb, d_latent, bias=False)
            # Option 10: self-conditioning. Encode the previous x0 estimate
            # (same Fourier coords) and add to the joint token. Zero-init the
            # final layer so a None/zero sc signal is a no-op at init.
            if self.self_conditioning:
                self.sc_in = nn.Sequential(
                    nn.Linear(coord_fourier_dim, d_latent),
                    _activation(activation),
                    nn.Linear(d_latent, d_latent),
                )
                nn.init.zeros_(self.sc_in[-1].weight)
                nn.init.zeros_(self.sc_in[-1].bias)

        # Base-cond projection — match the original Pose2DTransformerDecoder
        # `cond_to_tokens`: a 2-layer MLP (Linear→Act→Linear). Added to both
        # the image patch tokens (memory side, like the original) and the
        # joint queries so the 2D decoder receives cond context on both
        # sides of cross-attention.
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_latent),
            _activation(activation),
            nn.Linear(d_latent, d_latent),
        )

        # Global pooled image feat -> per-frame bias on every joint query.
        # Zero-init so existing checkpoints (which lack this param) start
        # from the same behaviour, then the model can learn to use it.
        # Mirrors Pose2DTransformerDecoder.global_feat_to_query.
        self.global_feat_to_query = nn.Linear(global_feat_dim, d_latent)
        nn.init.zeros_(self.global_feat_to_query.weight)
        nn.init.zeros_(self.global_feat_to_query.bias)

        # Trunk: triple-attention OR MM-DiT (v3) blocks (same module as 3D).
        self.use_mmdit = use_mmdit
        # AdaLN-Zero on the 2D branch, mirroring the 3D trunk. The 2D decoder
        # runs its OWN flow-matching process with its own per-clip noise level
        # (`ne` below), so it gets its own conditioner rather than borrowing the
        # 3D one: c = SiLU(Linear(noise_emb) + SideEmbedding(mano_side)). Unused
        # under MM-DiT (that block carries its own modulation) and degenerate in
        # regression mode, where `ne` is the zero buffer and only the hand-side
        # term varies -- still a useful signal, so it stays enabled there.
        self.use_adaln = bool(use_adaln) and not use_mmdit
        if self.use_adaln:
            self.adaln_noise_proj = nn.Linear(d_noise_emb, d_latent)
            self.adaln_side_emb = nn.Embedding(2, d_latent)
            nn.init.normal_(self.adaln_side_emb.weight, std=0.02)
        else:
            self.adaln_noise_proj = None
            self.adaln_side_emb = None
        if use_mmdit:
            from .mmdit_block import MMDiTBlock
            _Block = MMDiTBlock
        else:
            _Block = TripleAttentionBlock
        self.adaln_zero = bool(adaln_zero) and self.use_adaln
        if self.adaln_zero:
            self.final_norm = nn.LayerNorm(d_latent, elementwise_affine=False)
            self.final_mod = zero_module(nn.Linear(d_latent, 2 * d_latent, bias=True))
        _extra = {} if use_mmdit else {"use_adaln": self.use_adaln,
                                        "adaln_bounded": bool(adaln_bounded),
                                        "adaln_zero": self.adaln_zero}
        self.trunk = nn.ModuleList([
            _Block(
                d_latent=d_latent,
                d_feedforward=d_feedforward,
                n_heads=num_heads,
                dropout_p=dropout_p,
                activation=activation,
                d_noise_emb=d_noise_emb,
                use_rope=True,
                **_extra,
            )
            for _ in range(num_layers)
        ])
        # Per-joint output heads.
        if heatmap_head:
            # Option 1: soft-argmax over a GxG grid in output [-1,1]^2 space.
            G = heatmap_grid
            self.coord_logits = nn.Sequential(
                nn.LayerNorm(d_latent),
                nn.Linear(d_latent, d_latent),
                _activation(activation),
                nn.Linear(d_latent, G * G),
            )
            lin = torch.linspace(-1.0, 1.0, G)
            gy, gx = torch.meshgrid(lin, lin, indexing="ij")
            # (G*G, 2) anchor coordinates in [-1, 1].
            self.register_buffer("heatmap_anchors", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        else:
            self.coord_head = nn.Sequential(
                nn.LayerNorm(d_latent),
                nn.Linear(d_latent, d_latent),
                _activation(activation),
                nn.Linear(d_latent, 2),     # coordinate output is always 2 (x,y)
                nn.Tanh(),                  # constrain to [-1, 1] normalized space.
            )
        # conf head: 1 sigmoid channel [0,1] (BCE-trained).
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
            _activation(activation),
            nn.Linear(d_latent, 1),
            nn.Sigmoid(),
        )

        # Regression-mode zero noise emb (TripleAttentionBlock's temporal
        # TransformerBlock expects a (B, d_noise) tensor regardless).
        self.register_buffer("zero_noise_emb", torch.zeros(d_noise_emb))

    # ------------------------------------------------------------------
    # Encode image (same signature as Pose2DTransformerDecoder.encode_image).
    # ------------------------------------------------------------------
    def encode_image(
        self,
        img_patch_feat: Tensor,
        bbox_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, T = img_patch_feat.shape[:2]
        BT = B * T
        x = img_patch_feat.reshape(BT, *img_patch_feat.shape[2:])
        # Ablation: drop the bbox_mask cue entirely (train + inference) so the
        # model must localize from the image alone.
        if self.zero_bbox_mask:
            bbox_mask = None
        if bbox_mask is None:
            bbox_mask = torch.zeros(
                B, 1, self.patch_grid, self.patch_grid, device=x.device, dtype=x.dtype
            )
        if bbox_mask.dim() == 5:
            bbox_exp = bbox_mask.reshape(BT, 1, bbox_mask.shape[3], bbox_mask.shape[4])
        else:
            h, w = bbox_mask.shape[2], bbox_mask.shape[3]
            bbox_exp = bbox_mask[:, None].expand(B, T, 1, h, w).reshape(BT, 1, h, w)
        # Per-frame CFG dropout of the bbox cue (see __init__). Zeroed frames are
        # indistinguishable from 'no box', i.e. the null condition.
        # GATE FIX (2026-09-08): the inference harness never calls net.eval()
        # (only the DINO/WAN extractors), so self.training alone fired this at
        # eval and randomly stripped 30% of the box conditioning (ft_cfgbox
        # step-1400000 rows read PA 10.8 on byte-identical weights). Require
        # grad to be enabled too: true training only.
        # Bulletproof gate: the harness neither .eval()s the net NOR wraps
        # sampling in no_grad, so module state cannot distinguish train/eval.
        # The trainer exports HAND_TRAINING=1; nothing else does.
        if (os.environ.get("HAND_TRAINING") == "1"
                and self.bbox_cfg_mask_prob > 0
                and not self.zero_bbox_mask):
            _keep = (torch.rand(BT, 1, 1, 1, device=bbox_exp.device)
                     >= self.bbox_cfg_mask_prob).to(bbox_exp.dtype)
            bbox_exp = bbox_exp * _keep
        x = x + self.mask_embed(bbox_exp)

        cnn_feat = self.latent_cnn.conv_layers(x)
        global_feat = self.latent_cnn.global_pool(cnn_feat).squeeze(-1).squeeze(-1)
        global_feat = self.latent_cnn.global_proj(global_feat)
        global_feat = global_feat.reshape(B, T, self.global_feat_dim)

        # full_res_patches keeps the 16x16 grid; otherwise downsample to 8x8.
        feat = cnn_feat if self.full_res_patches else self.latent_cnn.spatial_downsample(cnn_feat)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self.latent_cnn.spatial_pos_embed
        return tokens, global_feat

    # ------------------------------------------------------------------
    # Forward (matches the call signature used by HandDenoiserPerJoint).
    # ------------------------------------------------------------------
    def forward(
        self,
        img_patch_feat: Tensor,
        noisy_joint_2d: Tensor | None = None,
        cond: Tensor | None = None,
        bbox_mask: Tensor | None = None,
        cached_tokens: Tensor | None = None,
        cached_global_feat: Tensor | None = None,
        cached_global_feat_detach: Tensor | None = None,
        noise_emb: Tensor | None = None,
        t: Tensor | None = None,
        sc_joint_2d: Tensor | None = None,
        gf_global_feat: Tensor | None = None,
        mano_side: Tensor | None = None,
        return_joint_tok: bool = False,
    ):
        if cached_tokens is None:
            cached_tokens, cached_global_feat = self.encode_image(
                img_patch_feat, bbox_mask=bbox_mask,
            )
        BT = cached_tokens.shape[0]
        B = cached_global_feat.shape[0]
        T = BT // B
        D = self.d_latent

        # (B, T, 64, D)
        patches = cached_tokens.reshape(B, T, -1, D)

        # ---- joint token initialisation ------------------------------
        if self.regression_mode:
            # 21 learnable joint queries broadcast to (B, T, 21, D).
            joint_tok = self.joint_queries.view(1, 1, self.num_joints, D).expand(B, T, -1, -1)
            ne = self.zero_noise_emb.unsqueeze(0).expand(B, -1)
        else:
            # Flow-matching: noisy 2D coords (B, T, 21, 2) → Fourier → D.
            if noisy_joint_2d is None:
                raise ValueError(
                    "Pose2DPerJointDecoder in flow-matching mode requires noisy_joint_2d.")
            J = self.num_joints
            fourier = fourier_encode(
                noisy_joint_2d.reshape(B * T * J, self.coord_in_ch),
                freqs=self.coord_fourier_freqs,
            ).reshape(B, T, J, -1)
            joint_tok = self.coord_in(fourier)
            # Add the per-clip noise-level embedding to every token.
            if t is None:
                raise ValueError(
                    "Pose2DPerJointDecoder in flow-matching mode requires t (noise level).")
            ne = self.noise_emb_table(t - 1)                       # (B, d_noise)
            noise_to_tok = self.noise_emb_to_token(ne)             # (B, D)
            joint_tok = joint_tok + noise_to_tok[:, None, None, :]
            # Option 10: add the self-conditioning x0 estimate (zeros if none).
            if self.self_conditioning:
                if sc_joint_2d is None:
                    sc_joint_2d = torch.zeros_like(noisy_joint_2d)
                sc_fourier = fourier_encode(
                    sc_joint_2d.reshape(B * T * J, self.coord_in_ch),
                    freqs=self.coord_fourier_freqs,
                ).reshape(B, T, J, -1)
                joint_tok = joint_tok + self.sc_in(sc_fourier)

        # Per-joint slot embedding (helps the trunk distinguish joints).
        joint_tok = joint_tok + self.joint_id_emb[None, None, :, :]

        # Base cond (mano_side / palm pose / intrinsics) → added to BOTH
        # the joint queries and the image patch tokens (memory side), so
        # cross-attention KV sees the same cond context as the queries
        # — matches the original Pose2DTransformerDecoder.cond_to_tokens.
        if cond is not None:
            cond_proj = self.cond_proj(cond.reshape(B, T, -1))     # (B, T, D)
            joint_tok = joint_tok + cond_proj.unsqueeze(2)
            patches = patches + cond_proj.unsqueeze(2)

        # Global pooled image feature → per-frame bias on every joint
        # query (mirrors Pose2DTransformerDecoder.global_feat_to_query).
        # Dual-image mode: if a separate (cropped-hand) global feat is supplied,
        # use it for the query bias so the queries get fine hand detail while the
        # patch memory still comes from the full image.
        gf_src = gf_global_feat if gf_global_feat is not None else cached_global_feat
        gf_bias = self.global_feat_to_query(gf_src)                # (B, T, D)
        joint_tok = joint_tok + gf_bias.unsqueeze(2)

        # ---- AdaLN conditioning c = f(this branch's timestep, hand side) ----
        adaln_c = None
        if self.use_adaln:
            if mano_side is not None:
                _side = mano_side.reshape(B, -1)[:, 0]
                _side_idx = (_side > 0.5).long()
            else:
                # no side supplied: fall back to the right-hand slot (1), which
                # is the only side this project optimises for.
                _side_idx = torch.ones(B, dtype=torch.long, device=joint_tok.device)
            adaln_c = torch.nn.functional.silu(
                self.adaln_noise_proj(ne) + self.adaln_side_emb(_side_idx)
            )

        if self.use_mmdit:
            img_stream = patches
            for block in self.trunk:
                joint_tok, img_stream = block(joint_tok, img_stream, ne)
        else:
            for block in self.trunk:
                joint_tok = block(joint_tok, patches, ne, cond=adaln_c)
            if getattr(self, "adaln_zero", False) and adaln_c is not None:
                f_shift, f_scale = self.final_mod(adaln_c).chunk(2, dim=-1)
                joint_tok = (self.final_norm(joint_tok) * (1 + f_scale[:, None, None, :])
                             + f_shift[:, None, None, :])

        # Per-joint heads.
        if self.heatmap_head:
            logits = self.coord_logits(joint_tok)               # (B, T, 21, G*G)
            probs = torch.softmax(logits, dim=-1)               # (B, T, 21, G*G)
            # soft-argmax: expected anchor coordinate. (G*G,2) -> (B,T,21,2)
            pose_2d = torch.einsum("btjg,gc->btjc", probs, self.heatmap_anchors)
        else:
            pose_2d = self.coord_head(joint_tok)        # (B, T, 21, 2), Tanh -> [-1,1]
            if self.coord_range != 1.0:
                pose_2d = pose_2d * self.coord_range    # -> [-coord_range, coord_range]
        # confidence: always the sigmoid conf_head. With conf_in_state the trunk
        # (joint_tok) was conditioned on the NOISED conf input, so this head
        # learns to denoise it -> predict clean conf (BCE vs GT visibility).
        confidence = self.conf_head(joint_tok)      # (B, T, 21, 1) sigmoid [0,1]
        if return_joint_tok:
            # Also expose the post-trunk per-joint feature tokens (B,T,21,d_latent)
            # so the 3D denoiser can consume them as a rich 2D-pose feature
            # (pass_2d_joint_tok) instead of only the predicted 2D coordinates.
            return pose_2d, confidence, cached_global_feat, joint_tok
        return pose_2d, confidence, cached_global_feat
