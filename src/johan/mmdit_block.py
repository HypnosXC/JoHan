"""MM-DiT block with 3D RoPE — "v3" trunk, a drop-in for TripleAttentionBlock.

Faithfully replicates the structure of diffusers' SD3 `JointTransformerBlock`
(per-stream AdaLN-Zero modulation + JOINT attention over the concatenated
streams + per-stream gated MLP), but adds 3D axial RoPE on the queries/keys
because (a) diffusers isn't installed on the cluster and (b) SD3's block uses a
learned positional embedding, not RoPE — so "MM-DiT + 3D RoPE" is a custom merge.

Two streams:
  * POSE  tokens  (B, T, S, D)   — the denoised state (2D joints, or 3D hand)
  * IMAGE tokens  (B, T, P, D)   — frozen image-patch features (conditioning)

Two attention stages per block:
  * Step A — per-frame JOINT MM-DiT attention over [pose(S) ⊕ image(P)]. This is
    where pose conditions on the image. 2D RoPE (row, col): pose token j ->
    (row=j, col=0); image patch -> (row=patch_row, col=patch_col+1). Updates BOTH
    streams (real MM-DiT). Per-frame so the big image stream (T·P) stays cheap.
  * Step B — FULL pose self-attention over ALL T·S pose tokens at once (not just
    temporal), so every joint at every frame attends to every other. 2D RoPE
    (t, joint). Image is not included here (T·P too large for full joint attn).
Together the pose tokens carry full (t, joint, stream) 3D positional information.

The image stream is updated across blocks too (real MM-DiT), so the trunk loop
must thread `(pose, image)` through; HandDenoiserPerJoint / Pose2DPerJointDecoder
branch on `use_mmdit` to do `pose, image = block(pose, image, noise_emb)`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _act(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class AxialRoPE(nn.Module):
    """N-D rotary position embedding (GPT-NeoX paired-half convention).

    Splits the per-head dim's frequency budget across `n_axes`; each token's
    position along every axis rotates its corresponding frequency block.
    """

    def __init__(self, head_dim: int, n_axes: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        half = head_dim // 2
        per = half // n_axes
        self.splits = [per] * (n_axes - 1) + [half - per * (n_axes - 1)]
        self.n_axes = n_axes
        self.head_dim = head_dim
        invs = []
        for s in self.splits:
            if s == 0:
                invs.append(torch.zeros(0))
            else:
                invs.append(1.0 / (theta ** (torch.arange(0, s).float() / s)))
        # store concatenated inv_freq + the split sizes
        self.register_buffer("inv_freq", torch.cat(invs), persistent=False)

    def cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """positions: (N, n_axes) float -> cos,sin each (N, head_dim)."""
        freqs = []
        off = 0
        for a, s in enumerate(self.splits):
            if s == 0:
                continue
            inv = self.inv_freq[off:off + s]
            off += s
            freqs.append(positions[:, a:a + 1] * inv[None, :])   # (N, s)
        freq = torch.cat(freqs, dim=-1)                          # (N, half)
        emb = torch.cat((freq, freq), dim=-1)                    # (N, head_dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """x: (B, H, N, Dh); cos/sin: (N, Dh)."""
        cos = cos[None, None].to(x.dtype)
        sin = sin[None, None].to(x.dtype)
        return x * cos + _rotate_half(x) * sin


class _Modulation(nn.Module):
    """AdaLN-Zero: noise_emb -> (shift1,scale1,gate1, shift2,scale2,gate2)."""

    def __init__(self, d_latent: int, d_noise_emb: int):
        super().__init__()
        self.lin = nn.Linear(d_noise_emb, 6 * d_latent)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)   # zero-init -> identity block at start

    def forward(self, c: Tensor):
        # c: (M, d_noise) -> 6 x (M, D)
        return self.lin(F.silu(c)).chunk(6, dim=-1)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # x: (M, N, D); shift/scale: (M, D)
    return x * (1 + scale[:, None]) + shift[:, None]


class MMDiTBlock(nn.Module):
    """One MM-DiT round. Signature-compatible inputs as TripleAttentionBlock but
    returns BOTH updated streams: (pose, image)."""

    def __init__(self, d_latent: int, d_feedforward: int, n_heads: int,
                 dropout_p: float, activation: str, d_noise_emb: int,
                 use_rope: bool = True):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_latent // n_heads
        self.use_rope = use_rope
        self.dropout_p = dropout_p
        # NB: streams are distinguished by their SEPARATE qkv/mlp/AdaLN weights
        # (as in SD3's JointTransformerBlock) — no explicit type embedding, so the
        # block is a strict identity at init (AdaLN-Zero) and nothing accumulates.

        # --- spatial JOINT attention (per frame) -----------------------
        self.sp_mod_pose = _Modulation(d_latent, d_noise_emb)
        self.sp_mod_img = _Modulation(d_latent, d_noise_emb)
        self.sp_norm1_pose = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.sp_norm1_img = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.sp_norm2_pose = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.sp_norm2_img = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.sp_qkv_pose = nn.Linear(d_latent, 3 * d_latent, bias=False)
        self.sp_qkv_img = nn.Linear(d_latent, 3 * d_latent, bias=False)
        self.sp_out_pose = nn.Linear(d_latent, d_latent, bias=False)
        self.sp_out_img = nn.Linear(d_latent, d_latent, bias=False)
        self.sp_mlp_pose = nn.Sequential(
            nn.Linear(d_latent, d_feedforward), _act(activation),
            nn.Dropout(dropout_p), nn.Linear(d_feedforward, d_latent))
        self.sp_mlp_img = nn.Sequential(
            nn.Linear(d_latent, d_feedforward), _act(activation),
            nn.Dropout(dropout_p), nn.Linear(d_feedforward, d_latent))
        self.rope_sp = AxialRoPE(self.head_dim, n_axes=2) if use_rope else None

        # --- FULL pose self-attention (all T*S tokens, joint+temporal) --
        self.tp_mod = _Modulation(d_latent, d_noise_emb)
        self.tp_norm1 = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.tp_norm2 = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.tp_qkv = nn.Linear(d_latent, 3 * d_latent, bias=False)
        self.tp_out = nn.Linear(d_latent, d_latent, bias=False)
        self.tp_mlp = nn.Sequential(
            nn.Linear(d_latent, d_feedforward), _act(activation),
            nn.Dropout(dropout_p), nn.Linear(d_feedforward, d_latent))
        # 2 axes: (time t, joint j)
        self.rope_tp = AxialRoPE(self.head_dim, n_axes=2) if use_rope else None

    def _split_heads(self, x: Tensor) -> Tensor:
        # (M, N, D) -> (M, H, N, Dh)
        M, N, _ = x.shape
        return x.view(M, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        M, H, N, Dh = x.shape
        return x.transpose(1, 2).reshape(M, N, H * Dh)

    # -- positions -------------------------------------------------------
    def _sp_positions(self, S: int, P: int, device) -> tuple[Tensor, Tensor]:
        # Image patches arrive flattened as (B,T,P,D) where encode_image did
        # feat.flatten(2) on (BT,C,H,W) -> ROW-MAJOR token index p = h*W + w.
        # So the true 2D grid position is recovered by (row=p//g, col=p%g) with
        # g=sqrt(P) (here P=256 -> 16x16, or P=64 -> 8x8). This is exactly
        # equivalent to a (B,T,g,g,D) grid; no reshape needed.
        g = int(round(math.sqrt(P)))
        assert g * g == P, (
            f"MM-DiT 2D RoPE expects a square P=g*g patch grid in row-major "
            f"order, got P={P} (g={g}). Re-derive _sp_positions if the patch "
            f"layout changes.")
        # pose: (row=j, col=0)
        pose_pos = torch.stack([
            torch.arange(S, device=device).float(),
            torch.zeros(S, device=device)], dim=-1)               # (S,2)
        # image: (row=h, col=w+1)  (+1 on col separates the two streams)
        rows = torch.arange(P, device=device) // g
        cols = torch.arange(P, device=device) % g
        img_pos = torch.stack([rows.float(), cols.float() + 1.0], dim=-1)  # (P,2)
        return pose_pos, img_pos

    def forward(self, pose: Tensor, image: Tensor, noise_emb: Tensor):
        """pose (B,T,S,D), image (B,T,P,D), noise_emb (B,d_noise)
        -> updated (pose, image)."""
        B, T, S, D = pose.shape
        P = image.shape[2]
        dev = pose.device

        # ===== Step A: per-frame JOINT MM-DiT attention =================
        xp = pose.reshape(B * T, S, D)
        xi = image.reshape(B * T, P, D)
        c = noise_emb.repeat_interleave(T, dim=0)                  # (B*T, d_noise)
        sh1p, sc1p, g1p, sh2p, sc2p, g2p = self.sp_mod_pose(c)
        sh1i, sc1i, g1i, sh2i, sc2i, g2i = self.sp_mod_img(c)

        hp = _modulate(self.sp_norm1_pose(xp), sh1p, sc1p)
        hi = _modulate(self.sp_norm1_img(xi), sh1i, sc1i)
        qp, kp, vp = self.sp_qkv_pose(hp).chunk(3, dim=-1)
        qi, ki, vi = self.sp_qkv_img(hi).chunk(3, dim=-1)
        q = self._split_heads(torch.cat([qp, qi], dim=1))         # (BT,H,S+P,Dh)
        k = self._split_heads(torch.cat([kp, ki], dim=1))
        v = self._split_heads(torch.cat([vp, vi], dim=1))
        if self.rope_sp is not None:
            pose_pos, img_pos = self._sp_positions(S, P, dev)
            cos, sin = self.rope_sp.cos_sin(torch.cat([pose_pos, img_pos], dim=0))
            q = AxialRoPE.apply(q, cos, sin)
            k = AxialRoPE.apply(k, cos, sin)
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        a = self._merge_heads(a)                                  # (BT, S+P, D)
        ap, ai = a[:, :S], a[:, S:]
        xp = xp + g1p[:, None] * self.sp_out_pose(ap)
        xi = xi + g1i[:, None] * self.sp_out_img(ai)
        xp = xp + g2p[:, None] * self.sp_mlp_pose(_modulate(self.sp_norm2_pose(xp), sh2p, sc2p))
        xi = xi + g2i[:, None] * self.sp_mlp_img(_modulate(self.sp_norm2_img(xi), sh2i, sc2i))
        pose = xp.reshape(B, T, S, D)
        image = xi.reshape(B, T, P, D)

        # ===== Step B: FULL pose self-attention over all T*S tokens ====
        xt = pose.reshape(B, T * S, D)                            # (B, T*S, D)
        sh1, sc1, g1, sh2, sc2, g2 = self.tp_mod(noise_emb)       # c = (B, d_noise)
        h = _modulate(self.tp_norm1(xt), sh1, sc1)
        q, k, v = self.tp_qkv(h).chunk(3, dim=-1)
        q = self._split_heads(q); k = self._split_heads(k); v = self._split_heads(v)
        if self.rope_tp is not None:
            # position of pose token (t, j): (T*S, 2) in row-major (t outer, j inner)
            ts = torch.arange(T, device=dev).float().repeat_interleave(S)   # (T*S,)
            js = torch.arange(S, device=dev).float().repeat(T)              # (T*S,)
            cos, sin = self.rope_tp.cos_sin(torch.stack([ts, js], dim=-1))
            q = AxialRoPE.apply(q, cos, sin)
            k = AxialRoPE.apply(k, cos, sin)
        at = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        at = self._merge_heads(at)
        xt = xt + g1[:, None] * self.tp_out(at)
        xt = xt + g2[:, None] * self.tp_mlp(_modulate(self.tp_norm2(xt), sh2, sc2))
        pose = xt.reshape(B, T, S, D)

        return pose, image
