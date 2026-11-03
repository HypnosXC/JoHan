from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal, assert_never

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn
from .mano_smplx_layer import ManoLayer
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


class HandDenoiseTraj(TensorDataclass):
    """Data structure for denoising. Contains tensors that we are denoising, as
    well as utilities for packing + unpacking them."""
    mano_betas: Float[Tensor, "*#batch timesteps 10"]
    """Hand shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    mano_poses: Float[Tensor, "*#batch timesteps 15 3"]
    """Relative poses for each hand joint."""

    mano_poses_mat: Float[Tensor, "*#batch timesteps 15 9"]
    """Relative rot matrix for each hand joint."""

    global_orientation: Float[Tensor, "*#batch timesteps 3"]
    """Global Orientation for hand."""

    global_ori_mat: Float[Tensor, "*#batch timesteps 9"]
    """Global Orientation Matrix for hand."""

    global_translation: Float[Tensor, "*#batch timesteps 3"]
    """Global Camera translation"""

    mano_side: Float[Tensor, "*#batch timesteps 1"]
    """right or left hand"""
    def __init__(self,**kwargs):
        *batch,time,_,_=kwargs["mano_poses"].shape
        self.mano_betas=kwargs["mano_betas"]
        self.mano_poses=kwargs["mano_poses"]
        self.mano_poses_mat=SO3.exp(kwargs["mano_poses"].float()).as_matrix().to(kwargs["mano_poses"].dtype).reshape((*batch,time,15,9))
        if kwargs.get("global_orientation",None) is not None:
            self.global_orientation=kwargs["global_orientation"]
            self.global_ori_mat=SO3.exp(kwargs["global_orientation"].float()).as_matrix().to(kwargs["global_orientation"].dtype).reshape((*batch,time,9))
        else:
            self.global_orientation=None
            self.global_ori_mat=None
        if kwargs.get("global_translation",None) is not None:
            self.global_translation=kwargs["global_translation"]
        else:
            self.global_translation=None
        self.mano_side=kwargs["mano_side"]
    @staticmethod
    def get_packed_dim(using_mat:bool) -> int:
        if using_mat:
            packed_dim = 10 + 15 * 9# + 9 + 3
        else:
            packed_dim = 10 + 15 * 3# + 3 + 3
        return packed_dim

    def apply_to_hand(self,) -> tuple[torch.Tensor, torch.Tensor]:
        mano_side = (torch.sum(self.mano_side > 0.5,dim = 1) > self.mano_side.shape[1] / 2).squeeze(-1).cpu().numpy()
        batch, time, _ , _ = self.mano_poses.shape
        #assert batch == 1 # actually we only can handle one video per process
        left_mano_layer = ManoLayer(
                            flat_hand_mean=True,
                            use_pca = False,
                            ncomps=45,
                            side="left",
                            mano_root=None,
                        ).to(self.mano_poses.device)
        right_mano_layer = ManoLayer(
                            flat_hand_mean=True,
                            use_pca = False,
                            ncomps=45,
                            side="right",
                            mano_root=None,
                        ).to(self.mano_poses.device)
        vertices=[]
        joints=[]
        for i in range(batch):
            if mano_side[i]==True:
                vert, jt = right_mano_layer(
                    torch.cat((self.global_orientation[i,:,:],self.mano_poses[i,:,:].reshape(time,-1)),dim=-1),
                    self.mano_betas[i,:,:],
                    self.global_translation[i,:,:]
                )
            else:
                vert, jt = left_mano_layer(
                    torch.cat((self.global_orientation[i,:,:],self.mano_poses[i,:,:].reshape(time,-1)),dim=-1),
                    self.mano_betas[i,:,:],
                    self.global_translation[i,:,:]
                )
            vertices.append(vert)
            joints.append(jt)
        vertices=torch.stack(vertices,dim=0)
        joints=torch.stack(joints,dim=0)/1000
        vertices = vertices/ 1000  # Convert to meters
        if mano_side[0] == True:
            faces_m = right_mano_layer.th_faces
        else:
            faces_m = left_mano_layer.th_faces
        return vertices, faces_m, joints

    # Fields packed into the flat trajectory vector for each `using_mat` variant.
    _PACK_FIELDS_MAT = ("mano_betas", "mano_poses_mat")
    _PACK_FIELDS_VEC = ("mano_betas", "mano_poses")

    def pack(self, using_mat: bool) -> Float[Tensor, "*#batch timesteps d_state"]:
        """Pack the trajectory into a single flat vector per timestep."""
        *batch, time, num_joints, _ = self.mano_poses.shape
        assert num_joints == 15
        fields = self._PACK_FIELDS_MAT if using_mat else self._PACK_FIELDS_VEC
        return torch.cat(
            [getattr(self, f).reshape((*batch, time, -1)) for f in fields],
            dim=-1,
        )


    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*#batch timesteps d_state"],
        mano_side: Float[Tensor, "batch time 1"],
        using_mat: bool,
        project_rotmats: bool = False,
        
    ) -> HandDenoiseTraj:
        """Unpack trajectory from a single flattened vector.

        Args:
            x: Packed trajectory.
            project_rotmats: If True, project the rotation matrices to SO(3) via SVD.
        """
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(using_mat)
        if using_mat:
            mano_betas, mano_poses_mat = torch.split(
                x, [10, 15 * 9], dim=-1
            )
            mano_poses_mat = mano_poses_mat.reshape((*batch, time, 15, 9))
            # SO3.log() doesn't support bfloat16 — cast to float32 for the rotation op.
            mat_f32 = mano_poses_mat.reshape((*batch, time, 15, 3, 3)).float()
            mano_poses = SO3.from_matrix(mat_f32).log().to(mano_poses_mat.dtype)
        else:
            mano_betas, mano_poses = torch.split(
                x, [10, 15 * 3], dim=-1
            )
            mano_poses = mano_poses.reshape((*batch, time, 15, 3))
            mano_poses_mat = SO3.exp(mano_poses.float()).as_matrix().to(mano_poses.dtype).reshape((*batch, time, 15, 9))
        assert mano_betas.shape == (*batch, time, 10)

        return HandDenoiseTraj(
            mano_betas=mano_betas,
            mano_poses=mano_poses,
            mano_side=mano_side
        )


@dataclass(frozen=True)
class HandDenoiserConfig:
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.2
    pose2d_patch_feat_channels: int = 768
    pose2d_d_latent: int = 256
    """Transformer dim after temporal compression (d_cnn * temporal_k)."""
    pose2d_num_joints: int = 21
    pose2d_num_layers: int = 4
    pose2d_num_heads: int = 4
    pose2d_d_feedforward: int = 1024
    pose2d_global_feat_dim: int = 64
    temporal_k: int = 1
    """Temporal compression factor: T → T/k for all transformers. 1 = no compression."""
    using_mat: bool = True
    using_img_feat: bool = True
    predict_2Dpose: bool = True
    predict_global_pose: bool = False
    global_pose_num_layers: int = 2
    use_optimal_2d: bool = False
    """Use clean 2D joints for 3D conditioning (two-stage inference)."""
    noise_2d_cond: bool = False
    """PURE-3D conditioning augmentation: feed the 2D->3D condition as the
    FLOW-NOISED 2D pose at the 3D flow timestep t —
        noise_2d_t = (1 - t_frac) * clean_2d + t_frac * eps
    — instead of the clean GT 2D. So the 3D lifter is trained on 2D inputs whose
    noise level tracks the 3D denoising step (clean 2D at t->0, pure noise at
    t->1), teaching it to tolerate imperfect / early-step 2D. Use with
    predict_2Dpose=False + use_optimal_2d=True (pure-3D, 2D-conditioned). At
    inference the fed 2D must be noised on the same schedule per step."""
    zero_img_feat_in_3d_cond: bool = False
    """If True, zero out the global image feature in the 3D denoiser's conditioning.
    Use with predict_2Dpose=False to feed only GT 2D pose (and side/intrinsics) as condition."""
    ignore_2d_cond: bool = False
    """If True, the 3D trunk ignores ALL 2D-pose conditioning (forces base_2d=None
    in the perjoint forward) — 3D is conditioned ONLY on image features (patch
    cross-attention + global feat) + side/betas. Ablation of the 2D intermediate.
    Pair with predict_2Dpose=False (no 2D head) and zeroed joint_2d* loss weights."""
    pose2d_absolute: bool = False
    """2D head emits ABSOLUTE pixel coords in [0, image_size] instead of the usual
    [-1,1] box-normalised ones. The LOSS and the 2D->3D conditioning both
    re-normalise into a box derived from the PREDICTION (detached, side-clamped),
    so the network carries absolute placement while the objective stays
    scale-invariant. Changes what the 2D state channels MEAN, so it is NOT
    resume-compatible with a normalised checkpoint -- train from scratch."""
    pose2d_absolute_min_wh: float = 0.0625
    """Minimum predicted-box side, in pixels. Dividing by a per-frame box is what
    caused the >853k divergences (see training_loss.py); with a PREDICTED box the
    early-training values are arbitrary, so this floor is what bounds 1/box."""
    pose2d_conf_in_state: bool = False
    """If True (flow-matching 2D only), fold per-joint confidence into the 2D
    flow-matching state as a 3rd channel [x, y, vis], so confidence is NOISED and
    DENOISED exactly like the coordinates (MSE flow target) instead of a separate
    sigmoid head trained with BCE. Removes the dominating conf-BCE gradient. The
    vis-channel x0 target is joint_visibility (0/1). Set conf_loss_bce=False with
    this (the conf term becomes the flow MSE on the 3rd channel)."""
    pose2d_state_only: bool = False
    """PURE-2D motion generation: the flow-matching state IS the per-joint 2D pose
    (+ confidence), with NO MANO/3D at all. The Pose2DPerJointDecoder becomes the
    whole denoiser (the MANO trunk is bypassed). Packed state dim = num_joints *
    (2 + conf_ch). Implies per_joint_arch + predict_2Dpose + pose2d_conf_in_state.
    Only joint_2d / joint_2d_conf / joint_2d_vel / joint_2d_accel losses apply;
    all MANO/3D loss weights must be 0. Requires from-scratch training (d_state
    changes 145 -> ~63). No 3D eval (PA-MPJPE undefined); 2D-MPJPE + AP only."""
    pose2d_regression: bool = False
    """If True, 2D pose decoder regresses joints directly from image features
    (learnable queries, no flow matching noising). 3D trajectory still uses flow matching."""
    pose2d_grad_checkpoint: bool = False
    """If True, gradient-checkpoint the Pose2DBlocks. Trades ~20% compute for
    ~60% activation memory. Only enable for large pose2d_d_latent configs."""
    d_2d_feat: int = 32
    """Per-joint 2D feature dimension. Higher = richer 2D features for 2D-3D alignment."""
    use_xattn_rot_to_2d: bool = False
    """If True, use cross-attention so each rotation query attends to ALL 21 2D joint features
    (instead of the hardcoded rot_joint→2D_joint mapping). Lets the model learn kinematic links."""
    xattn_rot_num_heads: int = 4
    pose2d_temporal_stride: int = 8
    """Stride for the 2D decoder's temporal self-attention KV side. With
    stride=k > 1, KV is computed only every k-th frame: Q@T × KV@(T/k) gives
    a k× speedup on this op (the dominant 2D-decoder cost). stride=1 keeps
    the symmetric Q=K=V=T behaviour. Note: enabling this reshapes the
    temporal projection layout (qkv → q + kv), so existing checkpoints will
    re-init those temporal weights when loaded with strict=False."""
    pose2d_fourier_freqs: int = 3
    """Fourier encoding frequencies for 2D pose coordinates before feeding to 3D model."""
    in_context_learning: bool = False
    activation: Literal["gelu", "relu"] = "gelu"

    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"

    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )


    cond_param: Literal[
        "ours", "all", "wrist_motion","differential","canonicalized", "absolute", "absrel", "absrel_global_deltas"
    ] = "only_side"
    """Which conditioning parameterization to use.

    "ours" is the default, we try to be clever and design something with nice
        equivariance properties.
    "canonicalized" contains a transformation that's canonicalized to aligned
        to the first frame.
    "absolute" is the naive case, where we just pass in transformations
        directly.
    """

    # Raw dim per cond_param (before Fourier encoding).
    _COND_RAW_DIMS = {
        "all": 62, "only_side": 1, "ours": 6, "wrist_motion": 13,
        "differential": 13, "canonicalized": 12, "absolute": 12,
        "absrel": 24, "absrel_global_deltas": 24,
    }

    @cached_property
    def d_cond_base(self) -> int:
        """make_cond output dim (after Fourier, before img feat concat)."""
        raw = self._COND_RAW_DIMS[self.cond_param]
        return raw + raw * self.fourier_enc_freqs * 2

    @cached_property
    def d_cond(self) -> int:
        """Total conditioning dim fed into latent_from_cond."""
        d_cond = self.d_cond_base
        if self.using_img_feat:
            # Both predict_2Dpose=True and False use pose2d_decoder for image encoding
            d_cond += self.pose2d_global_feat_dim
            if self.predict_global_pose:
                d_cond += self.pose2d_d_latent
        return d_cond

    def make_cond(
        self,
        rel_palm_pose: Float[Tensor, "batch time 3"],  # relative palm pose to the camera
        conds: HandDenoiseTraj,
    ) -> Float[Tensor, "batch time d_cond"]:
        """Build the per-frame conditioning vector before Fourier encoding."""
        if self.cond_param != "only_side":
            raise ValueError("Current ViT-H inference conditions only on hand side")
        return fourier_encode(conds.mano_side, freqs=self.fourier_enc_freqs)


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


