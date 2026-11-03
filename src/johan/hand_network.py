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
from .hand2d_net import Pose2DTransformerDecoder, GlobalPoseHead, TemporalCrossAttnBlock, TemporalSelfAttnBlock
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

    include_canonicalized_cpf_rotation_in_cond: bool = True

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
        batch, time, _ = conds.global_translation.shape

        if self.cond_param == "all":
            cond = conds.pack()
        elif self.cond_param == "only_side":
            cond = conds.mano_side
        elif self.cond_param == "ours":
            diff_cond = torch.zeros_like(rel_palm_pose)
            diff_cond[:, 1:, :] = rel_palm_pose[:, 1:, :] - rel_palm_pose[:, :-1, :]
            cond = torch.cat((rel_palm_pose, diff_cond), dim=-1)
        elif self.cond_param in ("wrist_motion", "differential"):
            # Both branches build the same per-frame [cur_motion(0) | diff_motion]
            # absolute+relative SE(3) representation. The original "differential"
            # branch used to zero out cur_motion via inv@cur, but that just yields
            # identity, so we use the same code for both names.
            trans = conds.global_translation
            prior_orien = SO3.exp(conds.global_orientation[:, :-1, :].float())
            cur_orien = SO3.exp(conds.global_orientation[:, 1:, :].float())
            prior_motion = SE3.from_rotation_and_translation(prior_orien, trans[:, :-1, :])
            cur_motion = SE3.from_rotation_and_translation(cur_orien, trans[:, 1:, :])
            diff_motion = prior_motion.inverse() @ cur_motion
            if self.cond_param == "differential":
                cur_motion = cur_motion.inverse() @ cur_motion  # identity
            cur_mat = cur_motion.as_matrix()[..., :3, :].reshape((batch, time - 1, 12))
            diff_mat = diff_motion.as_matrix()[..., :3, :].reshape((batch, time - 1, 12))
            cond = torch.cat((cur_mat[:, :1, :], diff_mat), dim=1)
            cond = torch.cat((cond, conds.mano_side), dim=-1)
        else:
            assert_never(self.cond_param)
        return fourier_encode(cond, freqs=self.fourier_enc_freqs)


class HandDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: HandDenoiserConfig):
        super().__init__()

        self.config = config
        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]
        if config.using_img_feat:
            # Always create pose2d_decoder for image encoding (even if not predicting 2D)
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
            # 2D-3D encoder: aligns 21 2D joints with 15 3D rotation joints.
            # Mapping: rotation joint i → output joint index (for 2D lookup)
            #   Thumb: rot 0,1,2 → out 1,2,3
            #   Index: rot 3,4,5 → out 5,6,7
            #   Middle: rot 6,7,8 → out 9,10,11
            #   Ring: rot 9,10,11 → out 13,14,15
            #   Pinky: rot 12,13,14 → out 17,18,19
            # Wrist (out 0) and fingertips (out 4,8,12,16,20) have no rotation params.
            self.pose2d_fourier_freqs = config.pose2d_fourier_freqs
            self.num_pose2d_joints = config.pose2d_num_joints  # 21

            self.register_buffer('rotation_to_output_joint', torch.tensor([
                1, 2, 3,    # thumb
                5, 6, 7,    # index
                9, 10, 11,  # middle
                13, 14, 15, # ring
                17, 18, 19  # pinky
            ], dtype=torch.long))

            # Per-coordinate Fourier dim for 2D joints
            fourier_2d_dim = 2 * (1 + 2 * config.pose2d_fourier_freqs)  # 2D coords
            self.fourier_2d_dim = fourier_2d_dim

            # Per-joint 2D encoder: Fourier(2D) → d_2d_feat
            # This will be concatenated with 9D rotation for aligned encoder input
            d_2d_feat = config.d_2d_feat  # configurable 2D feature dim
            self.d_2d_feat = d_2d_feat

            # Cross-attention from 15 rotation queries to all 21 2D joint features
            self.use_xattn_rot_to_2d = config.use_xattn_rot_to_2d
            if config.use_xattn_rot_to_2d:
                self.rot_queries_xattn = nn.Parameter(torch.randn(15, d_2d_feat) * 0.02)
                self.xattn_rot_to_2d = nn.MultiheadAttention(
                    embed_dim=d_2d_feat,
                    num_heads=config.xattn_rot_num_heads,
                    dropout=config.dropout_p,
                    batch_first=True,
                )
                self.xattn_rot_norm = nn.LayerNorm(d_2d_feat)
            self.per_joint_2d_encoder = nn.Sequential(
                nn.Linear(fourier_2d_dim, d_2d_feat),
                Activation(),
                nn.Linear(d_2d_feat, d_2d_feat),
            )

            # Encoder for joints without rotation (wrist + 5 fingertips = 6 joints)
            # Only 2D info: 6 × d_2d_feat → d_latent
            self.no_rot_joints_encoder = nn.Sequential(
                nn.Linear(6 * d_2d_feat, config.d_latent),
                Activation(),
                nn.Linear(config.d_latent, config.d_latent),
            )
            if config.predict_global_pose:
                self.global_pose_head = GlobalPoseHead(
                    d_latent=config.pose2d_d_latent,
                    num_layers=config.global_pose_num_layers,
                    num_heads=config.pose2d_num_heads,
                    d_feedforward=config.pose2d_d_feedforward,
                    dropout_p=config.dropout_p,
                    d_cond=config.d_cond_base,
                    d_noise_emb=config.d_noise_emb,
                    activation=config.activation,
                    temporal_k=config.temporal_k,
                )
        # MLP encoders and decoders for each modality we want to denoise.
        if config.using_mat:
            modality_dims: dict[str, int] = {
                "mano_betas": 10,
                "mano_poses_mat": 15 * 9,
            }
            # Aligned encoder input: 15 joints × (9D rotation + d_2d_feat)
            if config.using_img_feat:
                aligned_pose_dim = 15 * (9 + self.d_2d_feat)  # 15 × 41 = 615
            else:
                aligned_pose_dim = 15 * 9  # fallback to original
        else:
            modality_dims: dict[str, int] = {
                "mano_betas": 10,
                "mano_poses": 15 * 3,
            }
            aligned_pose_dim = 15 * 3  # no alignment for axis-angle mode
        assert sum(modality_dims.values()) == self.get_d_state()

        # Standard encoders for betas (no 2D alignment needed)
        self.encoders = nn.ModuleDict({
            "mano_betas": nn.Sequential(
                nn.Linear(10, config.d_latent),
                Activation(),
                nn.Linear(config.d_latent, config.d_latent),
                Activation(),
                nn.Linear(config.d_latent, config.d_latent),
            )
        })

        # Aligned pose encoder: takes rotation + 2D features
        pose_key = "mano_poses_mat" if config.using_mat else "mano_poses"
        self.aligned_pose_encoder = nn.Sequential(
            nn.Linear(aligned_pose_dim, config.d_latent),
            Activation(),
            nn.Linear(config.d_latent, config.d_latent),
            Activation(),
            nn.Linear(config.d_latent, config.d_latent),
        )
        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(normalized_shape=config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            }
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Linear(config.d_cond, config.d_latent)

        # Noise embedder.
        self.noise_emb = nn.Embedding(
            # index 0 will be t=1
            # index 999 will be t=1000
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        self.noise_emb_token_proj = (
            nn.Linear(config.d_noise_emb, config.d_latent, bias=False)
            if config.noise_conditioning == "token"
            else None
        )

        # Encoder / decoder layers.
        # Inputs are conditioning (current noise level, observations); output
        # is encoded conditioning information.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=False,  # No conditioning for encoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.encoder_layers)
            ]
        )
        # Decoder: alternating TemporalCrossAttn (Q=noise, KV=cond) and
        # TemporalSelfAttn (Q=K=V=noise) blocks — interleaved crossover.
        # decoder_layers = total attention layers. Half cross, half self.
        num_decoder_blocks = max(2, config.decoder_layers)
        # Ensure even count so cross and self pair up.
        if num_decoder_blocks % 2 == 1:
            num_decoder_blocks += 1
        self.decoder_st_blocks = nn.ModuleList()
        for i in range(num_decoder_blocks):
            cls = TemporalCrossAttnBlock if (i % 2 == 0) else TemporalSelfAttnBlock
            self.decoder_st_blocks.append(cls(
                d_latent=config.d_latent,
                d_noise_emb=config.d_noise_emb,
                num_heads=config.num_heads,
                d_feedforward=config.d_feedforward,
                dropout_p=config.dropout_p,
                activation=config.activation,
                use_rope=config.positional_encoding == "rope",
            ))

    def get_d_state(self) -> int:
        return HandDenoiseTraj.get_packed_dim(self.config.using_mat)

    def forward(
        self,
        x_t_packed: Float[Tensor, "batch time state_dim"],
        t: Float[Tensor, "batch"],
        *,
        rel_palm_pose: Float[Tensor, "batch time 3"],
        # extracted feat from ViT H
        project_output_rotmats: bool,
        # Observed hand positions, relative to the CPF.
        img_feat: Float[Tensor, "batch time 1280"]| None = None,
        mask: Bool[Tensor, "batch time"] | None,
        # Mask for when to drop out / keep conditioning information.
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
        conds: None,
        noisy_joint_2d: Float[Tensor, "batch time num_joints 2"] | None = None,
        clean_joint_2d: Float[Tensor, "batch time num_joints 2"] | None = None,
        bbox_mask: Float[Tensor, "batch time 1 h w"] | None = None,
        noisy_translation: Float[Tensor, "batch time 3"] | None = None,
        noisy_ori_mat: Float[Tensor, "batch time 9"] | None = None,
        cond_confidence: Float[Tensor, "batch time num_joints 1"] | None = None,
        img_feat_crop: Float[Tensor, "batch time 768 16 16"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Predict a denoised trajectory. Note that `t` refers to a noise
        level, not a timestep. `img_feat_crop` (dual-image mode) is accepted for
        call-signature compatibility with HandDenoiserPerJoint but unused here."""
        config = self.config

        x_t = HandDenoiseTraj.unpack(x_t_packed, using_mat=config.using_mat, mano_side=conds.mano_side)
        batch, time, _, _ = x_t.mano_poses.shape

        # Encode betas (no 2D alignment needed)
        betas_encoded = self.encoders["mano_betas"](x_t.mano_betas.reshape(batch, time, -1))

        # Pose encoding depends on whether we have 2D alignment
        pose_input = (x_t.mano_poses_mat if config.using_mat else x_t.mano_poses)
        pose_flat = pose_input.reshape(batch, time, -1)  # (B, T, 135) or (B, T, 45)

        # Embed the noise level.
        noise_emb = self.noise_emb(t - 1)

        # Prepare conditioning information.
        cond = config.make_cond(
            rel_palm_pose = rel_palm_pose,
            conds = conds
        )

        # --- Compute compressed time and helpers early ---
        k = config.temporal_k
        pad_t = (k - time % k) % k if k > 1 else 0
        def _tc_stride(x):
            """Temporal compress via striding: (B, T, D) → (B, T/k, D). Works for any D."""
            if k <= 1:
                return x
            return x[:, ::k, :][:, :((time + pad_t) // k), :]

        # Compress conditioning to T/k. Noisy input stays at full T.
        cond_c = _tc_stride(cond)  # (B, T/k, d_cond_base)
        time_c_3d = cond_c.shape[1]

        pose_2d = None
        confidence = None
        pred_translation = None
        pred_ori_mat = None
        global_img_feat = None
        if config.using_img_feat:
            if config.predict_2Dpose:
                # Pass make_cond output as conditioning to the 2D pose decoder.
                pose2d_cond = cond.reshape(batch * time, -1)  # (B*T, d_cond_base)

                need_two_passes = (clean_joint_2d is not None) and (not config.pose2d_regression)

                if need_two_passes:
                    # Cache CNN features once for both noisy + clean passes.
                    cached_tokens, cached_global_feat = self.pose2d_decoder.encode_image(
                        img_feat.float(), bbox_mask=bbox_mask,
                    )
                    # Noisy pass with gradients.
                    pose_2d, confidence, joint_feat, global_img_feat, tokens_2d = self.pose2d_decoder(
                        img_feat.float(), noisy_joint_2d=noisy_joint_2d, cond=pose2d_cond,
                        noise_emb=noise_emb,
                        cached_tokens=cached_tokens, cached_global_feat=cached_global_feat,
                    )
                    # Clean pass (no grad): just provides conditioning features for 3D.
                    # Use t=1 noise embedding for clean 2D input (minimum noise level).
                    # noise_emb uses (t-1) as index, so t=1 maps to index 0 (cleanest).
                    clean_noise_emb = self.noise_emb(torch.zeros_like(t))  # index 0 = t=1 = cleanest
                    with torch.no_grad():
                        _, _, _, clean_global_feat, _ = self.pose2d_decoder(
                            img_feat.float(), noisy_joint_2d=clean_joint_2d, cond=pose2d_cond,
                            noise_emb=clean_noise_emb,
                            cached_tokens=cached_tokens.detach(),
                            cached_global_feat=cached_global_feat.detach(),
                        )
                    cond_global_feat = clean_global_feat
                    # use the clean GT 2D directly as the noisy-input conditioning
                    cond_pose_2d = clean_joint_2d
                else:
                    # Single-pass: CNN runs inside forward (keeps entire graph in one call).
                    pose2d_input = noisy_joint_2d if not config.pose2d_regression else None
                    pose_2d, confidence, joint_feat, global_img_feat, tokens_2d = self.pose2d_decoder(
                        img_feat.float(), noisy_joint_2d=pose2d_input, cond=pose2d_cond,
                        bbox_mask=bbox_mask, noise_emb=noise_emb,
                    )
                    cond_global_feat = global_img_feat
                    # For 2D-3D encoding: use GT joint_2d if available, otherwise use predicted
                    cond_pose_2d = clean_joint_2d if clean_joint_2d is not None else pose_2d

                # 2D-3D aligned encoding: input to encoders
                # Align 21 2D joints with 15 rotation joints for encoder input
                pose_2d_joints = cond_pose_2d  # (B, T, 21, 2)
                n_joints = self.num_pose2d_joints
                rot_joint_idx = self.rotation_to_output_joint  # maps rot 0-14 → out joints

                # Fourier encode all 21 2D joints
                pose_2d_fourier = fourier_encode(
                    pose_2d_joints.reshape(batch * time * n_joints, 2),
                    freqs=self.pose2d_fourier_freqs
                ).reshape(batch, time, n_joints, -1)  # (B, T, 21, fourier_2d_dim)

                # Per-joint 2D encoding for all 21 joints → d_2d_feat each
                joint_2d_features = self.per_joint_2d_encoder(pose_2d_fourier)  # (B, T, 21, d_2d_feat)

                # Weight 2D features by confidence (visibility mask)
                # This down-weights occluded/uncertain joints in 2D-3D alignment
                # Use external cond_confidence if provided (e.g., from Stage 1), else use predicted
                conf_for_mask = cond_confidence if cond_confidence is not None else confidence
                if conf_for_mask is not None:
                    joint_2d_features = joint_2d_features * conf_for_mask  # (B, T, 21, d_2d_feat) * (B, T, 21, 1)

                # Get 2D features for joints WITH rotation params (15 joints)
                if self.use_xattn_rot_to_2d:
                    # Cross-attention: each of 15 rotation queries attends to all 21 2D features
                    bt = batch * time
                    kv = joint_2d_features.reshape(bt, 21, -1)  # (B*T, 21, d_2d_feat)
                    q = self.rot_queries_xattn.unsqueeze(0).expand(bt, -1, -1)  # (B*T, 15, d_2d_feat)
                    attended, _ = self.xattn_rot_to_2d(q, kv, kv)  # (B*T, 15, d_2d_feat)
                    joint_2d_with_rot = self.xattn_rot_norm(attended).reshape(batch, time, 15, -1)
                else:
                    joint_2d_with_rot = joint_2d_features[:, :, rot_joint_idx, :]  # (B, T, 15, d_2d_feat)

                # Get 2D features for joints WITHOUT rotation params (6 joints)
                no_rot_idx = torch.tensor([0, 4, 8, 12, 16, 20], device=pose_2d_joints.device)
                joint_2d_no_rot = joint_2d_features[:, :, no_rot_idx, :]  # (B, T, 6, d_2d_feat)

                # Concatenate 2D features with 9D rotation for aligned encoder input
                # aligned_pose_input: 15 × (9 + d_2d_feat) per timestep
                mano_poses_mat = x_t.mano_poses_mat  # (B, T, 15, 9)
                aligned_pose_input = torch.cat([mano_poses_mat, joint_2d_with_rot], dim=-1)  # (B, T, 15, 9+d_2d_feat)
                aligned_pose_flat = aligned_pose_input.reshape(batch, time, -1)  # (B, T, 15*(9+d_2d_feat))

                # Encode aligned pose features
                pose_encoded = self.aligned_pose_encoder(aligned_pose_flat)  # (B, T, d_latent)

                # Encode non-rotation joints (only 2D info)
                no_rot_flat = joint_2d_no_rot.reshape(batch, time, -1)  # (B, T, 6*d_2d_feat)
                no_rot_encoded = self.no_rot_joints_encoder(no_rot_flat)  # (B, T, d_latent)

                # x_t_encoded = betas + aligned_pose + non_rotation_joints
                x_t_encoded = betas_encoded + pose_encoded + no_rot_encoded

                # cond_c gets only the global image feature (joint_feat removed).
                global_feat_c = _tc_stride(cond_global_feat)  # (B, T/k, global_feat_dim)
                if config.zero_img_feat_in_3d_cond:
                    global_feat_c = torch.zeros_like(global_feat_c)
                cond_c = torch.cat((cond_c, global_feat_c), dim=-1)

                # Global pose prediction — stride tokens to T/k for GlobalPoseHead.
                if config.predict_global_pose:
                    tokens_c = _tc_stride(
                        tokens_2d.reshape(batch, time, 64, -1)
                    ).reshape(batch * time_c_3d, 64, -1)
                    cond_base_c = _tc_stride(cond[:, :, :config.d_cond_base]).reshape(batch * time_c_3d, -1)
                    noisy_trans_c = _tc_stride(noisy_translation) if noisy_translation is not None else None
                    noisy_ori_c = _tc_stride(noisy_ori_mat) if noisy_ori_mat is not None else None

                    pred_translation, pred_ori_mat, global_pose_feat = self.global_pose_head(
                        tokens_c, batch, time_c_3d, cond=cond_base_c,
                        noisy_translation=noisy_trans_c,
                        noisy_ori_mat=noisy_ori_c,
                        noise_emb=noise_emb,
                    )
                    # Outputs already at full T (heads predict k frames each), trim padding.
                    pred_translation = pred_translation[:, :time, :]
                    pred_ori_mat = pred_ori_mat[:, :time, :]
                    global_pose_feat = global_pose_feat[:, :time, :]
                    cond_c = torch.cat((cond_c, _tc_stride(global_pose_feat)), dim=-1)
            else:
                # predict_2Dpose=False: use GT 2D for 2D-3D encoding, no 2D prediction.
                pose2d_cond = cond.reshape(batch * time, -1)

                # Get global image features (no 2D pose prediction)
                cached_tokens, cached_global_feat = self.pose2d_decoder.encode_image(
                    img_feat.float(), bbox_mask=bbox_mask,
                )
                cond_global_feat = cached_global_feat

                # Use GT 2D joints for 2D-3D encoding (same as predict_2Dpose=True with clean_joint_2d)
                if clean_joint_2d is not None:
                    pose_2d_joints = clean_joint_2d  # (B, T, 21, 2)
                    n_joints = self.num_pose2d_joints
                    rot_joint_idx = self.rotation_to_output_joint

                    # Fourier encode all 21 2D joints
                    pose_2d_fourier = fourier_encode(
                        pose_2d_joints.reshape(batch * time * n_joints, 2),
                        freqs=self.pose2d_fourier_freqs
                    ).reshape(batch, time, n_joints, -1)

                    # Per-joint 2D encoding
                    joint_2d_features = self.per_joint_2d_encoder(pose_2d_fourier)

                    # Weight 2D features by confidence if provided
                    if cond_confidence is not None:
                        joint_2d_features = joint_2d_features * cond_confidence

                    # Get 2D features for joints WITH rotation params (15 joints)
                    if self.use_xattn_rot_to_2d:
                        bt = batch * time
                        kv = joint_2d_features.reshape(bt, 21, -1)
                        q = self.rot_queries_xattn.unsqueeze(0).expand(bt, -1, -1)
                        attended, _ = self.xattn_rot_to_2d(q, kv, kv)
                        joint_2d_with_rot = self.xattn_rot_norm(attended).reshape(batch, time, 15, -1)
                    else:
                        joint_2d_with_rot = joint_2d_features[:, :, rot_joint_idx, :]

                    # Get 2D features for joints WITHOUT rotation params (6 joints)
                    no_rot_idx = torch.tensor([0, 4, 8, 12, 16, 20], device=pose_2d_joints.device)
                    joint_2d_no_rot = joint_2d_features[:, :, no_rot_idx, :]

                    # Concatenate 2D features with 9D rotation
                    mano_poses_mat = x_t.mano_poses_mat
                    aligned_pose_input = torch.cat([mano_poses_mat, joint_2d_with_rot], dim=-1)
                    aligned_pose_flat = aligned_pose_input.reshape(batch, time, -1)
                    pose_encoded = self.aligned_pose_encoder(aligned_pose_flat)

                    # Encode non-rotation joints
                    no_rot_flat = joint_2d_no_rot.reshape(batch, time, -1)
                    no_rot_encoded = self.no_rot_joints_encoder(no_rot_flat)

                    x_t_encoded = betas_encoded + pose_encoded + no_rot_encoded
                else:
                    # Fallback: zero-pad if no GT 2D available
                    mano_poses_mat = x_t.mano_poses_mat
                    zero_2d_feat = torch.zeros(batch, time, 15, self.d_2d_feat, device=mano_poses_mat.device, dtype=mano_poses_mat.dtype)
                    aligned_pose_input = torch.cat([mano_poses_mat, zero_2d_feat], dim=-1)
                    aligned_pose_flat = aligned_pose_input.reshape(batch, time, -1)
                    pose_encoded = self.aligned_pose_encoder(aligned_pose_flat)
                    zero_no_rot = torch.zeros(batch, time, 6 * self.d_2d_feat, device=mano_poses_mat.device, dtype=mano_poses_mat.dtype)
                    no_rot_encoded = self.no_rot_joints_encoder(zero_no_rot)
                    x_t_encoded = betas_encoded + pose_encoded + no_rot_encoded

                # Add global image feature to conditioning
                global_feat_c = _tc_stride(cond_global_feat)
                if config.zero_img_feat_in_3d_cond:
                    global_feat_c = torch.zeros_like(global_feat_c)
                cond_c = torch.cat((cond_c, global_feat_c), dim=-1)
        else:
            # No image features: fallback to original encoding (just rotation, no 2D)
            x_t_encoded = betas_encoded + self.aligned_pose_encoder(pose_flat)
        # Randomly drop out conditioning information; this serves as a
        # regularizer that aims to improve sample diversity.
        if cond_dropout_keep_mask is not None:
            assert cond_dropout_keep_mask.shape == (batch,)
            cond_c = cond_c * cond_dropout_keep_mask[:, None, None]

        # Encoder (conditions) at T/k, decoder (noisy input) at full T.
        if config.positional_encoding == "rope":
            pos_enc_cond = 0
            pos_enc_dec = 0
        elif config.positional_encoding == "transformer":
            pos_enc_cond = make_positional_encoding(
                d_latent=config.d_latent, length=time_c_3d, dtype=cond_c.dtype,
            )[None, ...].to(x_t_encoded.device)
            pos_enc_dec = make_positional_encoding(
                d_latent=config.d_latent, length=time, dtype=x_t_encoded.dtype,
            )[None, ...].to(x_t_encoded.device)
        else:
            assert_never(config.positional_encoding)
        encoder_out = self.latent_from_cond(cond_c) + pos_enc_cond  # (B, T/k, 512)
        decoder_out = x_t_encoded + pos_enc_dec                     # (B, T, 512)
        time_3d = time_c_3d

        # Prepend noise embedding token to both encoder and decoder inputs.
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)[:, None, :]
            encoder_out = torch.cat([noise_emb_token, encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token, decoder_out], dim=1)
            num_tokens = time_3d + 1
        else:
            num_tokens = time_3d

        # Build encoder attention mask at compressed time.
        if mask is None:
            attn_mask = None
        else:
            mask_c = mask[:, ::k][:, :time_3d] if k > 1 else mask
            if self.noise_emb_token_proj is not None:
                mask_c = torch.cat([mask_c.new_ones((batch, 1)), mask_c], dim=1)
            attn_mask = mask_c[:, None, None, :].repeat(1, 1, num_tokens, 1)

        # Forward pass through transformer at compressed time T/k.
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, attn_mask, noise_emb=noise_emb)
        # Remove noise token before decoder blocks.
        has_noise_token = self.noise_emb_token_proj is not None
        if has_noise_token:
            encoder_body = encoder_out[:, 1:, :]  # (B, T/k, 512) — compressed cond
            decoder_body = decoder_out[:, 1:, :]  # (B, T, 512) — full T noise input
        else:
            encoder_body = encoder_out
            decoder_body = decoder_out

        # Decoder at full T, encoder at T/k — asymmetric temporal cross-attention.
        for block in self.decoder_st_blocks:
            decoder_body = block(
                decoder_body, encoder_body, batch, time,
                noise_emb=noise_emb, time_kv=time_3d,
            )

        decoder_out = decoder_body  # (B, T, d_latent)

        # Decode each modality from the shared latent.
        packed_output = torch.cat(
            [
                modality_decoder(decoder_out).reshape((batch, time, 15 * 3))
                if key == "mano_poses"
                else modality_decoder(decoder_out)
                for key, modality_decoder in self.decoders.items()
            ],
            dim=-1,
        )
        return packed_output, pose_2d, confidence, pred_translation, pred_ori_mat


@cache
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
            q, k, v, dropout_p=config.dropout_p if self.training else 0.0, attn_mask=attn_mask
        )
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
            q, k, v, dropout_p=config.dropout_p if self.training else 0.0, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
