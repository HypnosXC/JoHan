from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from .tensor_dataclass import TensorDataclass
class HandInput(TensorDataclass):
    # T_world_root: Float[Tensor, "*#batch timesteps 7"]
    # """Transformation from the world frame to the root frame at each timestep."""

    mano_betas: Float[Tensor, "*#batch 10"]
    """Hand shape parameters."""

    mano_pose: Float[Tensor, "*#batch timesteps 51"]

    joint_2d: Float[Tensor, "*#batch timesteps 21 2"]
    mano_joint_3d: Float[Tensor, "*#batch timesteps 21 3"]

    rgb_frames: Float[Tensor, "*#batch timesteps h w 3"]

    intrinsics: Float[Tensor, "*#batch 4"]
    """Camera intrinsics: fx, fy, ppx, ppy."""

    extrinsics: Float[Tensor, "*#batch 3 4"]
    """Camera extrinsics: 3x4 matrix."""

    # hand_quats: Float[Tensor, "*#batch timesteps 30 4"] | None
    # """Local orientations for each hand joint."""
    mask: Bool[Tensor, "*#batch timesteps"]
    """Mask to support variable-length sequence."""

    mano_side: Float[Tensor, "*#batch 1"]
    """Side of the hand, either 0->'left' or 1->'right'."""

    img_feature: Float[Tensor, "*#batch timesteps 768 16 16"]

    # img_shape: tuple[int, int]
    img_shape: Float[Tensor, "*#batch 2"]
    """Height and width of the input images."""

    joint_2d_norm_params_per_frame: Float[Tensor, "*#batch timesteps 4"]
    """Per-frame min/max params for denormalizing joint_2d: [min_x, min_y, max_x, max_y]."""

    bbox_mask: Float[Tensor, "*#batch 1 h w"]
    """Bounding box mask. HDF5 may provide one clip-level mask; WebDataset provides per-frame masks."""

    joint_visibility: Bool[Tensor, "*#batch timesteps 21"] | None = None
    """Physical per-joint visibility. True means visible, False means occluded/invalid.
    Used as ground truth for 2D joint confidence prediction."""

    # Optional cropped-image features (dual-image mode): full-image features go in
    # img_feature (patches + 3D cond); these cropped-hand features feed ONLY the 2D
    # decoder's global-feature query bias (gf_bias), restoring fine hand detail
    # while keeping full-image spatial localization.
    img_feature_crop: Float[Tensor, "*#batch timesteps 768 16 16"] | None = None

    # video_name: str

