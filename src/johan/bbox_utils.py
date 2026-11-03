"""Canonical bbox utilities — the SINGLE source of truth for 2D-pose bbox
normalization geometry and image cropping by per-frame bboxes.

Previously duplicated (and comment-drifted) across train_hand_motion_prior.py,
inference_hand.py and infer_with_bbox.py; those modules now import from here
(and re-export the names for backwards compatibility). Bodies are VERBATIM
from train_hand_motion_prior.py (the training-side definitions are the
convention anchors); compute_global_bbox_from_per_frame comes verbatim from
inference_hand.py (legacy-global-bbox support).

Convention notes:
  - bbox params are pixel coords [min_x, min_y, max_x, max_y];
  - 2D normalization boxes are NEVER clipped to image bounds;
  - joint_2d normalization: (px - min)/range * 2 - 1.
"""
from __future__ import annotations

import torch


def square_scale_bbox(
    min_xy: torch.Tensor,
    max_xy: torch.Tensor,
    square: bool = True,
    scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Square (side = max(w, h)) and/or scale a bbox about its centre — the
    training GT-bbox convention (gt_bbox_square + gt_bbox_scale), UNCLIPPED.
    Operates on (..., 2) min/max tensors."""
    center = (min_xy + max_xy) * 0.5
    wh = max_xy - min_xy
    if square:
        side = wh.max(dim=-1, keepdim=True).values
        wh = side.expand_as(wh)
    wh = wh * scale
    return center - wh * 0.5, center + wh * 0.5


def generate_bbox_mask_per_frame_gpu(
    norm_params_per_frame: torch.Tensor,  # (B, T, 4) [min_x, min_y, max_x, max_y]
    img_shape: torch.Tensor,  # (B, 2)   [H, W]
    min_overlap_ratio: float = 0.0,
) -> torch.Tensor:
    """Per-frame bbox mask, fully vectorized on GPU.

    Returns:
        (B, T, 1, H_max, W_max) float mask — 1 inside visible bbox, 0 outside.
        Frames whose bbox/image overlap < min_overlap_ratio are zeroed out.
        min_overlap_ratio <= 0 keeps ANY frame with positive overlap (overlap > 0
        -> valid); a positive ratio requires clip_area/full_area >= ratio.
        (Default changed 0.5 -> 0.0 on 2026-06-23: keep partially-visible frames.)
    """
    B, T, _ = norm_params_per_frame.shape
    device = norm_params_per_frame.device
    H = int(img_shape[:, 0].max().item())
    W = int(img_shape[:, 1].max().item())

    flat = norm_params_per_frame.reshape(B * T, 4)
    grid_y = torch.arange(H, device=device).float()[None, None, :, None]
    grid_x = torch.arange(W, device=device).float()[None, None, None, :]
    min_x = flat[:, 0].view(B * T, 1, 1, 1)
    min_y = flat[:, 1].view(B * T, 1, 1, 1)
    max_x = flat[:, 2].view(B * T, 1, 1, 1)
    max_y = flat[:, 3].view(B * T, 1, 1, 1)

    full_area = (max_x - min_x).clamp(min=0) * (max_y - min_y).clamp(min=0)
    clip_area = (max_x.clamp(max=W) - min_x.clamp(min=0)).clamp(min=0) * (
        max_y.clamp(max=H) - min_y.clamp(min=0)
    ).clamp(min=0)
    if min_overlap_ratio <= 0:
        valid = (full_area > 0) & (clip_area > 0)                       # overlap > 0 -> valid
    else:
        valid = (full_area > 0) & (clip_area / full_area.clamp(min=1e-6) >= min_overlap_ratio)

    inside = (grid_x >= min_x) & (grid_x <= max_x) & (grid_y >= min_y) & (grid_y <= max_y)
    mask = inside.float() * valid.float()
    return mask.reshape(B, T, 1, H, W)




def crop_images_by_bbox(
    images: torch.Tensor,
    bbox_params: torch.Tensor,
    output_size: int = 256,
    padding_mode: str = "zeros",
    bbox_margin: float = 0.2,
    square_bbox: bool = False,
) -> torch.Tensor:
    """Crop images based on per-frame bounding boxes and resize to output_size.

    Uses grid_sample for differentiable cropping with bilinear interpolation.

    Args:
        images: (B, T, H, W, C) or (B, T, C, H, W) RGB images
        bbox_params: (B, T, 4) bounding box [min_x, min_y, max_x, max_y] in pixel coords
        output_size: Target size for cropped images (square)
        padding_mode: 'zeros' or 'border' for out-of-bounds regions

    Returns:
        Cropped images: (B, T, C, output_size, output_size)
    """
    # Handle NHWC format
    if images.shape[-1] == 3:
        images = images.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)

    B, T, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # Flatten batch and time for grid_sample
    images_flat = images.reshape(B * T, C, H, W)
    bbox_flat = bbox_params.reshape(B * T, 4)

    # Extract bbox corners
    min_x = bbox_flat[:, 0]  # (BT,)
    min_y = bbox_flat[:, 1]
    max_x = bbox_flat[:, 2]
    max_y = bbox_flat[:, 3]

    # Add margin to bbox (each side fraction = bbox_margin, total scale = 1 + 2*bbox_margin)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    margin_x = bbox_w * bbox_margin
    margin_y = bbox_h * bbox_margin
    min_x = min_x - margin_x
    min_y = min_y - margin_y
    max_x = max_x + margin_x
    max_y = max_y + margin_y

    # Optionally square-pad around the bbox centre (WiLoR-style: side = max(w, h)).
    if square_bbox:
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        side = torch.maximum(max_x - min_x, max_y - min_y)
        half = side / 2.0
        min_x = cx - half
        min_y = cy - half
        max_x = cx + half
        max_y = cy + half

    # Create normalized sampling grid [-1, 1]
    # grid_sample expects coordinates in [-1, 1] where -1 is left/top, 1 is right/bottom
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, output_size, device=device, dtype=dtype),
        torch.linspace(0, 1, output_size, device=device, dtype=dtype),
        indexing='ij',
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (out_size, out_size, 2)
    grid = grid.unsqueeze(0).expand(B * T, -1, -1, -1)  # (BT, out_size, out_size, 2)

    # Map grid [0, 1] to pixel bbox coordinates, then normalize to [-1, 1] for grid_sample
    # pixel_x = min_x + grid_x * (max_x - min_x)
    # normalized_x = (pixel_x / (W - 1)) * 2 - 1
    bbox_w = (max_x - min_x).view(B * T, 1, 1, 1)
    bbox_h = (max_y - min_y).view(B * T, 1, 1, 1)
    min_x = min_x.view(B * T, 1, 1, 1)
    min_y = min_y.view(B * T, 1, 1, 1)

    # Scale grid to pixel coordinates
    pixel_x = min_x + grid[..., 0:1] * bbox_w
    pixel_y = min_y + grid[..., 1:2] * bbox_h

    # Normalize to [-1, 1] for grid_sample
    norm_x = (pixel_x / (W - 1)) * 2 - 1
    norm_y = (pixel_y / (H - 1)) * 2 - 1
    sample_grid = torch.cat([norm_x, norm_y], dim=-1)  # (BT, out_size, out_size, 2)

    # Sample using grid_sample (bilinear interpolation)
    cropped = torch.nn.functional.grid_sample(
        images_flat.float(),
        sample_grid.float(),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    )

    # Reshape back to (B, T, C, out_size, out_size)
    cropped = cropped.reshape(B, T, C, output_size, output_size).to(dtype)
    return cropped


def compute_global_bbox_from_per_frame(
    norm_params_per_frame: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute global bbox params from per-frame params.
    
    This is for backwards compatibility with models trained with the buggy design
    where joint_2d was normalized with global bbox but bbox_mask was per-frame.
    
    Args:
        norm_params_per_frame: (B, T, 4) or (T, 4) per-frame [min_x, min_y, max_x, max_y]
        mask: (B, T) or (T,) optional validity mask
        
    Returns:
        Global bbox params: (B, 4) or (4,) [min_x, min_y, max_x, max_y]
    """
    squeeze_batch = False
    if norm_params_per_frame.dim() == 2:
        norm_params_per_frame = norm_params_per_frame.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        squeeze_batch = True
    
    B, T, _ = norm_params_per_frame.shape
    min_xy = norm_params_per_frame[:, :, :2]  # (B, T, 2)
    max_xy = norm_params_per_frame[:, :, 2:]  # (B, T, 2)
    
    if mask is not None:
        mask_expand = mask[:, :, None].float().to(norm_params_per_frame.device) # (B, T, 1)
        inf = torch.tensor(float('inf'), device=norm_params_per_frame.device)
        min_xy = torch.where(mask_expand > 0, min_xy, inf)
        max_xy = torch.where(mask_expand > 0, max_xy, -inf)
    
    global_min = min_xy.min(dim=1).values  # (B, 2)
    global_max = max_xy.max(dim=1).values  # (B, 2)
    global_params = torch.cat([global_min, global_max], dim=-1)  # (B, 4)
    
    if squeeze_batch:
        global_params = global_params.squeeze(0)
    return global_params

