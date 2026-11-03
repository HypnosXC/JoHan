from __future__ import annotations
import os, time
import numpy as np
import torch
from tqdm.auto import tqdm
from . import hand_network
from .batch import HandInput
from .noise import CosineNoiseScheduleConstants
from .bbox_utils import crop_images_by_bbox, square_scale_bbox, generate_bbox_mask_per_frame_gpu
_DINO_EXTRACTOR = None
def _get_dino_extractor(device):
    if _DINO_EXTRACTOR is None:
        raise RuntimeError("Load the DINO backbone before sampling")
    return _DINO_EXTRACTOR

def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]


def linear_ts(num_steps: int = 50) -> np.ndarray:
    """Uniform timestep schedule for flow matching (from 1.0 to 0.0)."""
    return np.linspace(1.0, 0.0, num_steps + 1)


def sin_ts(num_steps: int = 50) -> np.ndarray:
    """Sin-shifted timestep schedule: warp the data-fraction s with sin(s*pi/2).

    s = linspace(0,1) (0=noise -> 1=data); shifted = sin(s*pi/2) is flat near
    s=1, so consecutive shifted values bunch toward the clean/data end. Returned
    as the noise-fraction schedule t_curr = 1 - sin(s*pi/2), running 1.0 (noise)
    -> 0.0 (clean) like linear_ts but with finer steps near the clean pose.
    Endpoints are exactly 1.0 and 0.0, so it is a drop-in for linear_ts."""
    s = np.linspace(0.0, 1.0, num_steps + 1)
    return 1.0 - np.sin(s * (np.pi / 2.0))


def _ts_schedule(num_steps: int = 10) -> np.ndarray:
    """sin_ts (denser near clean) by DEFAULT; set HAND_SIN_TS=0 for linear_ts.
    Default eval is now 10-step sin (fast; matches the clean-side flow recipe)."""
    return linear_ts(num_steps) if os.environ.get("HAND_SIN_TS", "1") == "0" else sin_ts(num_steps)


def build_per_frame_bbox_mask_for_inference(
    train_batch: HandInput,
    device: torch.device,
    mask_square: bool = False,
    mask_scale: float = 1.0,
) -> torch.Tensor | None:
    """Build per-frame bbox conditioning mask for inference.

    mask_square / mask_scale: decouple the MASK box from the 2D-norm box (square +
    scale the mask box only, e.g. 1/0.79 to match the WiLoR detector), matching
    training's bbox_mask_square / bbox_mask_scale. The joint_2d_norm_params (used
    for 2D denorm / metrics) are left untouched (raw box).
    """
    img_shape = getattr(train_batch, "img_shape", None)
    if img_shape is None:
        return None

    img_shape = img_shape.to(device)

    # Use per-frame norm_params directly
    norm_params_per_frame = getattr(train_batch, "joint_2d_norm_params_per_frame", None)
    if norm_params_per_frame is not None:
        _np = norm_params_per_frame.to(device)
        if mask_square or mask_scale != 1.0:
            _mn, _mx = square_scale_bbox(_np[..., :2], _np[..., 2:], square=mask_square, scale=mask_scale)
            _np = torch.cat([_mn, _mx], dim=-1)
        return generate_bbox_mask_per_frame_gpu(_np, img_shape)

    # Fallback to pre-computed global bbox_mask (expanded to all frames)
    mask = getattr(train_batch, "mask", None)
    bbox_mask = getattr(train_batch, "bbox_mask", None)
    if bbox_mask is None:
        return None
    bbox_mask = bbox_mask.to(device)
    if bbox_mask.dim() == 4 and mask is not None:
        T = mask.shape[1]
        bbox_mask = bbox_mask[:, None].expand(-1, T, -1, -1, -1).contiguous()
    return bbox_mask






def maybe_extract_dino_features(
    img_feature: torch.Tensor,
    device: torch.device,
    bbox_params: torch.Tensor | None = None,
    crop_before_dino: bool = False,
    bbox_margin: float = 0.2,
    square_bbox: bool = False,
) -> torch.Tensor | None:
    """Extract DINO features if input is RGB frames, otherwise return as-is.

    Args:
        img_feature: Either RGB frames (B, T, 3, 256, 256) or (B, T, 256, 256, 3)
                     or precomputed DINO features (B, T, 768, 16, 16)
        device: Target device
        bbox_params: (B, T, 4) bounding box for cropping [min_x, min_y, max_x, max_y]
        crop_before_dino: If True, crop images to bbox before DINO extraction

    Returns:
        DINO features with shape (B, T, 768, 16, 16), or None if input is invalid
    """
    if img_feature is None:
        return None

    img_feature = img_feature.to(device)

    # Check if already DINO features (768 channels) or RGB frames (3 channels)
    if img_feature.dim() == 5:
        # Shape: (B, T, C, H, W) or (B, T, H, W, C)
        if img_feature.shape[3] == 16 and img_feature.shape[4] == 16:
            # Already DINO features (B, T, C, 16, 16); C=768 ViT-B / 1024 ViT-L
            return img_feature.float()
        elif img_feature.shape[-1] == 3:
            # NHWC format - keep as is for potential cropping
            pass
        else:
            # Already NCHW format
            pass

        # Crop images to bbox if requested
        if crop_before_dino and bbox_params is not None:
            if img_feature.shape[-1] == 3:
                # NHWC format
                img_feature = crop_images_by_bbox(
                    img_feature, bbox_params.to(device), output_size=256,
                    bbox_margin=bbox_margin, square_bbox=square_bbox,
                )
            else:
                # Already NCHW, need to convert for cropping then back
                img_feature = img_feature.permute(0, 1, 3, 4, 2).contiguous()  # to NHWC
                img_feature = crop_images_by_bbox(
                    img_feature, bbox_params.to(device), output_size=256,
                    bbox_margin=bbox_margin, square_bbox=square_bbox,
                )
            # crop_images_by_bbox returns NCHW
        elif img_feature.shape[-1] == 3:
            # NHWC format without cropping, permute to NCHW
            img_feature = img_feature.permute(0, 1, 4, 2, 3).contiguous()

        # Now should be (B, T, 3, 256, 256)
        if img_feature.shape[2] == 3:
            B, T = img_feature.shape[:2]
            dino = _get_dino_extractor(device)
            with torch.no_grad():
                # Normalize with ImageNet mean/std
                mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
                frames = img_feature.to(torch.bfloat16)
                if frames.max() > 1.0:
                    frames = frames.mul_(1.0 / 255.0)
                frames = (frames - mean) / std
                # Extract features
                feats = dino(frames.reshape(-1, *frames.shape[2:]))
                # C is 768 (ViT-B) or 1024 (ViT-L); read it from the tensor.
                return feats.reshape(B, T, *feats.shape[1:]).float()
    elif img_feature.dim() == 4:
        # Shape: (T, C, H, W) - single sample without batch dim (C=768 B / 1024 L)
        if img_feature.shape[2] == 16 and img_feature.shape[3] == 16:
            return img_feature.float().unsqueeze(0)
        elif img_feature.shape[1] == 3 or img_feature.shape[-1] == 3:
            bbox_unsqueezed = bbox_params.unsqueeze(0) if bbox_params is not None else None
            return maybe_extract_dino_features(
                img_feature.unsqueeze(0), device, bbox_unsqueezed, crop_before_dino,
                bbox_margin=bbox_margin, square_bbox=square_bbox,
            )

    # Invalid shape (e.g., placeholder tensor) - return None
    return None


# --- WAN VAE feature extraction (online_wan models) -------------------------


def sample_hand(
    denoiser_network,
    train_batch,
    device,
    start_frame: int = 0,
    training_mode: str = "diffusion",
    num_flow_steps: int = 10,
    use_optimal_2d: bool = False,
    use_legacy_global_bbox: bool = False,
    crop_before_dino: bool = False,
    bbox_crop_margin: float = 0.2,
    bbox_crop_square: bool = False,
    crop_global_feat: bool = False,
    infer_window: int = 128,
    bbox_mask_square: bool = False,
    bbox_mask_scale: float = 1.0,
):
    """Run hand denoising sampling; global pose is recovered separately by PnP.

    Args:
        use_legacy_global_bbox: If True, use global bbox for 2D pose denormalization
            (for models trained with buggy design: joint_2d normalized by global bbox
            but bbox_mask was per-frame). Default False uses per-frame bbox.
        crop_before_dino: If True, crop images to hand bbox before DINO extraction.
            Must match the training setting. When True, predicted 2D joints are
            scaled by 1.4 to convert back to the original bbox normalization space
            (20% margin on each side = 1.4x scale).

    Returns:
        (traj, pose_2d_final): HandDenoiseTraj and optional (B, T, 21, 2) 2D joints.
    """
    # REPRODUCIBILITY (2026-07-16): the flow sampler starts from torch.randn with
    # no seed, so eval numbers carried run-to-run noise variance. On high-variance
    # models (e.g. nobbox) that reached ~4 PA and made two eval scripts disagree
    # (inference_hand 9.28 vs infer_with_bbox 13.44 on identical batches -- proven
    # via a 3D dump: GT joints matched 0.0mm, preds differed 5.3mm from noise
    # alone). Seed the global RNG at each call so sampling is deterministic and
    # identical across scripts/runs. HAND_EVAL_SEED overrides (default 0).
    _seed = int(os.environ.get("HAND_EVAL_SEED", "0"))
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    if training_mode == "diffusion":
        noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
            device=device,
        )
        alpha_bar_t = noise_constants.alpha_bar_t
        alpha_t = noise_constants.alpha_t
    else:
        alpha_bar_t = None
        alpha_t = None

    batch, seq_len, _ = train_batch.mano_pose.shape
    seq_len = seq_len - start_frame

    train_batch.to(device)
    using_mat = denoiser_network.config.using_mat
    using_img_feat = denoiser_network.config.using_img_feat
    # pose2d_state_only: the flow state IS the per-joint 2D pose [x, y, (conf)] —
    # there is NO MANO/3D trajectory. The network's x0 output is already the 2D
    # state, so we must NOT unpack/repack it as a MANO HandDenoiseTraj (that
    # asserts a 145-dim state and crashes). These helpers no-op the repack and
    # return None for the (nonexistent) 3D trajectory in that mode.
    state2d = getattr(denoiser_network.config, "pose2d_state_only", False)

    def _repack_x0(x):
        if state2d:
            return x
        return hand_network.HandDenoiseTraj.unpack(
            x, using_mat=using_mat, mano_side=x_0_packed.mano_side
        ).pack(using_mat=using_mat)

    def _traj_from(x):
        if state2d:
            return None
        return hand_network.HandDenoiseTraj.unpack(
            x, using_mat=using_mat, mano_side=x_0_packed.mano_side
        )

    x_0_packed = hand_network.HandDenoiseTraj(
        mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
        mano_poses=train_batch.mano_pose[:, start_frame:, 3:48].reshape(batch, seq_len, 15, 3),
        global_orientation=train_batch.mano_pose[:, start_frame:, 0:3],
        global_translation=train_batch.mano_pose[:, start_frame:, 48:],
        mano_side=train_batch.mano_side.unsqueeze(1).expand(batch, seq_len, -1),
    )
    rel_palm_pose = train_batch.mano_pose[:, start_frame:, 48:].to(device)

    # Get bbox_params for cropping if needed
    bbox_params = getattr(train_batch, "joint_2d_norm_params_per_frame", None)
    if bbox_params is not None:
        bbox_params = bbox_params[:, start_frame:]

    # Try img_feature first, fall back to rgb_frames if img_feature is invalid/placeholder.
    # Route by the run's use_feature (HAND_EVAL_USE_FEATURE, exported from run_config.yaml
    # in main). The old patch-channel guess sent DINOv3 ViT-H/16+ (also 1280-d) to the
    # HaMeR extractor; it is kept only as the fallback for runs without a run_config.
    _extract = maybe_extract_dino_features
    cond_feat = None
    if using_img_feat:
        cond_feat = _extract(
            train_batch.img_feature, device, bbox_params, crop_before_dino,
            bbox_margin=bbox_crop_margin, square_bbox=bbox_crop_square,
        )
        if cond_feat is None and hasattr(train_batch, 'rgb_frames') and train_batch.rgb_frames is not None:
            # img_feature was placeholder, extract from RGB frames instead
            rgb = train_batch.rgb_frames
            if rgb.dim() == 4:
                rgb = rgb.unsqueeze(0)  # Add batch dim if missing
            cond_feat = _extract(
                rgb, device, bbox_params, crop_before_dino,
                bbox_margin=bbox_crop_margin, square_bbox=bbox_crop_square,
            )

    # Dual-image mode: additionally extract CROPPED-hand features for the 2D
    # decoder's gf_bias (full-image cond_feat stays for patches + 3D cond). Mirror
    # training: crop the image to the hand bbox (force crop) and extract.
    cond_feat_crop = None
    if crop_global_feat and using_img_feat and cond_feat is not None:
        src = train_batch.img_feature
        cond_feat_crop = _extract(
            src, device, bbox_params, True,
            bbox_margin=bbox_crop_margin, square_bbox=bbox_crop_square,
        )
        if cond_feat_crop is None and hasattr(train_batch, 'rgb_frames') and train_batch.rgb_frames is not None:
            rgb = train_batch.rgb_frames
            if rgb.dim() == 4:
                rgb = rgb.unsqueeze(0)
            cond_feat_crop = _extract(
                rgb, device, bbox_params, True,
                bbox_margin=bbox_crop_margin, square_bbox=bbox_crop_square,
            )

    # Re-seed AFTER feature extraction: DINO extraction can consume RNG (and the
    # two eval scripts may extract differently), which would desync the sampling
    # noise below despite the function-start seed. Re-seed here so the noise is
    # provably identical across scripts/runs. If predictions STILL differ after
    # this, the divergence is in cond_feat itself (not the noise).
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    x_t_packed = torch.randn((batch, seq_len, denoiser_network.get_d_state()), device=device)
    x_t_list = []
    start_time = None

    # Build per-frame bbox mask for conditioning (B, T, 1, H, W).
    t_bbox_mask = build_per_frame_bbox_mask_for_inference(
        train_batch, device, mask_square=bbox_mask_square, mask_scale=bbox_mask_scale)
    # Ablation: HAND_ZERO_BBOX_MASK=1 forces the bbox_mask conditioning to all
    # zeros (encode_image still adds mask_embed(zeros), i.e. the "no box" embed),
    # to measure how much the model leans on the bbox localization cue at test.
    if os.environ.get("HAND_ZERO_BBOX_MASK") == "1" and t_bbox_mask is not None:
        t_bbox_mask = torch.zeros_like(t_bbox_mask)
        print("[HAND_ZERO_BBOX_MASK] bbox_mask forced to ZERO for this eval", flush=True)
    # BUGFIX: training ZEROS the bbox_mask for crop models (the crop is already the
    # hand region, so the mask carries no info). Match that at inference, else the
    # model gets a non-zero mask it never saw in training -> wrong mask_embed
    # conditioning -> degraded crop_dino/crop_wan predictions.
    if crop_before_dino and t_bbox_mask is not None:
        # Crop settings: bbox mask is meaningless; None == zeros for the model
        # (backward compatible with checkpoints trained on the zeroed mask).
        t_bbox_mask = None

    # Clean 2D for 2D-3D encoding:
    # - If predict_2Dpose=False: always use GT joint_2d for 2D-3D encoding
    # - If use_optimal_2d=True: start with GT for Stage 1, then use denoised 2D for Stage 2
    # - Otherwise (predict_2Dpose=True and not use_optimal_2d): starts as None, network uses predicted
    predict_2d = denoiser_network.config.predict_2Dpose
    if (not getattr(denoiser_network.config, "pure_3d", False)
            and (not predict_2d or use_optimal_2d)):
        # Use GT 2D initially; for use_optimal_2d, this will be updated to denoised 2D after Stage 1
        if hasattr(train_batch, 'joint_2d') and train_batch.joint_2d is not None:
            gt_j2d = train_batch.joint_2d[:, start_frame:].to(device)
            t_clean_j2d = gt_j2d
        else:
            gt_j2d = None
            t_clean_j2d = None
    else:
        # predict_2Dpose=True and single-stage: let network use predicted 2D
        gt_j2d = None
        t_clean_j2d = None

    # --- Helper ---
    def _predict_x0_window(x_t_window, t_int, rel_palm_window, conds_window,
                           noisy_joint_2d_window=None, img_feat_window=None,
                           noisy_trans_window=None, noisy_ori_mat_window=None,
                           bbox_mask_window=None, cond_confidence_window=None):
        # Dual-image: feed the cropped-hand features for gf_bias on the
        # whole-sequence window (regression eval). Shapes match cond_feat when the
        # window is the full sequence; windowed (opt2d/diffusion) crop paths aren't
        # dual-image so they stay None.
        img_feat_crop_window = None
        if (cond_feat_crop is not None and img_feat_window is not None
                and img_feat_window.shape[1] == cond_feat_crop.shape[1]):
            img_feat_crop_window = cond_feat_crop
        x0_pred, pose_2d_pred, conf_pred, pred_trans, pred_ori_mat = denoiser_network.forward(
            x_t_window,
            torch.tensor([t_int], device=device).expand((batch,)),
            rel_palm_pose=rel_palm_window,
            project_output_rotmats=False,
            conds=conds_window,
            img_feat=img_feat_window,
            img_feat_crop=img_feat_crop_window,
            mask=None,
            cond_dropout_keep_mask=torch.ones((batch,), device=device),
            noisy_joint_2d=noisy_joint_2d_window,
            clean_joint_2d=t_clean_j2d,
            bbox_mask=bbox_mask_window,
            noisy_translation=noisy_trans_window,
            noisy_ori_mat=noisy_ori_mat_window,
            cond_confidence=cond_confidence_window,
        )
        return x0_pred, pose_2d_pred, conf_pred, pred_trans, pred_ori_mat

    def _make_conds(start_t, end_t):
        return hand_network.HandDenoiseTraj(
            mano_betas=x_0_packed.mano_betas[:, start_t:end_t, :],
            mano_poses=x_0_packed.mano_poses[:, start_t:end_t, :, :],
            global_orientation=x_0_packed.global_orientation[:, start_t:end_t, :],
            global_translation=x_0_packed.global_translation[:, start_t:end_t, :],
            mano_side=x_0_packed.mano_side[:, start_t:end_t, :],
        ).to(device)

    def _make_overlap_weights():
        overlap_size = 32
        return (
            torch.from_numpy(
                np.minimum(
                    overlap_size,
                    np.minimum(
                        np.arange(1, seq_len + 1),
                        np.arange(1, seq_len + 1)[::-1],
                    ),
                ) / overlap_size,
            ).to(device).to(torch.float32)
        )

    # ---- Flow matching ----
    if training_mode == "flow_matching":
        max_t = denoiser_network.config.max_t
        # Flow time orientation (must mirror training_loss): 'flow' feeds the net
        # the data-fraction index s=1-t_curr (noise->data), 'diffusion' feeds the
        # noise-fraction t_curr. Learned noise_emb so this only relabels rows, but
        # train/infer MUST agree. Helper folds the floor + clamp(1,max_t).
        flow_conv = getattr(denoiser_network.config, "flow_time_convention", "diffusion")
        def _t_index(t_curr: float) -> int:
            frac = (1.0 - t_curr) if flow_conv == "flow" else t_curr
            return max(1, min(int(frac * max_t), max_t))
        num_joints = 21
        conf_final = None  # predicted visibility/confidence
        # conf_in_state: the 2D flow state is 3-channel [x, y, conf] (conf denoised
        # like the coords). The decoder outputs pose_2d (2-ch) + conf (1-ch sigmoid);
        # the conf channel's x0 estimate is the predicted conf, so we re-assemble the
        # 3-ch x0 = cat([pose_2d_pred, conf_pred]) for the flow step.
        conf_in_state = getattr(denoiser_network.config, "pose2d_conf_in_state", False)
        # conf_onehot: conf is a 2-channel one-hot [vis,occ] -> 4-ch 2D state; the
        # head outputs 2 logits, so the flow x0 of the conf channels is softmax(logits).
        conf_onehot = getattr(denoiser_network.config, "pose2d_conf_onehot", False)
        j2d_nch = (4 if conf_onehot else 3) if conf_in_state else 2

        def _j2d_state_x0(p2d_pred, cf_pred):
            if conf_in_state and p2d_pred is not None and cf_pred is not None:
                cf = torch.softmax(cf_pred, dim=-1) if (conf_onehot and cf_pred.shape[-1] == 2) else cf_pred
                return torch.cat([p2d_pred, cf], dim=-1)
            return p2d_pred

        def _flow_step(pred, noisy, t_curr, t_next):
            # Velocity-prediction Euler step. Mathematically identical to the
            # x0-renoise form `pred + (t_next/t_curr)*(noisy - pred)`, but written
            # in the v=(x_pred - z)/(1 - t); z = z + v*dt convention.
            # The repo's t_curr runs 1->0 (noise fraction), so integrate in
            # data-fraction time s = 1 - t_curr (0 at noise -> 1 at data):
            #   s  = 1 - t_curr ;  dt = t_curr - t_next (> 0)
            #   v  = (pred - noisy) / max(1 - s, 1e-5)   # note: 1 - s == t_curr
            #   z  = noisy + v*dt                        # lands on `pred` at t_next->0
            s = 1.0 - t_curr
            dt = t_curr - t_next
            v_pred = (pred - noisy) / max(1.0 - s, 1e-5)
            return noisy + v_pred * dt

        if use_optimal_2d:
            # === Stage 1: Denoise 2D joints only ===
            print("Stage 1: Denoising 2D joints...")
            joint_2d_t = torch.randn((batch, seq_len, num_joints, 2), device=device)
            conf_pred = None  # will be set in loop
            ts_frac_s1 = _ts_schedule(num_flow_steps)
            for i in tqdm(range(len(ts_frac_s1) - 1), desc="Stage 1 (2D)"):
                t_curr, t_next = ts_frac_s1[i], ts_frac_s1[i + 1]
                t_int = _t_index(t_curr)  # floor + flow/diffusion orientation (matches training)
                with torch.inference_mode():
                    _, pose_2d_pred, conf_pred, _, _ = _predict_x0_window(
                        x_t_packed, t_int, rel_palm_pose, _make_conds(0, seq_len),
                        noisy_joint_2d_window=joint_2d_t,
                        img_feat_window=cond_feat,
                        bbox_mask_window=t_bbox_mask,
                    )
                if pose_2d_pred is not None:
                    joint_2d_t = _flow_step(pose_2d_pred, joint_2d_t, t_curr, t_next)

            # Stage 1 complete: use denoised 2D for 2D-3D encoding in Stage 2.
            pose_2d_final = joint_2d_t
            conf_final = conf_pred  # predicted visibility/confidence
            # HAND_GT2D_COND=1: ORACLE mode — condition Stage 2 on GT 2D instead of
            # the Stage-1 denoised 2D (upper-bounds the 2D->3D lift quality).
            import os as _os
            if _os.environ.get("HAND_GT2D_COND") == "1" and gt_j2d is not None:
                if not globals().get("_GT2D_ANNOUNCED", False):
                    globals()["_GT2D_ANNOUNCED"] = True
                    print("[oracle] Stage-2 3D conditioning uses GT 2D (HAND_GT2D_COND=1)", flush=True)
                t_clean_j2d = gt_j2d
            else:
                t_clean_j2d = pose_2d_final  # Use denoised 2D as conditioning for Stage 2

            # === Stage 2: Denoise 3D with clean 2D conditioning ===
            print("Stage 2: Denoising 3D with clean 2D...")
            x_t_packed = torch.randn((batch, seq_len, denoiser_network.get_d_state()), device=device)
            x_t_list = []
            start_time = None

        ts_frac = _ts_schedule(num_flow_steps)
        # For Stage 2 (use_optimal_2d), use denoised 2D from Stage 1 instead of random noise.
        # This matches training where noisy_joint_2d at t→0 approaches clean GT.
        if use_optimal_2d:
            # pose_2d_final was set in Stage 1; use it as the "noisy" 2D (which is actually clean)
            joint_2d_t = pose_2d_final.clone()
        else:
            joint_2d_t = torch.randn((batch, seq_len, num_joints, j2d_nch), device=device)
        trans_t = torch.randn((batch, seq_len, 3), device=device)
        ori_mat_t = torch.randn((batch, seq_len, 9), device=device)
        conf_pred = None  # will be set in loop

        if seq_len > infer_window:
            # Window at the TRAINING clip length (infer_window = subseq_len). For a
            # sub_len model (infer_window<128) the clip is split into INDEPENDENT,
            # NON-overlapping infer_window-frame videos (unit weights, no overlap-
            # stitch) so each is a clean single forward exactly like training and
            # the model never sees >subseq_len frames. Standard models
            # (infer_window=128) keep the 64/overlap-32 stitched windowing.
            if infer_window < 128:
                window_size, overlap_size = infer_window, 0
                canonical_ow = torch.ones(seq_len, device=device, dtype=torch.float32)
            else:
                window_size, overlap_size = 64, 32
                canonical_ow = _make_overlap_weights()

            for i in tqdm(range(len(ts_frac) - 1)):
                t_curr, t_next = ts_frac[i], ts_frac[i + 1]
                t_int = _t_index(t_curr)  # floor + flow/diffusion orientation (matches training)

                with torch.inference_mode():
                    x_0_packed_pred = torch.zeros_like(x_t_packed)
                    j2d_accum = torch.zeros_like(joint_2d_t)
                    trans_accum = torch.zeros_like(trans_t)
                    ori_mat_accum = torch.zeros_like(ori_mat_t)
                    ow = torch.zeros((batch, seq_len, 1), device=device)
                    conf_accum = None  # full-seq conf buffer (built lazily on 1st window with conf)

                    for s in range(0, seq_len, window_size - overlap_size):
                        e = min(s + window_size, seq_len)
                        ow_slice = canonical_ow[None, :e - s, None]
                        ow[:, s:e, :] += ow_slice

                        img_feat_win = cond_feat[:, s:e] if cond_feat is not None else None
                        # Pass confidence from Stage 1 to weight 2D features during 2D-3D alignment
                        conf_win = conf_final[:, s:e, :, :] if conf_final is not None else None
                        temp_x0, temp_j2d, temp_conf, temp_trans, temp_ori_mat = _predict_x0_window(
                            x_t_packed[:, s:e, :], t_int,
                            rel_palm_pose[:, s:e, :], _make_conds(s, e),
                            noisy_joint_2d_window=joint_2d_t[:, s:e, :, :],
                            img_feat_window=img_feat_win,
                            noisy_trans_window=trans_t[:, s:e, :],
                            noisy_ori_mat_window=ori_mat_t[:, s:e, :],
                            bbox_mask_window=(
                                t_bbox_mask[:, s:e, ...]
                                if t_bbox_mask is not None and t_bbox_mask.dim() == 5
                                else t_bbox_mask
                            ),
                            cond_confidence_window=conf_win,
                        )
                        if temp_conf is not None:
                            # Accumulate into a FULL-seq buffer (overlap-weighted),
                            # NOT `conf_final = temp_conf` — the latter leaked the last
                            # window's (possibly short) conf into the next step's
                            # conf_final[:, s:e] slice => length mismatch when the
                            # trailing window < window_size (e.g. seq 128, win 64).
                            if conf_accum is None:
                                conf_accum = torch.zeros(
                                    (batch, seq_len) + tuple(temp_conf.shape[2:]),
                                    device=device, dtype=temp_conf.dtype,
                                )
                            conf_accum[:, s:e, ...] += temp_conf * ow_slice[..., None]
                        x_0_packed_pred[:, s:e, :] += temp_x0 * ow_slice
                        if temp_j2d is not None:
                            j2d_win = _j2d_state_x0(temp_j2d, temp_conf)
                            j2d_accum[:, s:e, :, :] += j2d_win * ow_slice[..., None]
                        if temp_trans is not None:
                            trans_accum[:, s:e, :] += temp_trans * ow_slice
                            ori_mat_accum[:, s:e, :] += temp_ori_mat * ow_slice

                    x_0_packed_pred /= ow
                    j2d_accum /= ow[..., None]
                    trans_accum /= ow
                    ori_mat_accum /= ow
                    if conf_accum is not None:
                        conf_final = conf_accum / ow[..., None]  # full-seq conf for next step / output
                    x_0_packed_pred = _repack_x0(x_0_packed_pred)

                if start_time is None:
                    start_time = time.time()

                x_t_packed = _flow_step(x_0_packed_pred, x_t_packed, t_curr, t_next)
                # In Stage 2 (use_optimal_2d), don't update 2D - it's already finalized from Stage 1
                if not use_optimal_2d:
                    joint_2d_t = _flow_step(j2d_accum, joint_2d_t, t_curr, t_next)
                trans_t = _flow_step(trans_accum, trans_t, t_curr, t_next)
                ori_mat_t = _flow_step(ori_mat_accum, ori_mat_t, t_curr, t_next)

                x_t_list.append(_traj_from(x_t_packed))
        else:
            for i in tqdm(range(len(ts_frac) - 1)):
                t_curr, t_next = ts_frac[i], ts_frac[i + 1]
                t_int = _t_index(t_curr)  # floor + flow/diffusion orientation (matches training)

                with torch.inference_mode():
                    # Pass confidence from Stage 1 to weight 2D features during 2D-3D alignment
                    x_0_packed_pred, pose_2d_pred, conf_pred, pred_trans, pred_ori_mat = _predict_x0_window(
                        x_t_packed, t_int, rel_palm_pose, _make_conds(0, seq_len),
                        noisy_joint_2d_window=joint_2d_t,
                        img_feat_window=cond_feat,
                        noisy_trans_window=trans_t,
                        noisy_ori_mat_window=ori_mat_t,
                        bbox_mask_window=t_bbox_mask,
                        cond_confidence_window=conf_final,
                    )
                x_0_packed_pred = _repack_x0(x_0_packed_pred)

                if start_time is None:
                    start_time = time.time()

                x_t_packed = _flow_step(x_0_packed_pred, x_t_packed, t_curr, t_next)
                # In Stage 2 (use_optimal_2d), don't update 2D - it's already finalized from Stage 1
                if pose_2d_pred is not None and not use_optimal_2d:
                    joint_2d_t = _flow_step(_j2d_state_x0(pose_2d_pred, conf_pred), joint_2d_t, t_curr, t_next)
                if pred_trans is not None:
                    trans_t = _flow_step(pred_trans, trans_t, t_curr, t_next)
                    ori_mat_t = _flow_step(pred_ori_mat, ori_mat_t, t_curr, t_next)

                x_t_list.append(_traj_from(x_t_packed))

        if not use_optimal_2d:
            # conf_in_state: joint_2d_t is 3-ch [x,y,conf]; the 2D output is the coords.
            pose_2d_final = joint_2d_t[..., :2] if conf_in_state else joint_2d_t
            conf_final = conf_pred  # predicted visibility/confidence
        # else: pose_2d_final and conf_final were set in Stage 1.

    # ---- Diffusion (DDIM) ----
    else:
        pose_2d_final = None
        conf_final = None

        if seq_len > infer_window:
            # Window at the TRAINING clip length (infer_window = subseq_len) so a
            # model trained on N-frame clips never sees >N frames at once (e.g.
            # sub_len=64 -> 64-frame windows). Default 128 keeps the standard
            # models' behaviour (window 64 for >128) unchanged.
            window_size, overlap_size = min(64, infer_window), min(32, infer_window // 2)
            canonical_ow = _make_overlap_weights()
            ts = quadratic_ts()

            for i in tqdm(range(len(ts) - 1)):
                t, t_next = ts[i], ts[i + 1]

                with torch.inference_mode():
                    x_0_packed_pred = torch.zeros_like(x_t_packed)
                    p2d_accum = None
                    ow = torch.zeros((batch, seq_len, 1), device=x_t_packed.device)

                    for s in range(0, seq_len, window_size - overlap_size):
                        e = min(s + window_size, seq_len)
                        ow_slice = canonical_ow[None, :e - s, None]
                        ow[:, s:e, :] += ow_slice

                        img_feat_win = cond_feat[:, s:e] if cond_feat is not None else None
                        temp_x0, temp_p2d, temp_conf, _, _ = _predict_x0_window(
                            x_t_packed[:, s:e, :], t,
                            rel_palm_pose[:, s:e, :], _make_conds(s, e),
                            img_feat_window=img_feat_win,
                            bbox_mask_window=(
                                t_bbox_mask[:, s:e, ...]
                                if t_bbox_mask is not None and t_bbox_mask.dim() == 5
                                else t_bbox_mask
                            ),
                        )
                        if temp_conf is not None:
                            conf_final = temp_conf  # last window's confidence
                        x_0_packed_pred[:, s:e, :] += temp_x0 * ow_slice
                        if temp_p2d is not None:
                            if p2d_accum is None:
                                p2d_accum = torch.zeros((batch, seq_len, temp_p2d.shape[2], 2), device=device)
                            p2d_accum[:, s:e, :, :] += temp_p2d * ow_slice[..., None]

                    x_0_packed_pred /= ow
                    if p2d_accum is not None:
                        p2d_accum /= ow[..., None]
                        pose_2d_final = p2d_accum
                    x_0_packed_pred = _repack_x0(x_0_packed_pred)

                sigma_t = torch.cat([
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    ) * 0.8,
                ])
                if start_time is None:
                    start_time = time.time()

                x_t_packed = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
                )
                x_t_list.append(_traj_from(x_t_packed))
        else:
            ts = quadratic_ts()
            for i in tqdm(range(len(ts) - 1)):
                t, t_next = ts[i], ts[i + 1]

                with torch.inference_mode():
                    conds = _make_conds(0, seq_len)
                    x_0_packed_pred, pose_2d_step, conf_step, _, _ = denoiser_network.forward(
                        x_t_packed,
                        torch.tensor([t], device=device).expand((batch,)),
                        rel_palm_pose=rel_palm_pose,
                        project_output_rotmats=False,
                        conds=conds,
                        img_feat=cond_feat,
                        cond_dropout_keep_mask=torch.ones((batch,), device=device),
                        mask=None,
                        bbox_mask=t_bbox_mask,
                    )
                    if pose_2d_step is not None:
                        pose_2d_final = pose_2d_step
                        conf_final = conf_step

                x_0_packed_pred = _repack_x0(x_0_packed_pred)

                sigma_t = torch.cat([
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    ) * 0.8,
                ])
                if start_time is None:
                    start_time = time.time()

                x_t_packed = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
                )
                x_t_list.append(_traj_from(x_t_packed))

    traj = x_t_list[-1]

    if (getattr(denoiser_network.config, "pose2d_conf_onehot", False)
            and conf_final is not None and conf_final.shape[-1] == 2):
        conf_final = torch.softmax(conf_final, dim=-1)[..., 0:1]
    return traj, pose_2d_final, conf_final
