from __future__ import annotations
import os
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float
from .mano_smplx_layer import ManoLayer
HAND_MANO_ROOT = None
def denormalize_2d_keypoints_minmax(
    joints_2d_norm: torch.Tensor,  # (..., J, 2)
    norm_params: torch.Tensor,     # (..., 4) [min_x, min_y, max_x, max_y]
) -> torch.Tensor:
    """Recover pixel coordinates from [-1, 1] normalized joints.

    norm_params can be:
      (4,)       — single set of params, broadcasts to any joint shape
      (B, 4)     — per-batch params for (B, T, J, 2) joints
      (B, 1, 4)  — same, explicit time broadcast
    """
    # Build min/range tensors with shape (..., 1, 2) for broadcasting with (..., J, 2).
    min_xy = norm_params[..., :2]   # (..., 2)
    max_xy = norm_params[..., 2:]   # (..., 2)
    range_xy = max_xy - min_xy      # (..., 2)

    # Add dimensions to broadcast over J (and T if needed).
    # Target: joints_2d_norm has shape (..., J, 2), so min_xy needs (..., 1, 2).
    while min_xy.dim() < joints_2d_norm.dim():
        min_xy = min_xy.unsqueeze(-2)
        range_xy = range_xy.unsqueeze(-2)

    joints_2d = (joints_2d_norm + 1.0) / 2.0 * range_xy + min_xy
    return joints_2d

def solve_global_pose_pnp(
    hand_pose_45: torch.Tensor,     # (N, 45) predicted local hand pose
    mano_betas: torch.Tensor,       # (N, 10)
    joints_2d_px: torch.Tensor,     # (N, 21, 2) predicted 2D in PIXELS
    intrinsics: torch.Tensor,       # (N, 4) fx, fy, cx, cy
    mano_side: str,
    ransac: bool = True,
    z_prior: np.ndarray | None = None,   # (N,) metric root depth, metres
    z_weight: float = 50.0,
    uv_conf: torch.Tensor | None = None,  # (N, 21) predicted P(visible), [0,1]
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover per-frame global orientation + translation by PnP.

    Replaces the GT global pose that inference otherwise falls back to (the model
    is trained with predict_global_pose: false, so it predicts only LOCAL pose).

    Convention, verified numerically to 1e-7 m on HO3D GT:
        J_full = R @ (J0 - root) + root + trans
    so PnP is solved with object points (J0 - root); its tvec is (root + trans),
    hence trans = tvec - root. Feeding un-centred J0 instead silently biases
    translation by (I - R) @ root (up to 0.18 m on HO3D).
    """
    if sequence_length is not None:
        frame_count = hand_pose_45.shape[0]
        if sequence_length <= 0 or frame_count % sequence_length:
            raise ValueError("PnP sequence_length must be positive and divide the frame count")
        if frame_count > sequence_length:
            recovered = []
            for start in range(0, frame_count, sequence_length):
                selection = slice(start, start + sequence_length)
                recovered.append(solve_global_pose_pnp(
                    hand_pose_45[selection], mano_betas[selection],
                    joints_2d_px[selection], intrinsics[selection], mano_side,
                    ransac=ransac,
                    z_prior=None if z_prior is None else z_prior[selection],
                    z_weight=z_weight,
                    uv_conf=None if uv_conf is None else uv_conf[selection]))
            return (torch.cat([result[0] for result in recovered]),
                    torch.cat([result[1] for result in recovered]))

    import cv2 as _cv2

    n = hand_pose_45.shape[0]
    dev = hand_pose_45.device
    pose48 = torch.cat([torch.zeros(n, 3, dtype=hand_pose_45.dtype), hand_pose_45.cpu()], dim=1)
    ml = ManoLayer(
        ncomps=45, side=mano_side, mano_root=HAND_MANO_ROOT,
        use_pca=False, flat_hand_mean=True,
    )
    with torch.no_grad():
        _, j0 = ml(pose48, mano_betas.cpu().float(), torch.zeros(n, 3))
    j0 = (j0 / 1000.0).numpy().astype(np.float64)          # (N,21,3) canonical, metres
    root = j0[:, 0, :]                                      # (N,3)
    obj = j0 - root[:, None, :]                             # root-centred object points

    uv = joints_2d_px.detach().cpu().numpy().astype(np.float64)
    K4 = intrinsics.detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
    # RANSAC inlier threshold, in FULL-RES pixels. The historical 8.0 is tighter
    # than our own 2D error at 1408 (~11px), which is why it rejected everything on
    # up to 26% of HOT3D frames. Left at 8.0 by default so existing HO3D MPJPE_pnp
    # numbers do not move silently; HAND_PNP_RANSAC_PX lets it be swept offline.
    _RANSAC_PX = float(os.environ.get("HAND_PNP_RANSAC_PX", "8.0"))
    # HAND_PNP_HUBER > 0 enables a robust per-frame refinement: Huber loss with
    # f_scale in FULL-RES px. Unlike RANSAC's hard gate (a joint is in or out),
    # Huber down-weights large-residual joints smoothly, so occluded/out-of-frame
    # joints stop poisoning the least-squares solve without being discarded.
    _HUBER_PX = float(os.environ.get("HAND_PNP_HUBER", "0"))
    # Confidence-aware PnP, both gated off by default so no caller changes
    # behaviour. HAND_PNP_CONF_WEIGHT=1 multiplies each joint's reprojection
    # residual by sqrt(conf) in the refinement (weighted least squares);
    # HAND_PNP_CONF_DROP=tau removes joints with conf < tau from the solve
    # entirely (kept only if >=6 joints survive -- PnP needs that many).
    _CW = os.environ.get("HAND_PNP_CONF_WEIGHT", "0") == "1"
    _CDROP = float(os.environ.get("HAND_PNP_CONF_DROP", "0"))
    cw = None
    if uv_conf is not None and (_CW or _CDROP > 0):
        cw = np.clip(np.nan_to_num(
            uv_conf.detach().cpu().numpy().astype(np.float64).reshape(n, 21),
            nan=1.0), 0.0, 1.0)   # NaN conf (no prediction) = do not down-weight
    ori = np.zeros((n, 3), dtype=np.float64)
    trans = np.zeros((n, 3), dtype=np.float64)
    _failed: list[int] = []          # frames where BOTH RANSAC and plain PnP failed
    _n_fallback = 0                  # frames rescued by the non-RANSAC fallback
    _n_degenerate = 0                # "successful" solves rejected by the sanity bound
    for i in range(n):
        fx, fy, cx, cy = K4[i]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        good = np.isfinite(uv[i]).all(axis=1)
        if _CDROP > 0 and cw is not None:
            _g2 = good & (cw[i] >= _CDROP)
            if _g2.sum() >= 6:
                good = _g2
        if good.sum() < 6:
            _failed.append(i)
            continue
        try:
            if ransac:
                ok, rvec, tvec, _ = _cv2.solvePnPRansac(
                    obj[i][good], uv[i][good], K, None,
                    flags=_cv2.SOLVEPNP_ITERATIVE, reprojectionError=_RANSAC_PX,
                    iterationsCount=200, confidence=0.99,
                )
            else:
                ok, rvec, tvec = _cv2.solvePnP(
                    obj[i][good], uv[i][good], K, None, flags=_cv2.SOLVEPNP_ITERATIVE
                )
        except _cv2.error:
            ok = False
        if not ok:
            # RANSAC found no consensus set. Its reprojectionError=8.0 is in FULL-RES
            # pixels while our typical 2D error is ~11px at 1408 -- above the inlier
            # threshold -- so this fires often (measured 7.6-26.2% of frames per
            # HOT3D video). The old code just `continue`d, leaving ori/trans at their
            # pre-initialised ZEROS, which are indistinguishable downstream from a
            # real solve at the origin: a hand placed at depth 0 instead of ~0.3m
            # contributes a 300mm error and inflates depth variance exactly where we
            # were blaming solver noise. Fall back to plain PnP, which uses all
            # points and always returns something; only if THAT fails is the frame
            # recorded as failed (and interpolated below), never silently zeroed.
            try:
                ok, rvec, tvec = _cv2.solvePnP(
                    obj[i][good], uv[i][good], K, None, flags=_cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    _n_fallback += 1
            except _cv2.error:
                ok = False
        if not ok:
            _failed.append(i)
            continue
        _has_zp = z_prior is not None and np.isfinite(z_prior[i]) and z_prior[i] > 1e-3
        _has_cw = _CW and cw is not None
        if _has_zp or _HUBER_PX > 0 or _has_cw:
            # depth+PnP: keep the reprojection term (which conditions rotation and
            # lateral translation well) and add a soft prior only on t_z, the one
            # direction 2D cannot pin down. A single median depth is used rather
            # than 21 per-joint depths because monocular depth error is spatially
            # CORRELATED across the hand -- feeding it per-joint tilts the fitted
            # frame and wrecks rotation (measured: 0 -> 37mm/50deg on clean 2D).
            from scipy.optimize import least_squares as _lsq
            _fx, _fy, _cx, _cy = K4[i]
            _obj, _uv = obj[i][good], uv[i][good]
            # sqrt because least_squares squares the residual: the squared error
            # of joint j is then scaled by conf_j. Floor 0.05 keeps zero-conf
            # joints from vanishing entirely (they still weakly anchor rotation).
            _w = np.sqrt(np.clip(cw[i][good], 0.05, 1.0)) if _has_cw else None

            def _res(prm):
                _R, _ = _cv2.Rodrigues(prm[:3])
                P = (_R @ _obj.T).T + prm[3:]
                z = np.clip(P[:, 2], 1e-6, None)
                rx = (P[:, 0] / z * _fx + _cx) - _uv[:, 0]
                ry = (P[:, 1] / z * _fy + _cy) - _uv[:, 1]
                if _w is not None:
                    rx, ry = rx * _w, ry * _w
                r = np.concatenate([rx, ry])
                if _has_zp:
                    r = np.concatenate([r, [z_weight * (prm[5] - float(z_prior[i]))]])
                return r

            try:
                _x0 = np.concatenate([rvec.reshape(3), tvec.reshape(3)])
                if _HUBER_PX > 0:
                    # scipy's "lm" supports only linear loss, so the robust path
                    # must use "trf". The z-prior residual (|z_weight*dz| ~ 1 for a
                    # typical 20mm disagreement) sits well inside f_scale's
                    # quadratic region, so the prior keeps its usual strength.
                    _o = _lsq(_res, _x0, method="trf", loss="huber",
                              f_scale=_HUBER_PX, max_nfev=300)
                else:
                    _o = _lsq(_res, _x0, method="lm", max_nfev=300)
                rvec, tvec = _o.x[:3].reshape(3, 1), _o.x[3:].reshape(3, 1)
            except Exception:
                pass
        # DEGENERATE-SOLVE GUARD. solvePnP on near-degenerate 2D (hand almost out
        # of frame, points nearly collinear/clustered) can return an astronomical
        # translation from a solve that still reports ok=True, and nothing later
        # questions it -- observed as MPJPE_pnp 4.8e7 mm on hot3dfull (ft EMA
        # @1225000): a few km-scale "successes" poisoned the mean while adjacent
        # checkpoints scored 23-31 mm. An egocentric hand lives within a few
        # metres of the camera, so bound the WRIST (= tvec) and route violations
        # through the same interpolation as outright solver failures.
        _tv = tvec.reshape(3)
        if not (np.isfinite(_tv).all() and 0.02 < _tv[2] < 10.0
                and float(np.linalg.norm(_tv)) < 10.0):
            _n_degenerate += 1
            _failed.append(i)
            continue
        ori[i] = rvec.reshape(3)
        trans[i] = tvec.reshape(3) - root[i]                # -> MANO trans convention

    # Frames where every solver failed must NOT stay at (0,0,0): fill them from the
    # nearest successful neighbours, which is what a smooth trajectory implies, and
    # is bounded error rather than a 300mm one. If nothing succeeded at all we leave
    # the zeros but say so loudly instead of returning a plausible-looking array.
    if _failed:
        _bad = np.zeros(n, dtype=bool)
        _bad[np.asarray(_failed, dtype=int)] = True
        _gidx = np.nonzero(~_bad)[0]
        if _gidx.size:
            _t = np.arange(n)
            for _c in range(3):
                trans[_bad, _c] = np.interp(_t[_bad], _gidx, trans[_gidx, _c])
            _near = _gidx[np.clip(np.searchsorted(_gidx, _t[_bad]), 0, _gidx.size - 1)]
            ori[_bad] = ori[_near]                    # nearest, not interpolated:
                                                     # axis-angle does not average
        print(f"[pnp] {len(_failed)}/{n} frames unsolved -> "
              f"{'interpolated' if _gidx.size else 'LEFT AT ZERO (no frame solved!)'}"
              f"; {_n_fallback} rescued by non-RANSAC fallback"
              f"; {_n_degenerate} rejected as degenerate (out-of-bounds tvec)",
              flush=True)
    elif _n_fallback:
        print(f"[pnp] {_n_fallback}/{n} frames rescued by the non-RANSAC fallback",
              flush=True)

    # ---- SEQUENCE-LEVEL REFINEMENT (HAND_PNP_SMOOTH > 0) --------------------
    # Per-frame PnP solves every frame independently, so its error -- dominated by
    # depth, the direction 2D barely constrains -- is uncorrelated in time and
    # shows up as jitter. A hand trajectory is smooth, so penalising ACCELERATION
    # lets steady motion through while suppressing that per-frame noise, and it
    # averages the depth error over the clip instead of leaving it per frame.
    # Second differences (not first) on purpose: a velocity penalty would fight
    # genuine fast motion, whereas an acceleration penalty only fights jerk.
    # Optimises the whole clip jointly: reprojection + w * ||d2 trans|| +
    # w_rot * ||d2 rotvec||, initialised from the per-frame solution above.
    _sw = float(os.environ.get("HAND_PNP_SMOOTH", "0"))
    if _sw > 0.0 and n >= 5:
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        _obj_t = torch.from_numpy(obj).to(_dev).float()                  # (N,21,3)
        # Non-finite uv entries MUST be scrubbed, not just masked: `_valid` below is
        # built from isfinite(uv) precisely because uv contains NaN, and masking by
        # MULTIPLYING with a 0/1 mask does not remove them -- NaN * 0 == NaN, in
        # both forward and backward. The loss went NaN on iteration 1 and the
        # poisoned pose surfaced minutes later as "SVD did not converge" inside
        # Procrustes, which is what killed the first sweep (219679-681).
        _uv_t = torch.nan_to_num(torch.from_numpy(uv).to(_dev).float(),
                                 nan=0.0, posinf=0.0, neginf=0.0)        # (N,21,2)
        _K_t = torch.from_numpy(K4).to(_dev).float()                     # (N,4)
        _valid = torch.from_numpy(np.isfinite(uv).all(axis=2)).to(_dev)  # (N,21)
        _r = torch.from_numpy(ori).to(_dev).float().requires_grad_(True)
        # optimise the PnP tvec (root + trans), convert back at the end
        _t = torch.from_numpy(trans + root).to(_dev).float().requires_grad_(True)
        _opt = torch.optim.Adam([_r, _t], lr=float(os.environ.get("HAND_PNP_SMOOTH_LR", "0.003")))
        _wrot = float(os.environ.get("HAND_PNP_SMOOTH_ROT", str(_sw)))

        def _rodrigues(v):
            # v.norm() has a NaN GRADIENT at v==0 (d|v|/dv is undefined there), and
            # clamping the forward value does not fix the backward. Take the sqrt of
            # a clamped square instead, which is smooth everywhere.
            th = (v * v).sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
            k = v / th
            K_ = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
            K_[:, 0, 1], K_[:, 0, 2] = -k[:, 2], k[:, 1]
            K_[:, 1, 0], K_[:, 1, 2] = k[:, 2], -k[:, 0]
            K_[:, 2, 0], K_[:, 2, 1] = -k[:, 1], k[:, 0]
            I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K_)
            th = th[..., None]
            return I + torch.sin(th) * K_ + (1 - torch.cos(th)) * (K_ @ K_)

        def _so3_log(R):
            """Rotation matrices -> axis-angle, numerically safe near theta=0/pi."""
            tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            th = torch.acos(cos)
            w = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                             R[:, 0, 2] - R[:, 2, 0],
                             R[:, 1, 0] - R[:, 0, 1]], dim=-1)
            # torch.where evaluates BOTH branches, and propagates NaN gradients
            # from the branch it does not take. At theta->0 the else-branch is
            # 0/0, so it must be made finite BEFORE the where -- clamping theta
            # inside the division does that; the where then only selects.
            small = th < 1e-4
            th_safe = th.clamp(min=1e-4)
            scale = torch.where(small,
                                torch.full_like(th, 0.5),
                                th_safe / (2.0 * torch.sin(th_safe)))
            return w * scale[:, None]

        for _it in range(int(os.environ.get("HAND_PNP_SMOOTH_ITERS", "300"))):
            _opt.zero_grad()
            _Rm = _rodrigues(_r)
            P = torch.einsum("nij,nkj->nki", _Rm, _obj_t) + _t[:, None, :]
            z = P[..., 2].clamp(min=1e-6)
            u = P[..., 0] / z * _K_t[:, 0:1] + _K_t[:, 2:3]
            v = P[..., 1] / z * _K_t[:, 1:2] + _K_t[:, 3:4]
            rep = (((u - _uv_t[..., 0]) ** 2 + (v - _uv_t[..., 1]) ** 2) * _valid).sum() / _valid.sum().clamp(min=1)
            d2t = _t[2:] - 2 * _t[1:-1] + _t[:-2]
            # ROTATION SMOOTHNESS ON THE MANIFOLD. Differencing axis-angle vectors
            # directly is wrong: SO(3) is not a vector space, the double cover
            # makes r and r*(1-2pi/|r|) the same rotation but far apart as
            # vectors, there is a discontinuity at |r|=pi, and the penalty would
            # depend on the arbitrary world frame. Use the RELATIVE rotation
            # instead -- omega_t = log(R_t^T R_{t+1}) is the body-frame angular
            # velocity, and its first difference is angular acceleration.
            _om = _so3_log(_Rm[:-1].transpose(1, 2) @ _Rm[1:])    # (N-1,3)
            d2r = _om[1:] - _om[:-1]                              # (N-2,3)
            loss = rep + _sw * (d2t ** 2).sum(-1).mean() + _wrot * (d2r ** 2).sum(-1).mean()
            # A single non-finite step poisons the whole trajectory, and the failure
            # surfaces far away as "SVD did not converge" inside Procrustes -- which
            # is what killed the first sweep (219679-681). Stop instead, keeping the
            # last good iterate, and clip so one bad frame cannot blow up the rest.
            if not torch.isfinite(loss):
                print(f"[pnp-smooth] non-finite loss at iter {_it}; keeping last good iterate",
                      flush=True)
                break
            loss.backward()
            if not (torch.isfinite(_r.grad).all() and torch.isfinite(_t.grad).all()):
                print(f"[pnp-smooth] non-finite grad at iter {_it}; stopping", flush=True)
                break
            torch.nn.utils.clip_grad_norm_([_r, _t], 10.0)
            _opt.step()
            with torch.no_grad():
                _best = loss.detach()
                if not torch.isfinite(_best):
                    break
                _r_best, _t_best = _r.detach().clone(), _t.detach().clone()
        _r_out = _r_best if "_r_best" in dir() else _r.detach()
        _t_out = _t_best if "_t_best" in dir() else _t.detach()
        ori = _r_out.cpu().numpy().astype(np.float64)
        trans = _t_out.cpu().numpy().astype(np.float64) - root
        if not (np.isfinite(ori).all() and np.isfinite(trans).all()):
            raise FloatingPointError("PnP smoothing produced non-finite pose")
    return (
        torch.from_numpy(ori).to(dev).float(),
        torch.from_numpy(trans).to(dev).float(),
    )


# ============================================================================
# Args
# ============================================================================


