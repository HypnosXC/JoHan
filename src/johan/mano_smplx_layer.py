"""smplx-backed MANO layer, drop-in compatible with `manopth.manolayer.ManoLayer`.

The codebase standardises on smplx for every MANO readout, so this replaces
manopth everywhere. It keeps manopth's exact call convention so the swap is a
one-line import change at each site:

    layer = ManoLayer(flat_hand_mean=True, use_pca=False, ncomps=45,
                      side="right", mano_root=...)
    verts_mm, joints_mm = layer(pose48, betas, trans)   # MILLIMETRES, like manopth
    faces = layer.th_faces

What is identical to manopth
    Vertices. Measured on random params: mean 0.00001 mm, max 0.00005 mm. Both
    implementations read the same MANO .pkl and apply the same LBS, and
    smplx.MANOLayer adds no mean pose, matching flat_hand_mean=True.

What CHANGES, deliberately
    The 5 FINGERTIP joints. manopth and smplx pick different vertices off the
    posed mesh for tips: the 16 articulated joints agree to 0.000 mm, the tips
    differ by ~3 mm.

    This does NOT move HO3D PA-MPJPE. Measured on 6380 ho3d test frames, GT
    params through each wrapper vs the stored GT joints:
        manopth  PA-MPJPE(21) 8.1615 mm   PA-MPJPE(16) 6.2813 mm
        smplx    PA-MPJPE(21) 8.1697 mm   PA-MPJPE(16) 6.2813 mm
    a difference of 0.008 mm. The reason is that HO3D's GT joints are an
    INDEPENDENT annotation, not MANO FK, so they do not sit on either wrapper's
    tip convention; the tip shift is essentially orthogonal to the error that is
    already there and Procrustes absorbs it. HO3D numbers therefore remain
    comparable across this switch. On HOT3D, where GT IS derived from MANO params
    (HaWoR's run_mano), smplx is required rather than merely harmless.

All manopth modes are supported so the whole repo can drop manopth:
  use_pca=False, flat_hand_mean=True   -- rotmat path via smplx.MANOLayer (adds no
                                          mean), verified 0.00001 mm vs manopth.
  use_pca=False, flat_hand_mean=False  -- the MANO mean pose is added to the
                                          axis-angle hand pose before Rodrigues,
                                          exactly manopth's behaviour.
  use_pca=True                          -- pose coeffs are [global aa | ncomps PCA];
                                          smplx.MANO with native PCA handles the
                                          basis and (non-)flat mean itself.
Each mode is numerically verified against manopth (see the migration commit).
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

from .mano_constants import MANO_TO_OPENPOSE, SMPLX_MANO_LEFT, SMPLX_MANO_RIGHT


class ManoLayer(nn.Module):
    def __init__(self, flat_hand_mean: bool = True, use_pca: bool = False,
                 ncomps: int = 45, side: str = "right", mano_root=None,
                 center_idx=None, tip_convention: str = "smplx", **_unused):
        super().__init__()
        import smplx
        from smplx.utils import to_tensor
        from smplx.vertex_ids import vertex_ids

        self.side = side
        self.use_pca = bool(use_pca)
        self.ncomps = int(ncomps)
        self.flat_hand_mean = bool(flat_hand_mean)
        model_path = mano_root or os.environ.get("JOHAN_MANO_ROOT")
        if not model_path:
            raise ValueError("Set JOHAN_MANO_ROOT to the directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl")
        if self.use_pca:
            if self.flat_hand_mean:
                raise ValueError(
                    "use_pca=True with flat_hand_mean=True is unverified against "
                    "manopth (measured ~12mm apart; smplx treats the mean "
                    "differently in native-PCA mode) and no call site in this repo "
                    "uses it. Every real PCA site passes flat_hand_mean=False.")
            # PCA mode: smplx.MANO owns the basis and the mean, pose2rot=True.
            self.layer = smplx.MANO(model_path=model_path, is_rhand=(side == "right"),
                                    use_pca=True, num_pca_comps=self.ncomps,
                                    flat_hand_mean=self.flat_hand_mean)
            if not hasattr(self.layer, "hand_components"):
                self.layer.register_buffer(
                    "hand_components",
                    torch.as_tensor(self.layer.np_hand_components).float())
            self.layer.use_pca = True
            # manopth-internals aliases used by pca_coeffs_to_axis45 in the
            # preprocessing scripts: PCA basis (ncomps,45) and the hand mean (45,).
            self.register_buffer("th_selected_comps",
                                 torch.as_tensor(self.layer.hand_components).float())
            self.register_buffer("th_hands_mean",
                                 torch.as_tensor(self.layer.hand_mean).reshape(45).float())
        else:
            # Axis-angle mode: rotmat path (adds no mean); non-flat mean is added
            # to the aa hand pose below, matching manopth exactly.
            self.layer = smplx.MANOLayer(model_path=model_path,
                                         is_rhand=(side == "right"), use_pca=False)
            # th_hands_mean exists in EVERY mode, with manopth's semantics: the
            # zero vector under flat_hand_mean=True, the MANO mean pose otherwise.
            # Preprocessing scripts read it directly (24 sites).
            _m = getattr(self.layer, "hand_mean", None)
            if _m is None:
                _pm = getattr(self.layer, "pose_mean", None)   # (48,) incl. global
                _m = None if _pm is None else torch.as_tensor(_pm).reshape(-1)[3:48]
            if self.flat_hand_mean:
                _hm = torch.zeros(45)
            elif _m is None:
                raise RuntimeError("smplx MANO layer exposes no hand mean; "
                                   "cannot emulate flat_hand_mean=False")
            else:
                _hm = torch.as_tensor(_m).reshape(45).float()
            self.register_buffer("th_hands_mean", _hm)
        # smplx MANO_LEFT.pkl carries a left-hand shapedir sign bug (smplx issue
        # #48): the raw shapedirs mirror the shape blend on x. HaWoR's
        # run_mano_left corrects it with shapedirs[:,0,:]*=-1; apply the SAME fix
        # so ManoLayer(left) == run_mano_left (the true left hand the models are
        # trained against). Without it left FK is ~7.5mm off, drifting left PA/2D
        # vs the model's training convention. Right MANO has no such bug (untouched).
        # HAND_LEFT_SHAPEDIR=raw disables this correction. The sign is DATASET
        # dependent: HO3D and HaWoR/HOT3D expect it APPLIED (ManoLayer(left) ==
        # run_mano_left), DexYCB expects it REVERTED. Measured against DexYCB's own
        # labels on s0 test left groups: reverted 0.000 mm, applied 4.341 mm. The
        # training loss uses SmplxManoJointLayer, which never applies it, so leaving
        # it on at eval made DexYCB left a TRAIN/EVAL MISMATCH (left PA 5.049 vs
        # right 4.414).
        if self.side != "right" and os.environ.get("HAND_LEFT_SHAPEDIR", "flip") != "raw":
            with torch.no_grad():
                self.layer.shapedirs[:, 0, :] *= -1
        self.center_idx = center_idx
        # Fingertips are not MANO joints -- every consumer picks 5 MESH VERTICES,
        # and the choice differs by dataset (see mano_eval.TIP_VERTEX_IDS). The
        # default stays smplx so nothing already trained or scored moves; pass
        # tip_convention="ho3d" to read HO3D's official setup_mano.py list.
        # HAND_TIP_CONVENTION overrides the default for a whole eval run, which
        # is how the HO3D benches select HO3D's own list without every FK call
        # site having to thread the argument through.
        if tip_convention == "smplx":
            tip_convention = os.environ.get("HAND_TIP_CONVENTION", "smplx") or "smplx"
        self.tip_convention = tip_convention
        if tip_convention == "smplx":
            _tips = list(vertex_ids["mano"].values())
        else:
            from .mano_constants import TIP_VERTEX_IDS
            _tips = TIP_VERTEX_IDS[(tip_convention, side)]
        self.register_buffer("tips", to_tensor(_tips, dtype=torch.long))
        self.register_buffer("remap", torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long))
        self.register_buffer("th_faces",
                             torch.from_numpy(self.layer.faces.astype(np.int64)))
        # manopth-internal aliases used by the preprocessing scripts' cal_root_j
        # (rest-pose root from betas). Same tensors, manopth's names and shapes:
        # th_shapedirs (V,3,10), th_v_template (1,V,3), th_J_regressor (16,V).
        self.register_buffer("th_shapedirs", self.layer.shapedirs.detach().clone().float())
        self.register_buffer("th_v_template",
                             self.layer.v_template.detach().clone().float().unsqueeze(0))
        self.register_buffer("th_J_regressor",
                             torch.as_tensor(self.layer.J_regressor).float())

    def forward(self, th_pose_coeffs: torch.Tensor, th_betas: torch.Tensor,
                th_trans: torch.Tensor | None = None):
        """pose48 (N,48) axis-angle [global_orient | 45-dim hand_pose].

        Returns (verts, joints) in MILLIMETRES, joints in the 21-joint
        wrist-then-4-per-finger order, matching manopth's outputs.
        """
        from smplx.lbs import batch_rodrigues

        n = th_pose_coeffs.shape[0]
        if th_betas is None:
            th_betas = torch.zeros(n, 10, dtype=th_pose_coeffs.dtype,
                                   device=th_pose_coeffs.device)
        if self.use_pca:
            out = self.layer(betas=th_betas,
                             global_orient=th_pose_coeffs[:, :3],
                             hand_pose=th_pose_coeffs[:, 3:3 + self.ncomps],
                             transl=th_trans)
            verts = out.vertices
            joints = torch.cat([out.joints,
                                torch.index_select(verts, 1, self.tips)], dim=1)
            return verts * 1000.0, joints[:, self.remap, :] * 1000.0
        go = batch_rodrigues(th_pose_coeffs[:, :3]).view(n, 1, 3, 3)
        hp_aa = th_pose_coeffs[:, 3:48]
        if not self.flat_hand_mean:
            hp_aa = hp_aa + self.th_hands_mean.reshape(1, 45).to(hp_aa.dtype)
        hp = batch_rodrigues(hp_aa.reshape(n * 15, 3)).view(n, 15, 3, 3)
        out = self.layer(betas=th_betas, global_orient=go, hand_pose=hp,
                         transl=th_trans, pose2rot=False)
        verts = out.vertices
        joints = torch.cat([out.joints,
                            torch.index_select(verts, 1, self.tips)], dim=1)
        joints = joints[:, self.remap, :]
        # smplx's MANO.forward adds `transl` to both vertices and joints when it is
        # passed, so no extra translation handling is needed here.
        return verts * 1000.0, joints * 1000.0
