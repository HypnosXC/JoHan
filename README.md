# JoHan

**Video-Conditioned Generative Joint 2D–3D Hand Motion Recovery**

[Paper (arXiv)](https://arxiv.org/abs/2610.10512) · [Project page](https://hypnosxc.github.io/JoHan-project/)

Code and model weights will be released here.

## Inference preview

This branch contains the hand-only inference package. It does not include training,
datasets, body estimation, Aria processing, benchmark runners, or cluster automation.
Model weights are not included in Git. Use a compatible JoHan joint 2D–3D checkpoint.

### Installation

Use Python 3.12+ and a CUDA-capable PyTorch installation:

```bash
pip install -e .
```

Obtain these assets separately under their respective licenses:

- A local DINOv3 checkout and the matching pretrained backbone weights.
- MANO_LEFT.pkl and MANO_RIGHT.pkl from MANO, in one directory.
- The WiLoR YOLO hand detector checkpoint (classes 0=left and 1=right).
- JoHan model_config.yaml and run_config.yaml in the checkpoint's parent directory.
- model.safetensors and dino_finetuned.pt in the checkpoint directory. EMA mode
  requires both model_ema.safetensors and dino_finetuned_ema.pt; it never mixes modes.
  train_meta.json is retained when present to resolve checkpoint-specific coord_range.

Legacy dataclass YAML tags are read as mappings without importing the old package.
Optimizer state, training data and EMA optimizer bookkeeping are unnecessary.

### Video inference

```bash
johan --video input.mp4 \
  --checkpoint checkpoints/release/checkpoints_130000 \
  --dino-repo assets/dinov3 --dino-weights assets/dino_pretrained.pth \
  --mano-root assets/mano --detector assets/detector.pt \
  --intrinsics 1000 1000 640 360 --output outputs/predictions.npz
```

Supply calibrated fx, fy, cx, cy in original-video pixels. The example intrinsics
are placeholders. Image size, backbone architecture and box conditioning come
from the checkpoint configuration. `--weights ema` selects paired EMA weights.
`--max-frames 128` limits a smoke run; `--steps 10` controls flow sampling.

The detector selects the most confident hand of each side per frame. This initial
release targets a single wearer/person, not multi-person identity tracking.
Inference windows stay within contiguous detection runs and never bridge gaps.

The NPZ contains frame × hand arrays (`hand=0` left, `hand=1` right):

- `mano_pose`: 45 local axis-angle parameters, radians.
- `mano_betas`: 10 shape coefficients.
- `global_orient`: camera-space root axis-angle, radians.
- `translation`: MANO camera-space translation, metres, recovered with PnP.
- `joints_2d`: 21 joints in original-video pixels.
- `bboxes`, `valid`, `intrinsics`, `fps`.

Missing detections have `valid=False` and NaN outputs. A JSON sidecar records
checkpoint, weight mode and coordinate convention. Camera-space trajectories
include camera motion; this preview does not claim world-space recovery or run
SLAM, body estimation, or Metric3D depth estimation. PnP can be ambiguous under
heavy occlusion and does not guarantee metric depth accuracy.

The current video reader keeps decoded frames in memory; trim long videos before
running this preview. No end-to-end validation is claimed until recorded below.

## Attribution

Reusable tensor/transform/transformer utilities derive from the MIT-licensed
upstream project by Brent Yi. Its copyright and license are preserved in LICENSE.
DINOv3, MANO and the WiLoR detector are separate assets governed by their own
licenses; this repository does not redistribute them.
