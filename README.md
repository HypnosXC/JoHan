<div align="center">

# JoHan
### Video-Conditioned Generative Joint 2D–3D Hand Motion Recovery

Chen Xu · Yunqi Li · Binbin Huang · Brent Yi · Shenghua Gao · Yi Ma

**The University of Hong Kong · UC Berkeley**

[**Paper**](https://arxiv.org/abs/2610.10512) · [**Project & Results**](https://hypnosxc.github.io/JoHan-project/) · [**Pretrained Weights**](https://huggingface.co/HypnosXC/JoHan_pretrained)

</div>

JoHan recovers hand motion from monocular RGB video by jointly generating 2D joint trajectories and local 3D hand pose. The generated 2D–3D correspondences then recover the hand's orientation and translation relative to the camera.

This repository provides the **ViT-H/512 inference release**. A single `johan` command runs hand detection, video-conditioned generation and geometric pose recovery for left and right hands, and saves the results for downstream use.

## Release

- [x] Video-to-hand inference with the ViT-H/512 model.
- [x] Paired JoHan and fine-tuned DINOv3 RAW weights at step 145000.
- [x] Camera-space MANO parameters and image-space 2D trajectories.

The current release focuses on inference. Training, dataset preparation and benchmark runners are not included. Qualitative results and interactive examples are available on the [project page](https://hypnosxc.github.io/JoHan-project/).

## Installation

### 1. Clone and create an environment

```bash
git clone https://github.com/HypnosXC/JoHan.git
cd JoHan
conda create -n johan python=3.12 -y
conda activate johan
```

### 2. Install dependencies

Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support appropriate for your system, then install JoHan and the Hugging Face CLI:

```bash
pip install -e .
pip install huggingface_hub
```

JoHan requires Python 3.12 or newer and PyTorch 2.7 or newer (below 3.0). The default device is CUDA; `--device cpu` is available for a slow smoke run. See [VALIDATION.md](VALIDATION.md) for the checks performed on this release.

## Pretrained Models and Assets

### 1. Download JoHan and its fine-tuned backbone

```bash
hf download HypnosXC/JoHan_pretrained --local-dir assets/JoHan_pretrained
```

The download includes **both** `model.safetensors` and its matching `dino_finetuned.pt`, together with inference configurations. Use these files as a pair. This release provides RAW weights; use `--weights raw` (the default).

### 2. Prepare external assets

| Asset | Source | Expected location |
| --- | --- | --- |
| DINOv3 implementation | [Meta DINOv3](https://github.com/facebookresearch/dinov3) | `assets/dinov3/` |
| WiLoR hand detector (`detector.pt`) | [WiLoR](https://github.com/rolpotamias/WiLoR#installation) | `assets/detector.pt` |
| Left and right MANO models | [MANO](https://mano.is.tue.mpg.de/) (registration required) | `assets/mano/MANO_LEFT.pkl`, `MANO_RIGHT.pkl` |

Clone the backbone implementation:

```bash
git clone https://github.com/facebookresearch/dinov3.git assets/dinov3
```

Obtain the remaining assets from the linked projects under their respective licenses. The detector must use WiLoR's class convention: `0=left`, `1=right`.

The backbone architecture is constructed from the local DINOv3 checkout, then strictly loaded from the matching trained `checkpoint/dino_finetuned.pt`. No original pretrained DINOv3 weight file is needed. The coordinate scale is stored in `model_config.yaml`.

Expected layout:

```text
assets/
├── JoHan_pretrained/
│   ├── model_config.yaml
│   ├── run_config.yaml
│   └── checkpoint/
│       ├── model.safetensors
│       └── dino_finetuned.pt
├── dinov3/
├── detector.pt
└── mano/
    ├── MANO_LEFT.pkl
    └── MANO_RIGHT.pkl
```

## Run on Your Video

Provide an RGB video and its calibrated camera intrinsics:

```bash
johan --video /path/to/video.mp4 \
  --checkpoint assets/JoHan_pretrained/checkpoint \
  --dino-repo assets/dinov3 \
  --mano-root assets/mano \
  --detector assets/detector.pt \
  --intrinsics 1000 1000 640 360 \
  --output outputs/predictions.npz
```

Replace the example intrinsics with **`fx fy cx cy` in the original video's pixel coordinates**, separated by spaces. These values are required; the CLI does not estimate camera calibration.

For a short first run, append `--max-frames 128`. The same command accepts fixed-camera and moving-camera videos; in both cases, the output is in the **camera coordinate frame**. Camera motion is therefore present in the recovered trajectories. This release does not run SLAM or world-space alignment.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `cuda` | PyTorch device, such as `cuda:0` or `cpu` |
| `--steps` | `10` | Number of flow-sampling steps |
| `--max-frames` | `0` | Maximum frames to process; `0` reads the whole video |
| `--confidence` | `0.3` | Hand detection confidence threshold |
| `--weights` | `raw` | Weight mode; `ema` requires a separate matching EMA pair |
| `--output` | `outputs/predictions.npz` | Prediction file; metadata is saved alongside it as JSON |

Run `johan --help` for the complete argument list.

## Outputs

Each run writes an NPZ archive and a JSON metadata file. Hand arrays use the order **`[left, right]`**, with `T` denoting the number of video frames.

| Key | Shape | Meaning |
| --- | --- | --- |
| `mano_pose` | `(T, 2, 45)` | Local MANO axis-angle pose, radians |
| `mano_betas` | `(T, 2, 10)` | MANO shape coefficients |
| `global_orient` | `(T, 2, 3)` | Camera-space root axis-angle, radians |
| `translation` | `(T, 2, 3)` | Camera-space MANO translation, metres, recovered by PnP |
| `joints_2d` | `(T, 2, 21, 2)` | Predicted joints in original-video pixels |
| `bboxes` | `(T, 2, 4)` | Inference boxes after square/scale adjustment, `x1, y1, x2, y2` pixels |
| `valid` | `(T, 2)` | Whether a hand observation is present |
| `intrinsics` | `(4,)` | Input `fx, fy, cx, cy` |
| `fps` | scalar | Video frame rate |

Missing detections have `valid=False` and NaN predictions. The JSON file records the checkpoint, weight mode, coordinate convention, frame count and sampling steps. The CLI exports numerical predictions; it does not render overlay or trajectory videos.

### Practical notes

- **One person per video.** The detector chooses the highest-confidence hand of each side per frame; multi-person identity tracking is not implemented.
- **Detection gaps.** Inference uses up to 128-frame windows within contiguous detections and does not fill gaps between runs.
- **Long videos.** Decoded frames are kept in host memory. Start with a short clip or limit the number of frames.
- **Pose ambiguity.** PnP uses predicted joints and supplied intrinsics; severe occlusion or inaccurate calibration can affect global pose and depth.
- **Model compatibility.** This release supports the current DINOv3 ViT-H/16+, 512-pixel joint 2D–3D architecture. Other backbones and ablation checkpoints are rejected.

## Project Layout

```text
JoHan/
├── src/johan/
│   ├── cli.py                 # Video input, detection and output export
│   ├── backbone.py            # DINOv3 feature extraction
│   ├── checkpoint.py          # Configuration and strict weight loading
│   ├── sampling.py            # Flow inference
│   ├── hand_network_perjoint.py
│   ├── hand2d_perjoint_net.py  # Joint 2D–3D architecture
│   ├── geometry.py            # 2D denormalization and camera-space PnP
│   └── mano_smplx_layer.py     # MANO hand model
├── scripts/export_checkpoint.py
├── tests/smoke.py
├── VALIDATION.md
└── pyproject.toml
```

Maintainers can export a new inference-only checkpoint bundle with `scripts/export_checkpoint.py`; optimizer state and training data are not required for inference.

## Acknowledgements

We thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3), [WiLoR](https://github.com/rolpotamias/WiLoR) and [MANO](https://mano.is.tue.mpg.de/) for their models and tools. Reusable tensor, transformation and transformer utilities build on Brent Yi's MIT-licensed work; the original copyright notice is retained in [LICENSE](LICENSE).

## Citation

If JoHan is useful for your research, please cite:

```bibtex
@misc{xu2026johan,
  title={Video-Conditioned Generative Joint 2D-3D Hand Motion Recovery},
  author={Xu, Chen and Li, Yunqi and Huang, Binbin and Yi, Brent and Gao, Shenghua and Ma, Yi},
  year={2026},
  eprint={2610.10512},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.10512}
}
```

## License

JoHan source code is released under the [MIT License](LICENSE). The weight repository documents the licenses for JoHan and its fine-tuned DINOv3 backbone separately. DINOv3, the WiLoR detector and MANO assets remain subject to their respective licenses and access requirements.
