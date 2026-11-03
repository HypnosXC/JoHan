# Validation

Target: the current ViT-H/512 model, source `2ff60ef`, v3 step 145000 RAW.
Real-video inference tested at commit `d56c223`; final code cleanup verified at
commit `b4f311e`.

- All package modules import without loading the original package, training
  datasets, Aria, or body-estimation code.
- CPU integration checks cover non-square-image intrinsics scaling, box-to-pixel
  conversion, detection gaps and final overlapping video windows.
- The actual 145000 hand-network weights load strictly. A two-frame, two-step
  forward run with synthetic DINO features produces finite hand/2D outputs.
- With the same weights, synthetic features and seed, the extracted sampler
  exactly matches the original sampler for local pose, shape, 2D joints and
  confidence: maximum absolute difference 0.0 in all four outputs.

These checks ran through CPU Slurm jobs 421008 (integration/model portion) and
421026 (reference sampler comparison, completed successfully). The initial
reference harness in 421008 failed on an AST line-number issue; 421026 corrected
that harness. This is not a GPU accuracy or performance benchmark.

Real-video end-to-end smoke job 421022 completed successfully on CPU in 4m46s:
two frames, two flow steps, four detected hand observations. It used the actual
ViT-H backbone, paired 145000 RAW weights, WiLoR detector and both MANO models,
and exported camera-space results through PnP. This short smoke check does not
establish full-length-video accuracy or GPU numerical/performance parity.

No model weights or licensed third-party assets are bundled in this repository.

Final CPU job 421048 completed with exit code 0 after removing obsolete
conditioning utilities. Package/geometry/window checks and all four exact
reference-sampler comparisons passed again. The saved real-video artifact
contains four valid observations with finite pose, shape, orientation,
translation and 2D joints, and nonzero translations. The real-video pipeline
was not rerun at this final cleanup commit.

Direct trained-backbone loading was verified at commit `9cf8e12` in CPU Slurm
job 421271 (completed, exit code 0). Features from one 512-pixel input matched
exactly between the former pretrained-then-fine-tuned loader and the new
architecture-only initialization followed by strict fine-tuned loading.
Package/geometry/window checks and actual hand-weight sampling also passed.
A two-frame real-video run produced four finite hand observations using only
the paired trained weights and YAML configurations: no original DINOv3 weights,
train_meta.json or checksum manifest were supplied to that inference run.
Coordinate scale is read directly from model_config.yaml. This check does not
establish GPU performance or full-video accuracy.
