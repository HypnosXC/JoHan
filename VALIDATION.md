# Validation

Target: the current ViT-H/512 model, source `2ff60ef`, v3 step 145000 RAW.
Inference implementation tested at commit `d56c223`.

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
