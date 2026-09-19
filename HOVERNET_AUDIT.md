# Local HoVer-style implementation audit (historical)

This document describes corrections made to the project's local port. It does
not certify an exact paper reproduction. Paper-reporting experiments must use
the pinned official repository, configuration, post-processing, and metrics
listed in `UPSTREAM_REPRODUCTIONS.md`.

Compared against the official `vqdang/hover_net` PyTorch repository and its
released fast-mode MoNuSAC checkpoint.

| Area | Previous implementation | Official behavior | Correction |
|---|---|---|---|
| Network | 12.3M-parameter U-Net trained from scratch | 37.6M-parameter Preact-ResNet50 plus three dense decoders | Added `hovernet_fast`; official checkpoint loads strictly |
| Geometry | 256 input and 256 supervised/output pixels | 256 input and valid central 164 output pixels | Center-cropped supervision and 164-stride mirror-padded assembly |
| HV head | `tanh`-bounded regression | Linear regression | Removed `tanh` for new checkpoints; legacy checkpoints retain it |
| HV gradient loss | One-pixel differences of both channels in both directions | 5x5 directional Sobel-like kernels on H and V separately, nuclear mask only | Implemented reference MSGE |
| Dice loss | Foreground-only mean | Sum over all channels, including background | Matched reference formulation |
| Watershed | 5x5 absolute Sobel and raw boundary elevation | Normalized 21x21 directional Sobel, inverse blurred distance, filled/opened markers | Ported reference algorithm |
| Instance type | Mean class probability | Majority pixel class with background fallback | Matched reference behavior |
| Augmentation | Flip, 90-degree rotation, weak brightness/contrast | Arbitrary affine, broad color/stain perturbation, blur/noise | Added rotation/scale, HSV, contrast, blur, and noise |
| Training | AdamW/cosine, one phase | ImageNet initialization; frozen and unfrozen 50-epoch Adam/StepLR phases | Added ImageNet weights and optimizer/scheduler reset at phase two |
| Metrics | AJI+ reported as AJI; type matching at IoU > 0.5 | AJI and AJI+ are distinct; reported type scores use 12px centroid matching | Report both AJI variants and HoVer-style centroid type F1 |

## Controlled result

Re-evaluating the original U-Net checkpoint after only tiling, post-processing,
and type-assignment corrections improved mean test PQ from **0.4236 to 0.5079**,
DQ from **0.5522 to 0.6551**, and pooled type macro-F1 from **0.4765 to
0.5643**. This isolates a large pipeline error independently of retraining.

## MoNuSAC test-label caveat

The supplied test XMLs in this workspace are sparse in multiple tiles. For
example, some images contain hundreds of visually apparent nuclei but only a
few XML polygons. The official released checkpoint therefore detects many
unlabelled nuclei and receives false-positive penalties: under this local XML
evaluation it obtains mean PQ **0.3302**, despite qualitatively denser and more
plausible overlays. These values must not be presented as the published
HoVer-Net benchmark. Quantitative comparison requires the exact challenge
evaluation masks/protocol or an exhaustively annotated test set.

## Artifacts

- Corrected legacy checkpoint: `outputs/results/monusac/unet_hover_pipeline_fixed/`
- Official released checkpoint: `outputs/pretrained/hovernet_fast_monusac_type_tf2pytorch.tar`
- Official reference predictions: `outputs/results/monusac/official_hovernet_reference/`
- ImageNet Preact-ResNet50 initialization: `outputs/pretrained/imagenet_resnet50_preact.tar`
