# PDE-guided geometric selective scans

This is an experimental ablation framework, not a claim that the scan is an
exact PDE streamline method or that it improves the baseline. The pinned
official VMamba implementation in
`src/nuclei_seg/mamba_unet/official/mamba_sys.py` is unchanged.

## Architecture

The model keeps the native Mamba-UNet encoder, bottleneck, decoder, skip
connections, GLySAC split, seed 42, 256-pixel crops, augmentation, optimizer,
schedule, evaluation, and minimum-validation-loss checkpoint rule. The output
representation is deliberately changed for PDE ablations:

- NP head: two foreground/background logits.
- PDE head: one scalar-field logit.
- TP head: four GLySAC type logits (background plus three foreground types).
- Guide head in geometric modes: one coarse scalar-field logit at 16×16.

The mask supplies watershed foreground support. Smoothed PDE-field maxima
supply markers and the negative field supplies watershed topography. HoVer H/V
targets are not reintroduced into this model.

For a 256×256 input, encoder features are 64×64×96, 32×32×192,
16×16×384, and 8×8×768 (NHWC inside VMamba). The default guide is predicted
after encoder layer 1 at 16×16. Guided blocks are configurable; the default
uses encoder layers 2 and 3 plus decoder layer 1, six `SS2D` blocks in total,
at only 16×16 and 8×8 resolutions. Earlier layers stay Cartesian.

The model's public forward method accepts only `images`. Ground-truth PDE is
used to supervise the PDE and guide heads, never to construct training or
inference permutations.

## Discrete ordering

For each batch item and guided resolution, the detached predicted guide is
bilinearly resized to `[H,W]`, clamped to `[0,1]`, and differentiated with
centered finite differences.

Normal ordering is a deterministic stable lexicographic sort of:

1. scalar potential;
2. local gradient angle `atan2(du/dy, du/dx)`;
3. raster index as an exact tie breaker.

Tangential ordering avoids a single global center across a multi-nucleus
patch. The latent map is divided into configurable local windows (4×4 by
default) traversed in serpentine order. Within each window, pixels are stably
sorted by:

1. quantized potential level (16 bins by default);
2. polar angle around that window's PDE-weighted soft center;
3. raster index.

The reverse directions are exact flips of the forward sequences. Exact inverse
permutations are constructed with `scatter_`, then used to restore every scan
output to raster order. For input `[B,C,H,W]`, flattening gives `[B,C,L]`, a
permutation `[B,L]` is expanded and gathered across channels, selective scan
runs on `[B,K*C,L]`, and inverse gather restores `[B,C,L]` before reshaping.
`K=4` for PDE/hybrid and `K=2` for normal-only or tangential-only.

Permutations and their gradients/centers are computed once per distinct
resolution in each model forward and reused by all guided blocks at that
resolution. Construction is `O(B L log L)` per resolution. The sorting and
integer permutations are non-differentiable; the guide is explicitly detached
for this path. Its regression loss still trains the guide head normally.

## Scan modes and ablations

| Ablation | Representation and scan | Command |
|---|---|---|
| A | Original Cartesian VMamba + NP/HV/TP | `./run_mamba_unet.sh` |
| B | Cartesian VMamba + NP/PDE/TP | `./run_geometric_mamba.sh --scan-mode cartesian` |
| C | Four PDE-derived directions + NP/PDE/TP | `./run_geometric_mamba.sh --scan-mode pde` |
| D | Cartesian plus gated four-direction PDE branch | `./run_geometric_mamba.sh --scan-mode hybrid` |
| E | Normal forward/reverse only | `./run_geometric_mamba.sh --scan-mode normal` |
| F | Tangential forward/reverse only | `./run_geometric_mamba.sh --scan-mode tangential` |

Append `--smoke-test` to any command for a one-epoch, reduced-data pipeline
check. Use a distinct `--output-dir` for repeated runs. Default full run:

```bash
./build.sh
./run_geometric_mamba.sh --scan-mode hybrid
```

The identical pipeline can be run on MoNuSAC with its four foreground classes:

```bash
./prepare_monusac.sh --overwrite
./run_geometric_mamba_monusac.sh --smoke-test
./run_geometric_mamba_monusac.sh
```

Preparation converts XML/TIFF annotations to the same MAT/PNG record format,
uses a seed-42 patient-level validation split, and writes `ignore_map` for
`Ambiguous` test annotations. Evaluation masks those pixels in the true map,
prediction, and metrics. Ambiguous nuclei are therefore excluded rather than
silently assigned to one of the four MoNuSAC classes.
The canonical MoNuSAC IDs are background 0, epithelial 1, lymphocyte 2,
macrophage 3, and neutrophil 4.

The hybrid result is
`Y_cartesian + gate * Y_geometric`, with one learnable gate per inner channel.
Every gate starts exactly at zero. A regression test verifies that a converted
hybrid block is bit-exact with its Cartesian source at initialization.

## Training, pretraining, and outputs

The four logged loss terms are NP CE+Dice, foreground-focused Smooth-L1 PDE
regression, TP CE+Dice, and coarse-guide regression. Pretrained parameters use
learning rate `1e-5`; new heads and geometric gates use `1e-4`. All parameters
are fine-tuned.

The verified checkpoint is the official VMamba-T epoch-292 ImageNet-1K
classification checkpoint. Compatible encoder tensors retain their names and
shapes through `PDEGuidedSS2D` conversion. A smoke run loaded 197 unique
source-to-target assignments. Missing core tensors are the six new gates and
native decoder/upsampling tensors that do not exist in the classification
checkpoint; all are listed explicitly in `pretrained_load_report.json`.

Each run saves:

- `geometric_training_history.csv`, including every loss and validation metric;
- minimum-validation-loss best and latest checkpoints;
- validation/test summary JSON and per-image/classwise CSV files;
- compressed instance, type, mask-probability, and PDE predictions;
- a nine-panel validation figure (RGB, GT instances, GT PDE, predicted mask,
  predicted PDE, watershed, gradient magnitude, normal, tangent);
- predicted and synthetic 16×16 scan-rank figures;
- `geometric_scan_diagnostics.json` and `computational_overhead.json`;
- a checkpoint-free `*_results.zip` archive.

Metrics include Dice, IoU, accuracy, precision/recall/F1, AJI/AJI+, DQ/SQ/PQ,
binary and multiclass DQ/SQ/PQ, detection F1, macro and classwise type F1, and
classwise panoptic metrics. Validation metrics are logged but only validation
loss selects the checkpoint.

## Verification performed

The Docker GPU suite covers finite bounded per-instance PDE targets, separate
extrema for touching nuclei, clean-target watershed separation, permutation
bijections at multiple shapes, exact inverse reconstruction, true
forward/reverse relationships, distinct normal/tangential orders on an
irregular field, per-resolution cache reuse, no GT-guidance forward API,
unchanged original model structure, finite output shapes for all five modes,
and exact zero-gate hybrid equivalence. The current suite result is 21 passed.
Both the original NP/HV/TP pipeline and Cartesian/hybrid PDE pipelines also
complete end-to-end smoke runs and produce their final archives.

On the local GPU, a representative hybrid smoke run measured 18.76 ms versus
12.79 ms for Cartesian PDE inference at batch size one, about 1.47× full-model
forward overhead. Timing is hardware- and load-dependent; every run measures
and records its own value.

## Scientific limitations

- Potential sorting is not local streamline integration. Equal-potential
  tokens from different nuclei can become adjacent; measured normal-order
  spatial jumps can be large.
- Window-local tangential ordering is safer than one global polar center but
  can still jump between disconnected nuclei inside a window.
- Discrete ranks can change sharply under small guide perturbations. Every run
  reports rank stability under `1e-3` noise.
- Early predicted guidance can be poor. The detached ordering and zero-gated
  hybrid initialization protect the pretrained Cartesian path, but do not
  prove later learned guidance is useful.
- Sorting adds cost and training noise. Benefits, if any, must be established
  with the A–F ablations rather than inferred from implementation success.
- A future method may need connected-region grouping or local streamline
  integration. Those changes are intentionally outside this first controlled
  implementation.
