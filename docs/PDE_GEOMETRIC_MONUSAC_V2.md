# MoNuSAC and GLySAC geometric Mamba v2 revision audit

The v2 configuration is `configs/pde_geometric_mamba_monusac_v2.json`, with
output directory `outputs/pde_geometric_mamba_monusac_v2_hybrid`. The MoNuSAC
launcher now defaults to this configuration. The v1 JSON, completed v1
outputs, and pinned official VMamba implementation are preserved. No full
training was performed for this revision. GLySAC uses
`configs/pde_geometric_mamba_glysac_v2.json` and the fresh output
`outputs/pde_geometric_mamba_glysac_v2_hybrid`.

The scientific checkpoint **always minimizes total validation loss**:
`val_total < best_val_loss` is the only improvement test. The selected model
is `geometric_best_checkpoint.pth`; `geometric_latest_checkpoint.pth` remains
the resumable latest model. No separate best-metric or best-loss checkpoint is
written. PQ, Dice, F1, AJI, accuracy, and all other evaluation metrics are
logged but do not affect selection, patience, or the epoch-based StepLR.
There are no configurable checkpoint metric or mode fields. Metadata records
`"checkpoint_selection": "minimum_validation_total_loss"`,
`best_validation_loss`, and the selected `best_epoch`. Patience is eight
non-improving validation events; ties count as non-improvements and lower loss
resets the counter. The counter accumulates
during warmup, but stopping is disabled until epoch 80, giving the epoch-75
StepLR decay a full subsequent validation interval. Configuration validation
rejects a minimum stopping epoch below 75. Epochs in history are one-based;
checkpoint epochs are zero-based.

Best and latest checkpoint dictionaries save the model, optimizer, scheduler,
AMP scaler, epoch, history, `best_val_loss`, `best_epoch`, and
`bad_validations`. The loss-only tracker contains just those last three fields.
Resume restores
these fields. When resuming latest into a new directory, the previous selected
best checkpoint must accompany it so final evaluation can use the actual best
model. Superseded metric-selected checkpoints are rejected before any output
writes. Older loss-selected checkpoints remain compatible where their
architecture matches. A missing best epoch is recovered from the canonical
selected checkpoint; a missing embedded history is recovered from the adjacent
CSV, limited to the resumed epoch. If a legacy patience counter is unavailable,
it resets to zero with an explicit warning. Resume with the same architecture
and objective configuration. Worker/sampler RNG state is not serialized, so
resume does not promise bitwise equality to an uninterrupted run.

The TP head still has five logits, background plus four nucleus types.
With `type_loss_foreground_only=true`, CE and Dice operate only where
`instances > 0` and `types > 0`. Class zero never contributes as a target;
all five logits remain in the training softmax denominator. Dice excludes
background and averages only classes present in the batch. An empty or wholly
untyped batch produces a differentiable zero type loss. The original CE/Dice
objective remains available with the flag disabled.

The v2 class-weight basis is `instance`: count each training-tile instance once
by its majority nonzero type, resolving ties toward the lowest ID and excluding
untyped instances. Weights are inverse square roots of instance counts,
normalized to mean one over foreground classes present in training. Absent
classes receive neutral weight one; background receives weight zero in this
objective. `foreground_pixel` and `legacy_pixel` bases remain available. Legacy
weights retain the previous whole-image pixel-frequency formula and
normalization. Actual training counts, class names, weights, counting scope,
and formula are recorded in `geometric_run_config.json` before the first
training epoch. Validation/test labels never determine these weights.

The local annotation-only audit found 178 training tiles, with instance counts
12,479 epithelial, 13,268 lymphocyte, 529 macrophage, and 531 neutrophil.
Their weights are respectively 0.342626, 0.332283, 1.664114, and 1.660977;
background count/weight are zero. The split identity is
`c3d3eca805b48ec98457e4a5739b9936720058780ba3163dc6c2422e951d9cc7`.
Counts must be recomputed if the dataset/split changes.

GLySAC retains `type_encoding="auto"` and the existing classes background,
other, lymphocyte, epithelial. The local data are detected as
`original_10_class` and merged using the existing GLySAC mapping. Its v2 config
deliberately retains the legacy all-pixel type CE/Dice objective and
`type_weight_basis="legacy_pixel"`; MoNuSAC's type flags/weights are not copied.
Weights are calculated exclusively from the GLySAC training split. Both
datasets use source stage zero and a 32x32 guide to retain more spatial samples;
this choice does not by itself establish better segmentation accuracy.

The full GLySAC audit uses 29 training tiles. Pixel counts for background,
other, lymphocyte, epithelial are 24,153,725; 845,405; 1,603,020; 2,397,850,
with weights 0.298493, 1.595487, 1.158661, 0.947359 respectively. Its split
identity is `edf16c64430947fcc94df06147d0798a373cae1f9fa2b3b4da16125ed5098d2d`.

Inference normalizes TP logits over channels 1..4 before overlap blending,
then selects their argmax and adds one to the class ID. Type zero is forced
outside final predicted instance support. Foreground normalization avoids
underflow or patch suppression caused by an unconstrained background logit.
NP supplies foreground extent; PDE supplies internal peaks/topography;
the existing mask-constrained watershed supplies final instances.

The history logs `mask_ce`, `mask_dice`, `field_foreground`,
`field_background`, `type_ce`, `type_dice`, `guide_foreground`,
`guide_background`, and `total`, each with train/validation prefixes.
Aggregate mask/field/type/guide losses remain. Background regression terms are
logged before multiplying by the existing 0.1 weight. The existing objective
weights remain mask=1, field=1, type=1, guide=0.5.

The exact validation objective is the arithmetic mean, over the deterministic
validation patch batches, of the actual scalar returned by `PDEGeometricLoss`:

```text
L_mask  = weighted binary CE + foreground Dice
L_field = mean_fg SmoothL1(sigmoid(field), target)
        + 0.1 * mean_bg SmoothL1(sigmoid(field), target)
L_type  = configured weighted type CE + configured type Dice
L_guide = mean_fg SmoothL1(sigmoid(guide), interpolated target)
        + 0.1 * mean_bg SmoothL1(sigmoid(guide), interpolated target)
L_total = L_mask + L_field + L_type + 0.5 * L_guide
val_total = (1 / number_of_validation_batches) * sum(batch L_total)
```

SmoothL1 uses beta=0.1; its foreground/background means divide by their own
masked pixel counts, clamped to at least one. Guide targets are bilinearly
resized and guide foreground masks use nearest-neighbor resizing. Without a
guide head, `L_guide=0`. Custom loss weights remain configurable and recorded.
Validation `loss` is a compatibility alias of the canonical returned total.
An empty loader or non-finite returned loss is rejected. All subcomponents use
the same batch averaging. History includes `val_total`,
`best_val_loss_so_far`, and `bad_validations`; metrics remain separate columns.

For each instance, a four-neighbor binary erosion defines strict interior and
an explicit one-pixel inner contour. Jacobi updates approximate `Delta u=-1`
only in that interior; contour and outside remain exactly zero on every
iteration. Each non-degenerate instance is independently normalized to maximum
one. Instances without an interior receive zero targets. The finite iteration
count approximates a discrete Poisson solve; max normalization rescales its
source/amplitude, so the normalized field does not literally retain unit
source strength. This is supervised regression to a PDE-derived target.

Normal ordering uses serpentine spatial windows, nearest quantized gradient
orientation, perpendicular-projection ray bins, ascending potential inside
each group, and raster-index tie breaks. Gradients below 0.01 use a stable
direction toward the unweighted window center. The v2 settings are eight
directions, ray width one feature token, and window side four. Ray bins use
`floor(projection / ray_width + 1e-6)`. The old global potential/gradient-angle
sort remains available as `pde_normal_scan_algorithm="legacy_global"` for
an ordering ablation. Tangential ordering remains window, 16-bin potential
band, local angle around a potential-weighted window center, and raster tie.
Reverse orders are exact flips and inverse permutations restore raster layout.

The native guide is produced after `encoder_0`: 32x32, with a 192-channel
input to its 1x1 head. It is interpolated to each guided feature resolution:

| Guided SS2D stage | Feature/scan size | Input channels |
|---|---:|---:|
| encoder_1 | 32x32 | 192 |
| encoder_2 | 16x16 | 384 |
| encoder_3 | 8x8 | 768 |
| decoder_1 | 16x16 | 384 |

The MoNuSAC parameter count changes from **19,128,585 to 19,129,161**, a net
**+576**: two newly guided encoder_1 blocks add 768 gate parameters, while the
guide head loses 192 input weights. Guided block count changes from six to
eight; pretrained parameter count remains 19,121,280. Parameter count alone
does not characterize the extra scan/sort runtime at 32x32.

Synthetic diagnostics use seed 42 Gaussian guide noise with standard deviation
1e-3 at the original guide resolution, followed by interpolation and clipping.
Jumps include all consecutive steps. Directional alignment excludes group and
window transitions, unresolved endpoint/averaged gradients, and zero-length
steps. Normal cosine is signed ascent; tangent cosine is absolute, allowing
either contour direction. Counts of valid steps accompany scores.

On the irregular block-shaped regression field resized to 16x16:

| Diagnostic | Legacy normal | Revised normal | Tangential |
|---|---:|---:|---:|
| Mean jump, tokens | 3.6304 | 1.8735 | 1.4571 |
| Maximum jump, tokens | 19.8494 | 7.6158 | 7.0000 |
| Mean absolute rank change | 30.3828 | 0.1484 | 0.0078 |
| Fraction unchanged rank | 13.6719% | 85.1563% | 99.2188% |
| Valid local alignment steps | 0 | 57 | 67 |
| Signed normal / absolute tangent cosine | unavailable | 0.9804 | 0.8734 |

The zero valid legacy steps means its locally filtered alignment is unavailable,
not that global alignment has been established. Permutation, inverse, and
gather/restore checks are exact for every order. On the smooth two-peak field,
normal mean/max jumps change from 6.0636/18.0278 to 1.7474/6.0000 tokens;
normal ascent cosine is 0.8897 over 116 steps, tangent absolute cosine is
0.8533 over 126 steps. These are synthetic results, not new MoNuSAC metrics.

The CPU-only audit is reproducible with:

```bash
PYTHONPATH=src python3 scripts/check_geometric_revision.py \
  --count-training-labels --output /tmp/geometric-v2-audit.json
```

The generated audit is saved at
`outputs/pde_geometric_mamba_monusac_v2_hybrid/loss_only_revision_checks.json`
and `outputs/pde_geometric_mamba_glysac_v2_hybrid/loss_only_revision_checks.json`, and is
separate from actual training metadata. Audit output creation refuses to
overwrite an existing file. Launchers accept explicit `--output-dir`; smoke
runs otherwise append `_smoke`. Fresh training refuses a directory already
containing training metadata or checkpoints.

The earlier `revision_checks.json` is preserved as a historical audit from
before the loss-only correction; its superseded selection settings are not
used by either current config or the experiment.

Current run commands:

```bash
./run_geometric_mamba_monusac.sh
./run_geometric_mamba.sh
./run_geometric_mamba_monusac.sh --smoke-test
./run_geometric_mamba.sh --smoke-test
./run_geometric_mamba_monusac.sh --scan-mode normal
./run_geometric_mamba.sh --scan-mode tangential
./run_geometric_ablations.sh monusac --smoke-test
./run_geometric_ablations.sh glysac --smoke-test
```

Both dataset runners support cartesian, pde, normal, tangential, and hybrid
scan modes, arbitrary argument passthrough, `CONFIG` and `IMAGE_NAME` environment
overrides, and `--help`/`-h` without starting Docker. Docker still uses
`--gpus all --shm-size=8g --network host`, the project mounted at `/app`,
`PYTHONPATH=src`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. MoNuSAC still checks its
prepared split CSV. The ablation runner requires an explicit dataset and
`--smoke-test` or `--full`; it sequentially runs cartesian, normal, tangential,
hybrid and stops on the first failure. No full ablations were launched.

Verification: `py_compile` passed for changed Python files; all three shell
runners passed `bash -n`, `--help`, and `-h`; `git diff --check` passed.
`pytest tests/test_pde_geometric.py` passed **70 tests with GPU access**, with
no skips. This includes all actual CUDA ablation shape tests and zero-gate
equivalence. The CPU-only run passed 63 and skipped seven GPU tests. Tests
cover both mandated loss/PQ examples, tie handling, loss-only patience,
optimizer/scheduler/scaler/epoch/history restoration, legacy loss-state
migration, rejection of metric-selected resumes, exact shell argument quoting,
explicit ablation execution/failure behavior, and mode-specific output paths.

The GPU-enabled Docker runtime is available even though the sandbox host and
CPU-only container checks do not expose the driver. One MoNuSAC and one GLySAC
hybrid smoke run completed, each with one reduced-data epoch (two training
tiles, one validation tile, one test tile). Both loaded 197 pretrained tensors,
verified NP/PDE/TP outputs at 256x256 and guides at 32x32, and saved loss-selected
best/latest checkpoints, full diagnostics, predictions, figures, and archives:

| Dataset | Selected epoch | Total validation loss | Output directory |
|---|---:|---:|---|
| MoNuSAC | 1 | 13.3941144943 | `outputs/pde_geometric_mamba_monusac_v2_loss_only_smoke` |
| GLySAC | 1 | 10.7896060944 | `outputs/pde_geometric_mamba_glysac_v2_loss_only_smoke` |

Both histories contain all 13 required loss terms plus `val_total`,
`best_val_loss_so_far`, `bad_validations`, and separate evaluation metrics.
Checkpoint/run metadata explicitly say `minimum_validation_total_loss`.
There are no separate best-loss or best-metric files. Smoke counts/weights are
computed from their reduced training subsets and are distinct from full-split
audits. The smoke runs are pipeline checks, not accuracy experiments or stable
timing benchmarks. PyTorch warned that the CUDA binary cross-entropy kernel is
not covered by its deterministic implementation; deterministic mode remains
the existing warn-only setting. No full training was run.

The local scans approximate ray ascent and local level-set traversal, not
numerical streamline integration. Group/window transitions still carry
selective-scan state; disconnected nuclei can share a group. Quantization,
exact potential ties, and guide errors can still change discrete ranks.
Guidance is predicted and detached for sorting, with regression supervision;
there is no PDE residual loss, so this must not be called a PINN. A 32x32 guide
improves sampling relative to 16x16 but does not establish accurate internal
orientation for every nucleus. Accuracy/runtime benefits require matched
ablations and new GPU experiments.

For exact reproduction of completed v1 results, use the original JSON with
code commit `1a4c261464288bdce5460235156ff76c9047d65e`. Merely selecting the
old JSON in the revised code retains its old architecture/type objective and
minimum-loss principle but uses revised target construction and local scan
defaults. Old
architecture model weights remain loadable with that architecture; v2 changes
the guide-head input shape and adds two gates, so old geometric checkpoints
are not directly loadable into v2 with strict state loading. Start v2 from the
existing official ImageNet checkpoint. NP/TP channel APIs remain compatible.

Changed or added files:

- `configs/pde_geometric_mamba_monusac_v2.json`: revised experiment settings.
- `configs/pde_geometric_mamba_glysac_v2.json`: geometry v2 with GLySAC encoding/classes and objective retained.
- `run_geometric_mamba_monusac.sh`: defaults to v2.
- `run_geometric_mamba.sh`: defaults to GLySAC v2.
- `run_geometric_ablations.sh`: explicit sequential ablations with fail-fast behavior.
- `src/nuclei_seg/mamba_unet/geometric_config.py`: auditable options/validation.
- `src/nuclei_seg/mamba_unet/geometric_checkpoint.py`: strict minimum-loss selection/patience state and legacy migration checks.
- `src/nuclei_seg/mamba_unet/geometric_experiment.py`: checkpointing/resume, inference, history, metadata, diagnostics, output guards.
- `src/nuclei_seg/mamba_unet/geometric_loss.py`: foreground typing, weights, loss subcomponents.
- `src/nuclei_seg/mamba_unet/data.py`: type-count bases, including instances.
- `src/nuclei_seg/mamba_unet/geometric_model.py`: normal-scan option propagation.
- `src/nuclei_seg/mamba_unet/pde_field.py`: strict Dirichlet contour.
- `src/nuclei_seg/mamba_unet/pde_scan.py`: local normal rays, groups, diagnostics.
- `tests/test_pde_geometric.py`: scientific/selection/shape regression tests.
- `scripts/check_geometric_revision.py`: reusable CPU parameter/scan/label audit.
- `docs/PDE_GEOMETRIC_SCAN.md`: identifies the historical v1 description.
- `docs/PDE_GEOMETRIC_MONUSAC_V2.md`: this revision audit.
- `outputs/pde_geometric_mamba_monusac_v2_hybrid/loss_only_revision_checks.json`: corrected MoNuSAC audit; the earlier report is preserved.
- `outputs/pde_geometric_mamba_glysac_v2_hybrid/loss_only_revision_checks.json`: GLySAC audit including its own training-label weights.
- `outputs/pde_geometric_mamba_monusac_v2_loss_only_smoke/` and its `_results.zip`: MoNuSAC smoke artifacts.
- `outputs/pde_geometric_mamba_glysac_v2_loss_only_smoke/` and its `_results.zip`: GLySAC smoke artifacts. Generated outputs remain ignored by Git.
