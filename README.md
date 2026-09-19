# Nuclei segmentation and classification

This project studies simultaneous nuclear instance segmentation and type
classification. The active baseline is the original 2015 U-Net topology:
64/128/256/512/1024 channels, unpadded convolutions, ReLU activations, max
pooling, transposed-convolution upsampling, and cropped skip connections. It
has 31,032,265 parameters for MoNuSAC and uses the original 572 input / 388
output geometry. The canonical final decoder feature feeds the NP, HV, and
type heads required by this task.

1. **NP branch:** two-class background/nucleus segmentation.
2. **HoVer branch:** normalized horizontal and vertical offsets from each
   nuclear pixel to its instance centroid.
3. **Type branch:** pixel-wise nuclear type, including background.

At inference, gradients of the H/V maps produce separation evidence for a
marker-controlled watershed. Nuclear type is assigned by averaging type
probabilities inside each resulting instance. `--type-boundary-weight` can
optionally add type-probability changes as a weak watershed boundary cue;
the default `0` keeps post-processing faithful to the standard HoVer approach.

The local `hovernet.py` port is retained for earlier experiments, but it is not
the paper-reporting HoVer-Net reproduction. Future HoVer-Net and SoNNeT results
must run the pinned authors' repositories and their own training,
post-processing, and metric code. See [`UPSTREAM_REPRODUCTIONS.md`](UPSTREAM_REPRODUCTIONS.md).

### Official HoVer-Net on GlySAC

The sibling `../hover_net` checkout is wired to read the existing raw GlySAC
dataset in this repository and use the same three-foreground-class mapping.
The raw images and annotations are not copied into the upstream checkout.
HoVer-Net's required derived `.npy` patches and all checkpoints remain under
this repository:

```text
data/glysac_dataset/                         # existing source data
data/processed/official_hovernet/glysac/     # derived training patches
outputs/models/glysac/official_hovernet_fast_glysac_3class/  # checkpoints/logs
```

Validation is a deterministic, source-grouped 20% holdout from `Train`; the
dataset's `Test` images and labels are not used during training or validation.

Prepare patches once, then train with the official model/training code:

```bash
./build.sh
./run_official_hovernet.sh prepare
./run_official_hovernet.sh train
```

Or run both steps with `./run_official_hovernet.sh`. The configuration uses
HoVer-Net fast geometry (540-pixel extraction windows, 256-pixel network input,
164-pixel output), four type logits including background, and the ImageNet
Preact-ResNet50 weights already stored in `outputs/pretrained`. Set `GPU_IDS`,
`RUN_NAME`, or the `HOVERNET_*_BATCH_SIZE` environment variables to override
the defaults.

## Datasets

- **MoNuSAC:** background plus epithelial, lymphocyte, macrophage, and
  neutrophil. Ambiguous test annotations are discarded during preparation so
  they do not become an unsupported fifth nucleus class.
- **GlySAC:** raw IDs 1/2/9/10 become miscellaneous, 4/5/6/7 become
  lymphocyte, and 3/8 become epithelial. The model therefore predicts three
  foreground classes plus background.

Training and validation are split by source patient/slide group to reduce data
leakage. The official dataset test splits remain untouched.

Training crops use class-aware foreground sampling before augmentation. This
oversamples images containing rare macrophage and neutrophil annotations while
retaining a configurable fraction of unconstrained random crops.

## Commands

Build once and train the configured GlySAC HoVer-Net experiment:

```bash
./build.sh
./run.sh
```

Select a separate run name without rebuilding:

```bash
RUN_NAME=glysac_hovernet_repeat ./run.sh
```

The training command automatically prepares raw XML/MAT annotations. It can
also be run explicitly:

```bash
python3 src/run.py prepare --dataset monusac --data-dir data
python3 src/run.py train --dataset monusac --data-dir data --output-dir outputs
python3 src/run.py evaluate \
  --checkpoint outputs/models/monusac/unet_hover_baseline/best.pt \
  --data-dir data --output-dir outputs
```

The default `runScript.sh` trains the fast HoVer-Net architecture on GlySAC at
its reference 256-pixel input / 164-pixel output geometry. It rebuilds the
processed labels with the three-class mapping, initializes the encoder from
ImageNet Preact-ResNet50 weights, trains for 50 frozen-encoder epochs followed
by 50 unfrozen epochs, and evaluates the best checkpoint on the untouched test
split. An interrupted run resumes automatically from `last.pt`; choose a new
`RUN_NAME` to start a separate experiment.

## Recorded metrics

Every epoch records total and component losses, binary Dice, and foreground
type pixel accuracy. Full test evaluation records binary Dice/IoU, AJI,
detection quality (DQ), segmentation quality (SQ), panoptic quality (PQ), AJI+,
detection F1, macro type F1, per-class F1, and per-class PQ. Results are saved
as JSON/CSV together with predicted instance and type maps.

`hover_f1_*` fields use the official 12-pixel centroid pairing and weighting.
The plain `pooled_f1_*` fields use IoU-matched instances and are retained for
comparison with earlier experiments.

For the default `original_unet_monusac_baseline_v2` run, outputs are organized
as:

```text
outputs/models/<dataset>/original_unet_monusac_baseline_v2/
  best.pt
  last.pt
  history.jsonl
  training_curves.png
  model_profile.json
  model_summary.txt

outputs/results/<dataset>/original_unet_monusac_baseline_v2/
  summary.json
  model_profile.json
  model_summary.txt
  per_image_metrics.csv
  predictions/*.npz
  overlays/*.png
  visualizations/*.png
```

`runScript.sh` performs both training and test evaluation, so future completed
runs automatically contain full-test metrics and cached predictions. By default,
only eight test images receive overlays and detailed NP/HoVer panels; set
`VISUALIZATIONS=4 ./run.sh` to change that limit without reducing test coverage.

The console and saved history report training, validation, and total time for
each epoch. Model and test profiling is written alongside each run, including
parameter count, parameter size, GFLOPs for the configured input tile, tile latency,
full-image inference time, post-processing time, and throughput. GFLOPs count
convolution and linear multiply-accumulates as two floating-point operations.
