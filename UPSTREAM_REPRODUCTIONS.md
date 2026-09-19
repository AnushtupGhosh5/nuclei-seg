# Upstream reproduction policy

Paper results must come from the authors' code at a recorded commit, without
architecture, loss, augmentation, post-processing, inference, or metric
substitutions. Compatibility patches, if unavoidable, must be isolated and
documented separately from the upstream source.

## HoVer-Net

- Repository: <https://github.com/vqdang/hover_net>
- Pinned commit: `67e2ce5e3f1a64a2ece77ad1c24233653a9e0901`
- MoNuSAC mode: authors' `fast` mode (256 input, 164 valid output)
- Required scope: upstream model, target generation, augmentation, two-phase
  training recipe, watershed post-processing, inference tiling, and metrics

The Python 3.12 container applies a compatibility alias for `np.bool`, which
was removed after the upstream NumPy pin. This changes no augmentation logic;
it only permits the authors' pinned `imgaug==0.4.0` to run on NumPy 1.26.

The word `fast` is the official mode used by the authors for their MoNuSAC
checkpoint. It must not be replaced by a custom speed-optimized implementation.

## SoNNeT

- Repository: <https://github.com/QuIIL/Sonnet>
- Pinned commit: `6cc6c2bbada1084edc82d041d13c307a109806bc`
- Upstream runtime: Python 3.6 and TensorFlow 1.12
- Required scope: upstream patch extraction, model, self-guided ordinal loss,
  training, inference, processing, and `compute_stats.py`

SoNNeT needs a dedicated legacy-runtime container. It must not be translated
to PyTorch or merged into the current shared loss/post-processing pipeline for
the reproduction experiment.

The standalone Kaggle template is `notebooks/sonnet_kaggle.ipynb`. Its legacy
environment and outputs live entirely under `/kaggle/working`, so it cannot
modify this project's Docker environment.
