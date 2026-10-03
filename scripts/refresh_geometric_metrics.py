"""Refresh completed geometric reports from saved predictions without inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nuclei_seg.mamba_unet.data import Record, read_annotation, detect_type_encoding
from nuclei_seg.mamba_unet.metrics import MetricAccumulator, write_classification_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir
    config = json.loads((directory / "geometric_run_config.json").read_text())
    if config["smoke_test"]:
        parser.error("This refresh script requires full-image predictions, not cropped smoke predictions")
    manifest = pd.read_csv(directory / "split_manifest.csv")
    encoding = config["type_encoding"]
    if encoding == "auto":
        records = [Record(stem=row.stem, split=row.split, image=Path(row.image), label=Path(row.label))
                   for row in manifest.itertuples()]
        encoding = detect_type_encoding(records, "auto")
    for split in ("val", "test"):
        accumulator = MetricAccumulator(config["match_iou"], config["magnification"],
                                        dict(enumerate(config["type_classes"])))
        for record in manifest[manifest.split == split].itertuples():
            truth, types = read_annotation(Path(record.label), encoding)
            with np.load(directory / f"geometric_predictions_{split}" / f"{record.stem}.npz") as prediction:
                ignore = prediction["ignore_map"].astype(bool)
                accumulator.update(record.stem, np.where(ignore, 0, truth), np.where(ignore, 0, types),
                                   prediction["inst_map"], prediction["type_map"],
                                   float(prediction["inference_seconds"]))
        summary, per_image, pixel, instance, panoptic = accumulator.finalize()
        # Preserve additional metadata from the completed run.
        path = directory / f"geometric_metrics_{split}.json"
        previous = json.loads(path.read_text())
        previous.update(summary)
        path.write_text(json.dumps(previous, indent=2, allow_nan=False))
        for name, frame in (("per_image", per_image), ("classwise_pixel", pixel),
                            ("classwise_instance", instance), ("classwise_panoptic", panoptic)):
            frame.to_csv(directory / f"geometric_{name}_{split}.csv", index=False)
        write_classification_reports(directory, split, pixel, instance, "geometric_")
        print(f"Refreshed {split}: {summary['n_images']} images")


if __name__ == "__main__":
    main()
