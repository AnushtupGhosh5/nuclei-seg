import numpy as np
import pytest

from nuclei_seg.mamba_unet.metrics import MetricAccumulator, classification_report


def test_classification_report_imbalanced_and_absent_classes():
    confusion = np.array([[8, 2, 0], [1, 1, 0], [0, 0, 0]])
    report = classification_report(confusion, {0: "background", 1: "nucleus", 2: "absent"})
    nucleus = report.iloc[1]
    assert nucleus.recall == pytest.approx(0.5)
    assert nucleus.precision == pytest.approx(1 / 3)
    assert nucleus.f1 == pytest.approx(0.4)
    assert nucleus.specificity == pytest.approx(0.8)
    assert nucleus.accuracy == pytest.approx(0.75)
    assert nucleus.balanced_accuracy == pytest.approx(0.65)
    overall = report[report.scope == "overall"].iloc[0]
    assert overall.accuracy == pytest.approx(0.75)
    assert overall.balanced_accuracy == pytest.approx(0.65)
    assert overall.micro_f1 == pytest.approx(0.75)
    assert overall.weighted_recall == pytest.approx(0.75)
    assert overall.f1 == overall.macro_f1
    assert list(report.scope[-2:]) == ["overall", "overall_no_background"]


def test_empty_report_has_finite_aggregates():
    report = classification_report(np.zeros((2, 2), dtype=int), {0: "background", 1: "nucleus"})
    overall = report[report.scope == "overall"].iloc[0]
    assert overall.accuracy == overall.balanced_accuracy == overall.macro_f1 == 0


def test_matched_classification_is_separate_from_detection():
    accumulator = MetricAccumulator(0.5, 40, {0: "background", 1: "a", 2: "b"})
    instances = np.zeros((32, 32), dtype=np.int32)
    instances[2:6, 2:6] = 1
    instances[20:24, 20:24] = 2
    true_types = np.where(instances > 0, instances, 0)
    predicted_types = np.where(instances > 0, 1, 0)
    accumulator.update("tile", instances, true_types, instances, predicted_types, 1)
    summary, _, pixel, instance, _ = accumulator.finalize()
    report = instance.attrs["classification_report"]
    assert report.loc[report.scope == "overall", "accuracy"].iloc[0] == pytest.approx(0.5)
    assert summary["instance_matched_overall_balanced_accuracy"] == pytest.approx(0.5)
    assert "balanced_accuracy" in pixel.columns
    assert "official_weighted_f1" in instance.columns
