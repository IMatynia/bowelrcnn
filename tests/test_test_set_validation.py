import numpy as np
import pytest
from src.new.models.testing.statistics import BasicMetricsCalculator, bowel_sound_iou, BowelSoundRaw


def test_metrics_zeros():
    FN, FP, TN, TP = 0, 0, 0, 0
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((0., 0., 0., 0., 0., 0.))


def test_metrics_simple():
    FN, FP, TN, TP = 0, 0, 1, 1
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    # accuracy, precision, recall/sensitivity, specificity, f1_score
    assert res.values() == pytest.approx((1., 1., 1., 1., 1., 1.))

    FN, FP, TN, TP = 1, 0, 1, 1
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((0.5, 0.666666, 1.0, 0.5, 1.0, 0.6666666), abs=0.001)

    FN, FP, TN, TP = 0, 1, 1, 1
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((0.5, 0.666666, 0.5, 1.0, 0.5, 0.6666666), abs=0.001)

    FN, FP, TN, TP = 1, 1, 1, 1
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((1/3, 0.5, 0.5, 0.5, 0.5, 0.5), abs=0.001)

    FN, FP, TN, TP = 10, 5, 20, 5
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((5/20, 0.6250, 0.5000, 0.3333, 0.8000, 0.4000), abs=0.001)

    FN, FP, TN, TP = 5, 10, 200, 20
    res = BasicMetricsCalculator.calculate_metrics(FN, FP, TN, TP)
    assert res.values() == pytest.approx((20/35, 0.9362, 0.6667, 0.8000, 0.9524, 0.7273), abs=0.001)


@pytest.mark.parametrize(
    "bs1, bs2, iou_result",
    [
        [BowelSoundRaw(start=1, end=2), BowelSoundRaw(start=1, end=2), 1.0],
        [BowelSoundRaw(start=1, end=2), BowelSoundRaw(start=2, end=3), 0.0],
        [BowelSoundRaw(start=1, end=3), BowelSoundRaw(start=2, end=5), 1/4],
    ]
)
def test_bs_iou(bs1, bs2, iou_result):
    assert bowel_sound_iou(bs1, bs2) == pytest.approx(iou_result)
