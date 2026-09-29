"""Tests for the confusion-matrix based segmentation metrics."""

import numpy as np
import pytest

from utils.data_utils import level_to_num_classes
from utils.metrics import SegmentationMetrics

# Every level is scored on its own confusion matrix, so the tests below always say
# explicitly which level they exercise.
LEVEL_0 = 0
LEVEL_2 = 2


def build_mask_with_every_class() -> np.ndarray:
    """
    Build a (1, 1, num_classes) mask holding one pixel of every class.

    Every class is present exactly once, which means a perfect prediction can only reach an
    mIoU of 1.0 if all classes are handled, not just the background. Levels with fewer classes
    simply ignore the higher labels, so the same mask works at every level.
    """
    num_classes = max(level_to_num_classes.values())
    mask = np.zeros((1, 1, num_classes), dtype=np.int64)
    for class_idx in range(num_classes):
        mask[:, :, class_idx] = class_idx
    return mask


@pytest.fixture
def metrics() -> SegmentationMetrics:
    """A freshly reset metrics object."""
    return SegmentationMetrics()


@pytest.mark.parametrize("level_idx", sorted(level_to_num_classes))
def test_perfect_prediction_scores_one(metrics, level_idx):
    """A prediction identical to the ground truth reaches an mIoU of 1.0."""
    target = build_mask_with_every_class()

    metrics.add_batch(target, target, level_idx=level_idx)

    assert metrics.mean_intersection_over_union(level_idx=level_idx) == 1.0
    assert metrics.pixel_accuracy(level_idx=level_idx) == 1.0
    assert metrics.pixel_accuracy_class(level_idx=level_idx) == 1.0


def test_metrics_match_a_hand_computed_confusion_matrix(metrics):
    """Pixel accuracy and per-class IoU agree with a case worked out by hand."""
    # 4 pixels at level 0 (background / body):
    #   gt   = [0, 0, 0, 1]
    #   pred = [0, 0, 1, 1]
    # class 0: intersection 2, union 3 -> 2/3
    # class 1: intersection 1, union 2 -> 1/2
    ground_truth = np.array([[0, 0, 0, 1]])
    prediction = np.array([[0, 0, 1, 1]])

    metrics.add_batch(ground_truth, prediction, level_idx=LEVEL_0)

    assert metrics.pixel_accuracy(level_idx=LEVEL_0) == pytest.approx(3 / 4)

    iou_without_background = metrics.intersection_over_union_per_class(level_idx=LEVEL_0)
    assert iou_without_background == pytest.approx([0.5])

    iou_with_background = metrics.intersection_over_union_per_class(
        level_idx=LEVEL_0, exclude_background=False
    )
    assert iou_with_background == pytest.approx([2 / 3, 0.5])

    # Class accuracy is recall, which is not the same thing as IoU: class 1 covers every pixel
    # it truly has (recall 1.0) but also claims one background pixel, which is what drags its
    # IoU down to 0.5.
    assert metrics.pixel_accuracy_class(level_idx=LEVEL_0) == pytest.approx(1.0)
    assert metrics.pixel_accuracy_class(level_idx=LEVEL_0, exclude_background=False) == (
        pytest.approx((2 / 3 + 1.0) / 2)
    )


def test_classes_absent_from_the_confusion_matrix_score_zero(metrics):
    """A class that never appears is reported as 0 rather than dividing by zero."""
    empty = np.zeros((1, 4, 4), dtype=np.int64)

    metrics.add_batch(empty, empty, level_idx=LEVEL_2)

    per_class = metrics.intersection_over_union_per_class(level_idx=LEVEL_2)
    assert per_class.shape == (level_to_num_classes[LEVEL_2] - 1,)
    assert not np.isnan(per_class).any()
    assert per_class == pytest.approx(np.zeros_like(per_class))
    assert metrics.mean_intersection_over_union(level_idx=LEVEL_2) == 0.0


def test_background_only_prediction_scores_zero_for_body(metrics):
    """Predicting background everywhere gives the body class an IoU of 0."""
    target = build_mask_with_every_class()

    metrics.add_batch(target, np.zeros_like(target), level_idx=LEVEL_0)

    assert metrics.mean_intersection_over_union(level_idx=LEVEL_0) == 0.0
    # Half of the level-0 pixels are background in the target, and all of them are correct.
    assert metrics.pixel_accuracy(level_idx=LEVEL_0) == pytest.approx(0.5)


def test_confusion_matrix_accumulates_over_batches(metrics):
    """Metrics are computed over a whole split, so batches must accumulate."""
    target = build_mask_with_every_class()
    prediction = np.zeros_like(target)

    metrics.add_batch(target, prediction, level_idx=LEVEL_0)
    first = metrics.confusion_matrix_by_level[LEVEL_0].copy()
    assert first.sum() == level_to_num_classes[LEVEL_0]

    metrics.add_batch(target, prediction, level_idx=LEVEL_0)

    accumulated = metrics.confusion_matrix_by_level[LEVEL_0]
    assert accumulated.sum() == first.sum() * 2
    assert accumulated == pytest.approx(first * 2)
    # Repeating the same batch cannot change the ratios it is built from.
    assert metrics.mean_intersection_over_union(level_idx=LEVEL_0) == 0.0


def test_reset_clears_every_level(metrics):
    """reset() zeroes the confusion matrices of all three levels."""
    target = build_mask_with_every_class()
    for level_idx in level_to_num_classes:
        metrics.add_batch(target, target, level_idx=level_idx)

    metrics.reset()

    for level_idx, num_classes in level_to_num_classes.items():
        confusion_matrix = metrics.confusion_matrix_by_level[level_idx]
        assert confusion_matrix.shape == (num_classes, num_classes)
        assert confusion_matrix.sum() == 0


def test_mismatched_mask_shapes_are_rejected(metrics):
    """Comparing masks of different shapes is a bug in the caller, not a metric."""
    with pytest.raises(AssertionError):
        metrics.add_batch(
            np.zeros((1, 4, 4), dtype=np.int64), np.zeros((1, 4, 5), dtype=np.int64), level_idx=0
        )
