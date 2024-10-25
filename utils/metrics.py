import numpy as np

from utils.data_utils import level_to_num_classes


class SegmentationMetrics:
    """
    A class for calculating and managing segmentation performance metrics across
    different levels of detail.

    Provides metrics such as pixel accuracy and mean Intersection over Union (mIoU).
    The class maintains confusion matrices for each level and updates them based on predictions.

    Attributes:
        confusion_matrix_by_level (Dict[int, np.ndarray]): A dictionary storing
            confusion matrices for each level.
    """

    def __init__(self) -> None:
        """
        Initialize the SegmentationMetrics with the required number of classes for each level.
        """
        self.confusion_matrix_by_level = {
            level_idx: np.zeros((num,) * 2)
            for level_idx, num in level_to_num_classes.items()
        }

    def pixel_accuracy(self, level_idx: int) -> float:
        """
        Calculate pixel accuracy for a given level.

        Args:
            level_idx (int): The index of the level to calculate accuracy for.

        Returns:
            float: Pixel accuracy as the ratio of correctly predicted pixels to the total pixels.
        """
        confusion_matrix = self.confusion_matrix_by_level[level_idx]
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self, level_idx: int, exclude_background=True) -> float:
        """
        Calculate class-wise pixel accuracy for a given level, optionally excluding the background class.

        Args:
            level_idx (int): The index of the level to calculate accuracy for.

        Returns:
            float: Mean pixel accuracy across all classes, optionally excluding background.
        """
        confusion_matrix = self.confusion_matrix_by_level[level_idx]

        # Set index range based on whether to exclude background
        start_index = 1 if exclude_background else 0

        # Start from 1 to skip the background class
        acc_per_class = (
            np.diag(confusion_matrix)[start_index:]
            / confusion_matrix.sum(axis=1)[start_index:]
        )
        acc = np.nanmean(acc_per_class)

        return acc

    def mean_intersection_over_union(
        self, level_idx: int, exclude_background=True
    ) -> float:
        """
        Calculate the mean Intersection over Union (mIoU) for a given level, optionally excluding the background class.

        Args:
            level_idx (int): The index of the level to calculate mIoU for.
            exclude_background (bool): Whether to exclude the background class (assumed to be class 0).

        Returns:
            float: The mean IoU score across all classes, optionally excluding background.
        """
        confusion_matrix = self.confusion_matrix_by_level[level_idx]

        # Set index range based on whether to exclude background
        start_index = 1 if exclude_background else 0

        # Exclude background (class index 0)
        intersection = np.diag(confusion_matrix)[start_index:]
        ground_truth = np.sum(confusion_matrix, axis=1)[start_index:]
        predicted = np.sum(confusion_matrix, axis=0)[start_index:]
        union = ground_truth + predicted - intersection

        # Calculate IoU for each class and ignore division by zero
        iou_per_class = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=(union != 0),
        )

        # Return the mean IoU, ignoring NaNs
        miou = np.nanmean(iou_per_class)
        return miou

    def _generate_matrix(
        self, gt_mask: np.ndarray, pred_mask: np.ndarray, level_idx: int
    ) -> np.ndarray:
        """
        Generate a confusion matrix for a batch of ground truth and predicted masks.

        Args:
            gt_mask (np.ndarray): Ground truth masks.
            pred_mask (np.ndarray): Predicted masks.
            level_idx (int): The index of the level to generate the matrix for.

        Returns:
            np.ndarray: The confusion matrix for the given level.
        """
        num_class = level_to_num_classes[level_idx]

        mask = (
            (gt_mask >= 0)
            & (gt_mask < num_class)
            & (pred_mask >= 0)
            & (pred_mask < num_class)
        )

        label = num_class * gt_mask[mask].astype("int") + pred_mask[mask]
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)
        return confusion_matrix

    def add_batch(
        self, gt_mask: np.ndarray, pred_mask: np.ndarray, level_idx: int
    ) -> None:
        """
        Update the confusion matrix for a given level using a batch of masks.

        Args:
            gt_mask (np.ndarray): Ground truth masks.
            pred_mask (np.ndarray): Predicted masks.
            level_idx (int): The index of the level to update the matrix for.
        """
        assert (
            gt_mask.shape == pred_mask.shape
        ), "Shape mismatch between ground truth and predictions."

        self.confusion_matrix_by_level[level_idx] += self._generate_matrix(
            gt_mask, pred_mask, level_idx=level_idx
        )

    def reset(self) -> None:
        """
        Reset the confusion matrices for all levels.
        """
        for level_idx in level_to_num_classes.keys():
            num_class = level_to_num_classes[level_idx]
            self.confusion_matrix_by_level[level_idx] = np.zeros((num_class,) * 2)
