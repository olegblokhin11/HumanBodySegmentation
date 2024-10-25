import os
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.data_utils import decode_segmap_sequence


class TensorboardSummary:
    """
    A class for handling TensorBoard logging, including visualization of images,
    predictions, and ground truth labels during training.

    Attributes:
        directory (str): Directory path where the logs will be saved.
    """

    def __init__(self, directory: str) -> None:
        """
        Initialize the TensorboardSummary with a specified directory.

        Args:
            directory (str): Directory path to save TensorBoard logs.
        """
        self.directory = directory

    def create_summary(self, experiment_name: str = None) -> SummaryWriter:
        """
        Create a TensorBoard SummaryWriter instance with a unique subdirectory.

        Args:
            experiment_name (str): Optional name to distinguish this experiment.
                                   If not provided, a timestamp will be used.

        Returns:
            SummaryWriter: TensorBoard writer object.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = self._get_unique_dir_name(self.directory, experiment_name)
        writer = SummaryWriter(log_dir=log_dir)
        return writer

    def visualize_image(
        self,
        writer: SummaryWriter,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor,
        n_classes: int,
        global_step: int,
    ) -> None:
        """
        Visualize input images, predicted labels, and ground truth labels in a single grid.

        Args:
            writer (SummaryWriter): TensorBoard writer object.
            image (torch.Tensor): Batch of input images.
            gt_mask (torch.Tensor): Batch of ground truth masks.
            pred_mask (torch.Tensor): Batch of predicted masks.
            n_classes (int): Number of classes.
            global_step (int): The current step of training, used for logging.
        """
        # Prepare grid of input images
        grid_input = make_grid(image[:3].clone().cpu().data, 3, normalize=True)

        # Prepare grid of ground truth masks
        gt_masks = torch.squeeze(gt_mask[:3], 1).detach().cpu().numpy()
        grid_gt = make_grid(
            decode_segmap_sequence(gt_masks, n_classes=n_classes),
            3,
            normalize=False,
            value_range=(0, 255),
        )

        # Prepare grids of predicted masks
        pred_masks = torch.max(pred_mask[:3], 1)[1].detach().cpu().numpy()
        grid_pred = make_grid(
            decode_segmap_sequence(pred_masks, n_classes=n_classes),
            3,
            normalize=False,
            value_range=(0, 255),
        )

        # Stack all images (input, predicted, and ground truth) vertically
        combined_grid = torch.cat((grid_input, grid_gt, grid_pred), dim=1)

        # Write a single image grid to TensorBoard
        writer.add_image(
            "Visualization (Image / Gt / Pred)", combined_grid, global_step
        )

    def _get_unique_dir_name(self, base_dir: str, experiment_name: str) -> str:
        """
        Generate a unique directory name by adding a suffix if the directory already exists.

        Args:
            base_dir (str): Base directory path.
            experiment_name (str): Desired experiment name.

        Returns:
            str: Unique directory name with suffix if needed.
        """
        log_dir = os.path.join(base_dir, experiment_name)
        suffix = 1

        # Add suffix if the directory already exists
        while os.path.exists(log_dir):
            log_dir = os.path.join(base_dir, f"{experiment_name}_{suffix}")
            suffix += 1

        return log_dir
