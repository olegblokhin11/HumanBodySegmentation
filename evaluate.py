import os

import torch
from tqdm import tqdm

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from utils.cli import add_checkpoint_arg, build_parser, load_config, resolve_checkpoint
from utils.data_utils import (
    hierarchy,
    hierarchy_level_0,
    hierarchy_level_1,
    level_str_to_level_idx,
    level_to_num_classes,
)
from utils.dataset import initialize_data_loader
from utils.device import get_device
from utils.metrics import SegmentationMetrics

LEVEL_NAMES = {0: hierarchy_level_0, 1: hierarchy_level_1, 2: hierarchy}


class Tester:
    """
    The Tester class encapsulates the model evaluation pipeline including loading the model,
    preparing the data, and computing metrics.

    Attributes:
        config (dict): Configuration dictionary containing test parameters.
        device (str): The device to be used for inference (e.g., 'cuda', 'cpu').
        model (HierarchicalDeepLabV3): The hierarchical segmentation model.
        val_loader (DataLoader): DataLoader for validation data.
        metrics (SegmentationMetrics): Evaluation metrics for the segmentation task.
    """

    def __init__(self, config, checkpoint_path):
        """
        Initialize the Tester class with the provided configuration.

        Args:
            config (dict): Configuration dictionary.
            checkpoint_path (str): Path to the model checkpoint file.
        """
        self.config = config
        self.device = get_device()
        self.model = self._initialize_model(checkpoint_path)
        self.val_loader = self._initialize_dataloader()
        self.metrics = SegmentationMetrics()
        self.num_mask_levels = len(level_to_num_classes)

    def _initialize_model(self, checkpoint_path):
        """
        Initialize the segmentation model and load weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the model checkpoint file.

        Returns:
            HierarchicalDeepLabV3: The loaded model.
        """
        model = HierarchicalDeepLabV3(
            num_classes_level_0=level_to_num_classes[0],
            num_classes_level_1=level_to_num_classes[1],
            num_classes_level_2=level_to_num_classes[2],
            backbone=self.config["network"]["backbone"],
            # Weights come from the checkpoint, so skip downloading pretrained ones.
            pretrained=False,
        )
        model.to(self.device)

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from '{checkpoint_path}'")
            # weights_only=False is required: checkpoints carry optimizer state and
            # non-tensor metadata on top of the model weights.
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"Successfully loaded checkpoint '{checkpoint_path}'")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        model.eval()
        return model

    def _initialize_dataloader(self):
        """
        Initialize the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        _, val_loader = initialize_data_loader(self.config)
        return val_loader

    def run_evaluation(self):
        """
        Run the evaluation on the validation dataset and compute metrics.
        """
        self.metrics.reset()
        test_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        tbar = tqdm(self.val_loader, desc="\r")
        with torch.no_grad():
            for _i, samples in enumerate(tbar):
                images = samples["image"].to(self.device)
                masks = {
                    level: samples[level].to(self.device)
                    for level in ["mask_level_0", "mask_level_1", "mask_level_2"]
                }

                outputs = self.model(images)
                loss = sum([criterion(outputs[level], masks[level]) for level in masks]) / len(
                    masks
                )
                test_loss += loss.item()

                # Add batch sample into metrics
                for level in masks:
                    gt_mask = masks[level].cpu().numpy()
                    pred_mask = outputs[level].argmax(dim=1).cpu().numpy()
                    level_idx = level_str_to_level_idx[level]
                    self.metrics.add_batch(gt_mask, pred_mask, level_idx)

        avg_miou_over_level = 0.0
        for level_idx in range(self.num_mask_levels):
            acc = self.metrics.pixel_accuracy(level_idx=level_idx)
            acc_class = self.metrics.pixel_accuracy_class(level_idx=level_idx)
            miou = self.metrics.mean_intersection_over_union(level_idx=level_idx)

            print(
                f"Level [{level_idx}] - Acc: {acc:.3f}, "
                f"Acc_class: {acc_class:.3f}, mIoU: {miou:.3f}"
            )
            avg_miou_over_level += miou

        print(f"Average Loss: {test_loss / len(self.val_loader):.3f}")
        print(f"Average mIoU across all levels: {avg_miou_over_level / self.num_mask_levels:.3f}")

        self.print_per_class_iou()

    def print_per_class_iou(self) -> None:
        """
        Print the IoU of every individual class, grouped by hierarchy level.

        Background (class 0) is omitted, matching the metric definition in the task.
        """
        print("\nPer-class IoU (background excluded):")
        for level_idx in range(self.num_mask_levels):
            level_names = LEVEL_NAMES[level_idx]
            iou_per_class = self.metrics.intersection_over_union_per_class(level_idx=level_idx)

            print(f"  Level [{level_idx}]:")
            for offset, iou in enumerate(iou_per_class):
                class_idx = offset + 1
                class_name = level_names.get(class_idx, f"class_{class_idx}")
                print(f"    {class_name:<12} {iou:.3f}")


def main():
    """
    Main function to start the evaluation process.
    """
    parser = add_checkpoint_arg(build_parser("Evaluate a trained hierarchical segmentation model."))
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    print(f"Evaluating checkpoint: {checkpoint_path}")

    tester = Tester(config, checkpoint_path)
    tester.run_evaluation()


if __name__ == "__main__":
    main()
