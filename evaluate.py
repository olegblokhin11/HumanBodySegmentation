import os

import torch
import yaml
from tqdm import tqdm

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from train import get_device
from utils.data_utils import level_str_to_level_idx, level_to_num_classes
from utils.dataset import initialize_data_loader
from utils.metrics import SegmentationMetrics


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
        )
        model.to(self.device)

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
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
            for i, samples in enumerate(tbar):
                images = samples["image"].to(self.device)
                masks = {
                    level: samples[level].to(self.device)
                    for level in ["mask_level_0", "mask_level_1", "mask_level_2"]
                }

                outputs = self.model(images)
                loss = sum(
                    [criterion(outputs[level], masks[level]) for level in masks]
                ) / len(masks)
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
                f"Level [{level_idx}] - Acc: {acc:.3f}, Acc_class: {acc_class:.3f}, mIoU: {miou:.3f}"
            )
            avg_miou_over_level += miou

        print(f"Average Loss: {test_loss / len(self.val_loader):.3f}")
        print(
            f"Average mIoU across all levels: {avg_miou_over_level / self.num_mask_levels:.3f}"
        )


def main():
    """
    Main function to start the evaluation process.
    """
    # TODO: write parsing from args

    config_path = "./configs/baseline_heavy.yml"
    checkpoint_path = "./checkpoints/deeplab_resnet50_lr_0.001_batch_4_2/checkpoint_best.pth.tar"  # Update this path as needed

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tester = Tester(config, checkpoint_path)
    tester.run_evaluation()


if __name__ == "__main__":
    main()
