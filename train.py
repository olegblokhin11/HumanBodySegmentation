import torch
import yaml
from tqdm import tqdm

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from utils.data_utils import level_str_to_level_idx, level_to_num_classes
from utils.dataset import initialize_data_loader
from utils.lr_scheduler import get_lr_scheduler
from utils.metrics import SegmentationMetrics
from utils.saver import Saver
from utils.tensorboard_summary import TensorboardSummary


def get_device():
    """
    Determine the appropriate device for training (GPU, MPS, or CPU).

    Returns:
        str: The device to be used for training ('cuda', 'mps', or 'cpu').
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device


class Trainer:
    """
    The Trainer class encapsulates the entire training pipeline, including model initialization,
    data loading, training, validation, and saving model checkpoints.

    Attributes:
        config (dict): Configuration dictionary containing training parameters.
        device (str): The device to be used for training (e.g., 'cuda', 'cpu').
        best_pred (float): The best metric (avg mIoU) achieved during training.
        batch_size (int): The batch size for training.
        saver (Saver): Manages experiment saving and checkpointing.
        summary (TensorboardSummary): For visualizing training progress.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (HierarchicalDeepLabV3): The hierarchical segmentation model.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        criterion (torch.nn.Module): Loss function used during training.
        metrics (SegmentationMetrics): Evaluation metrics for the segmentation task.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    """

    def __init__(self, config):
        """
        Initialize the Trainer class with the provided configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        self.best_pred = 0.0
        self.batch_size = config["training"]["batch_size"]

        self._initialize_experiment()
        self._initialize_dataloaders()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_criterion()
        self._initialize_scheduler()

    def _initialize_experiment(self):
        """
        Set up the experiment environment, including directories for saving
        model checkpoints and TensorBoard summaries.
        """
        experiment_name = f"deeplab_{self.config['network']['backbone']}_lr_{self.config['training']['lr']}_batch_{self.config['training']['batch_size']}"
        self.saver = Saver(self.config, experiment_name)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(
            self.config["training"]["tensorboard"]["log_dir"]
        )
        self.writer = self.summary.create_summary(experiment_name)

    def _initialize_dataloaders(self):
        """
        Initialize the DataLoader objects for training and validation datasets.
        """
        self.train_loader, self.val_loader = initialize_data_loader(self.config)

    def _initialize_model(self):
        """
        Initialize the hierarchical segmentation model based on the provided backbone architecture.
        """
        self.model = HierarchicalDeepLabV3(
            num_classes_level_0=level_to_num_classes[0],
            num_classes_level_1=level_to_num_classes[1],
            num_classes_level_2=level_to_num_classes[2],
            backbone=self.config["network"]["backbone"],
        )
        self.model.to(self.device)

        # TODO: add load weights

    def _initialize_optimizer(self):
        """
        Initialize the optimizer for training the model using the SGD algorithm.
        """
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config["training"]["lr"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )

    def _initialize_criterion(self):
        """
        Initialize the loss function (CrossEntropyLoss) and segmentation metrics.
        """

        # TODO: check Focal Loss

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.metrics = SegmentationMetrics()

    def _initialize_scheduler(self):
        """
        Initialize the learning rate scheduler based on the training configuration.
        """
        self.num_iters_per_epoch = len(self.train_loader)
        self.scheduler = get_lr_scheduler(
            self.config, self.num_iters_per_epoch, self.optimizer
        )

    def _get_num_levels(self):
        """
        Get number of levels.
        """
        return len(level_to_num_classes.keys())

    def _get_current_lr(self):
        """
        Retrieve the current learning rate from the optimizer.

        Returns:
            float: Current learning rate.
        """
        return self.optimizer.param_groups[0]["lr"]

    def train_one_epoch(self, epoch):
        """
        Perform training for a single epoch.

        Args:
            epoch (int): The current epoch number.
        """
        # Print the current learning rate and best_pred
        current_lr = self._get_current_lr()
        print(
            f"[Epoch {epoch}] Learning Rate: {current_lr:.3e}, Best Prediction (mIoU): {self.best_pred:.4f}"
        )

        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)

        for i, samples in enumerate(tbar):
            images = samples["image"].to(self.device)
            masks = {
                level: samples[level].to(self.device)
                for level in ["mask_level_0", "mask_level_1", "mask_level_2"]
            }

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = sum(
                [self.criterion(outputs[level], masks[level]) for level in masks]
            )
            loss /= self.batch_size

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description(f"Train loss: {train_loss / (i + 1):.3f}")
            self.writer.add_scalar(
                "train/total_loss_iter", loss.item(), epoch * len(tbar) + i
            )

            # Log the current learning rate
            current_lr = self._get_current_lr()
            self.writer.add_scalar(
                "train/learning_rate", current_lr, epoch * len(tbar) + i
            )

            if i % (self.num_iters_per_epoch // 10) == 0:
                global_step = i + self.num_iters_per_epoch * epoch

                # Visualize only last mask level
                self.summary.visualize_image(
                    self.writer,
                    images,
                    masks["mask_level_2"],
                    outputs["mask_level_2"],
                    level_to_num_classes[2],
                    global_step,
                )

            self.scheduler.step()

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print(f"[Epoch {epoch}] Loss: {train_loss:.3f}")
        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
            is_best=False,
            filename="checkpoint_last.pth.tar",
        )

    def validate(self, epoch):
        """
        Perform validation for a single epoch and log the results.

        Args:
            epoch (int): The current epoch number.
        """

        self.model.eval()
        self.metrics.reset()
        test_loss = 0.0
        tbar = tqdm(self.val_loader, desc="\r")

        with torch.no_grad():
            for i, samples in enumerate(tbar):
                images = samples["image"].to(self.device)
                masks = {
                    level: samples[level].to(self.device)
                    for level in ["mask_level_0", "mask_level_1", "mask_level_2"]
                }
                outputs = self.model(images)
                loss = (
                    sum(
                        [
                            self.criterion(outputs[level], masks[level])
                            for level in masks
                        ]
                    )
                    / self.batch_size
                )
                test_loss += loss.item()
                tbar.set_description(f"Val loss: {test_loss / (i + 1):.3f}")

                for level in masks:
                    gt_mask = masks[level].cpu().numpy()
                    pred_mask = outputs[level].argmax(dim=1).cpu().numpy()
                    level_idx = level_str_to_level_idx[level]
                    self.metrics.add_batch(gt_mask, pred_mask, level_idx)

        self._log_validation_results(epoch, test_loss)

    def _log_validation_results(self, epoch, test_loss):
        """
        Log the validation results including mIoU for each level and checkpoint management.

        Args:
            epoch (int): The current epoch number.
            test_loss (float): Total loss accumulated during validation.
        """
        avg_miou_over_level = 0.0
        num_mask_levels = self._get_num_levels()
        for level_idx in range(num_mask_levels):
            # Compute metrics
            acc = self.metrics.pixel_accuracy(level_idx=level_idx)

            # Compute mIoU and Acc class with excluded background class (0)
            acc_class = self.metrics.pixel_accuracy_class(
                level_idx=level_idx, exclude_background=True
            )
            miou = self.metrics.mean_intersection_over_union(
                level_idx=level_idx, exclude_background=True
            )

            # Log metrics
            self.writer.add_scalar(f"val/mIoU [{level_idx}]", miou, epoch)
            self.writer.add_scalar(f"val/Acc [{level_idx}]", acc, epoch)
            self.writer.add_scalar(f"val/Acc_class [{level_idx}]", acc_class, epoch)

            print(
                f"Level [{level_idx}] Acc: {acc}, Acc_class: {acc_class}, mIoU: {miou}"
            )
            avg_miou_over_level += miou

        # Save the best model based on average mIoU
        new_pred = avg_miou_over_level / num_mask_levels
        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best=True,
                filename="checkpoint_best.pth.tar",
            )
        self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)


def main():
    """
    The main function to start the training process, including loading configurations,
    initializing the trainer, and managing the training loop.
    """
    # TODO: write parsing from args

    with open("./configs/baseline_heavy.yml") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    for epoch in range(config["training"]["start_epoch"], config["training"]["epochs"]):
        trainer.train_one_epoch(epoch)
        if epoch % config["training"]["val_interval"] == 0:
            trainer.validate(epoch)


if __name__ == "__main__":
    main()
