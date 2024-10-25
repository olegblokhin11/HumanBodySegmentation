import os
import shutil
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import torch


class Saver:
    """
    A class for managing model checkpoints during training, including saving
    and organizing experiments.

    Attributes:
        config (Dict): The configuration dictionary for training parameters.
        checkpoint_dir (str): Directory path where checkpoints for the current experiment will be saved.
    """

    def __init__(self, config: Dict, experiment_name: str = None) -> None:
        """
        Initialize the Saver with a specified configuration and experiment name.

        Args:
            config (Dict): Configuration dictionary containing training parameters.
            experiment_name (str): Optional name to distinguish this experiment.
                                   If not provided, a timestamp will be used.
        """
        self.config = config
        base_dir = config["training"]["checkpoint_dir"]

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Set up the experiment directory with a unique name
        self.checkpoint_dir = self._get_unique_dir_name(base_dir, experiment_name)
        os.makedirs(self.checkpoint_dir)

    def _get_unique_dir_name(self, base_dir: str, experiment_name: str) -> str:
        """
        Generate a unique directory name by adding a suffix if the directory already exists.

        Args:
            base_dir (str): Base directory path.
            experiment_name (str): Desired experiment name.

        Returns:
            str: Unique directory name with suffix if needed.
        """
        checkpoint_dir = os.path.join(base_dir, experiment_name)
        suffix = 1

        # Add suffix if the directory already exists
        while os.path.exists(checkpoint_dir):
            checkpoint_dir = os.path.join(base_dir, f"{experiment_name}_{suffix}")
            suffix += 1

        return checkpoint_dir

    def save_checkpoint(
        self, state: Dict, is_best: bool, filename: str = "checkpoint.pth.tar"
    ) -> None:
        """
        Saves a checkpoint to disk.

        Args:
            state (Dict): A dictionary containing the model state, optimizer state, epoch number, etc.
            is_best (bool): Boolean flag indicating if this is the best-performing checkpoint.
            filename (str): The filename for the checkpoint.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, checkpoint_path)

        if is_best:
            best_pred = state["best_pred"]
            best_pred_path = os.path.join(self.checkpoint_dir, "best_pred.txt")
            with open(best_pred_path, "w") as f:
                f.write(str(best_pred))

            # Save a separate copy of the best model
            best_model_path = os.path.join(self.checkpoint_dir, "model_best.pth.tar")
            shutil.copyfile(checkpoint_path, best_model_path)

    def save_experiment_config(self) -> None:
        """
        Save the experiment configuration parameters to a file for reference.
        """
        config_file = os.path.join(self.checkpoint_dir, "parameters.txt")
        with open(config_file, "w") as log_file:
            config_params = OrderedDict()
            # Dataset Information
            config_params["dataset_path"] = self.config["dataset"]["dataset_path"]
            config_params["dataset_name"] = self.config["dataset"]["dataset_name"]
            config_params["base_size"] = self.config["image"]["base_size"]
            config_params["crop_size"] = self.config["image"]["crop_size"]

            # Dataset Aug Parameters
            config_params["scale_factor"] = self.config["image"]["scale_factor"]
            config_params["brightness"] = self.config["image"]["brightness"]
            config_params["contrast"] = self.config["image"]["contrast"]
            config_params["saturation"] = self.config["image"]["saturation"]
            config_params["hue"] = self.config["image"]["hue"]
            config_params["rot_degrees"] = self.config["image"]["rot_degrees"]

            # Network Information
            config_params["backbone"] = self.config["network"]["backbone"]

            # Training Information
            config_params["workers"] = self.config["training"]["workers"]
            config_params["epochs"] = self.config["training"]["epochs"]
            config_params["start_epoch"] = self.config["training"]["start_epoch"]
            config_params["batch_size"] = self.config["training"]["batch_size"]
            config_params["learning_rate"] = self.config["training"]["lr"]
            config_params["lr_scheduler"] = self.config["training"]["lr_scheduler"]

            config_params["momentum"] = self.config["training"]["momentum"]
            config_params["weight_decay"] = self.config["training"]["weight_decay"]
            config_params["val_interval"] = self.config["training"]["val_interval"]
            config_params["tensorboard_log_dir"] = self.config["training"][
                "tensorboard"
            ]["log_dir"]

            # Optional Attributes (Commented out fields or optional attributes)
            if "weights_initialization" in self.config["training"]:
                config_params["use_pretrained_weights"] = self.config["training"][
                    "weights_initialization"
                ].get("use_pretrained_weights", False)
                config_params["restore_from"] = self.config["training"][
                    "weights_initialization"
                ].get("restore_from", "None")

            # Save to the parameters file
            for key, val in config_params.items():
                log_file.write(f"{key}: {val}\n")
