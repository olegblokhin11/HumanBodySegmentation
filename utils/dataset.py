import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import utils.custom_transforms as tr
from utils.data_utils import level_0, level_1, level_2


class PascalPartDataset(Dataset):
    """
    A dataset class for loading and processing the Pascal Part dataset.

    This class handles loading images and corresponding segmentation masks,
    applying transformations, and mapping segmentation masks to different levels
    of class hierarchy.
    """

    def __init__(self, config: Dict, mode: str = "train") -> None:
        """
        Initialize the PascalPartDataset with configurations and mode.

        Args:
            config (Dict): Configuration dictionary containing dataset paths and image processing settings.
            mode (str): The mode for the dataset, either 'train' or 'val'. Default is 'train'.
        """
        assert mode in ["train", "val"], "Invalid mode. Use 'train' or 'val'."

        self.image_dir = os.path.join(config["dataset"]["dataset_path"], "JPEGImages")
        self.mask_dir = os.path.join(config["dataset"]["dataset_path"], "gt_masks")

        self.base_size = config["image"]["base_size"]
        self.crop_size = config["image"]["crop_size"]

        scale_factor = (
            config["image"]["scale_factor"]["min_coef"],
            config["image"]["scale_factor"]["max_coef"],
        )
        brightness = config["image"]["brightness"]
        contrast = config["image"]["contrast"]
        saturation = config["image"]["saturation"]
        hue = config["image"]["hue"]
        degrees = config["image"]["rot_degrees"]

        self.transforms = PascalPartDataset.get_transform(
            mode=mode,
            scale_factor=scale_factor,
            base_size=self.base_size,
            crop_size=self.crop_size,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            degrees=degrees,
        )

        samples_list_name = "train_id.txt" if mode == "train" else "val_id.txt"
        samples_list = os.path.join(
            config["dataset"]["dataset_path"], samples_list_name
        )

        self.image_paths, self.mask_paths = self.load_image_and_mask_paths(samples_list)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an image and its corresponding segmentation masks by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed image and three levels of segmentation masks.
        """
        image, mask = self.make_image_and_mask_pair(idx)
        image, mask = self.transforms(image, mask)

        # Convert the mask to different levels
        mask_np = np.array(mask).astype(np.int64)
        mask_level_0 = np.vectorize(level_0.get)(mask_np)
        mask_level_1 = np.vectorize(level_1.get)(mask_np)
        mask_level_2 = np.vectorize(level_2.get)(mask_np)

        # Convert masks to tensors
        mask_level_0 = torch.from_numpy(mask_level_0).long()
        mask_level_1 = torch.from_numpy(mask_level_1).long()
        mask_level_2 = torch.from_numpy(mask_level_2).long()

        return {
            "image": image,
            "mask_level_0": mask_level_0,
            "mask_level_1": mask_level_1,
            "mask_level_2": mask_level_2,
        }

    def make_image_and_mask_pair(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        Read an image and its corresponding segmentation mask from disk.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Image.Image, Image.Image]: The image and its mask.
        """
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = np.load(self.mask_paths[index])
        mask = Image.fromarray(mask.astype(np.uint8))
        return image, mask

    def load_image_and_mask_paths(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load the paths for images and masks based on a sample list.

        Args:
            file_path (str): Path to the file containing sample IDs.

        Returns:
            Tuple[List[str], List[str]]: Lists of image and mask file paths.
        """
        with open(file_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]
        image_paths = [
            os.path.join(self.image_dir, name + ".jpg") for name in image_names
        ]
        mask_paths = [
            os.path.join(self.mask_dir, name + ".npy") for name in image_names
        ]
        return image_paths, mask_paths

    @staticmethod
    def get_transform(
        mode: str = "train",
        scale_factor: Tuple[float, float] = (0.5, 2.0),
        base_size: int = 512,
        crop_size: int = 512,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
        degrees: int = 15,
    ) -> v2.Compose:
        """
        Get the transformation pipeline for the dataset.

        Args:
            mode (str): The mode for the dataset, either 'train' or 'val'.
            scale_factor (Tuple[float, float]): Scale factor for random scaling image (min_coef, max_coef).
            base_size (int): Base size for image resizing.
            crop_size (int): Crop size for image cropping.

        Returns:
            v2.Compose: A transformation pipeline.
        """
        assert mode in ["train", "val"], "Invalid mode. Use 'train' or 'val'."

        if mode == "train":
            transform = v2.Compose(
                [
                    tr.RandomHorizontalFlip(),
                    tr.RandomScaleCrop(
                        scale_factor=scale_factor,
                        base_size=base_size,
                        crop_size=crop_size,
                    ),
                    tr.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                    ),
                    tr.RandomRotation(degrees=degrees),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )
        else:  # mode == "val"
            transform = v2.Compose(
                [
                    tr.FixScaleCrop(crop_size=crop_size),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )

        return transform

    @staticmethod
    def preprocess(
        image: Image.Image, mask: Image.Image, crop_size: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess an image and mask with a fixed transformation pipeline.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.
            crop_size (int): Crop size for image preprocessing.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed image and mask tensors.
        """
        preprocess_transforms = v2.Compose(
            [
                tr.FixScaleCrop(crop_size=crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return preprocess_transforms(image, mask)


def initialize_data_loader(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Initialize data loaders for training and validation datasets.

    Args:
        config (Dict): Configuration dictionary containing dataset paths and image processing settings.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_dataset = PascalPartDataset(config, mode="train")
    val_dataset = PascalPartDataset(config, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batched data for the DataLoader.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of sample dictionaries.

    Returns:
        Dict[str, torch.Tensor]: Batched data as a dictionary.
    """
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "mask_level_0": torch.stack([x["mask_level_0"] for x in batch]),
        "mask_level_1": torch.stack([x["mask_level_1"] for x in batch]),
        "mask_level_2": torch.stack([x["mask_level_2"] for x in batch]),
    }
