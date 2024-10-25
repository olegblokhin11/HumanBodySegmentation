from typing import Dict, List

import numpy as np
import torch

# Semantic segmentation hierarchy dictionary
hierarchy: Dict[int, str] = {
    0: "background",
    1: "low_hand",
    2: "torso",
    3: "low_leg",
    4: "head",
    5: "up_leg",
    6: "up_hand",
}

hierarchy_level_0: Dict[int, str] = {0: "background", 1: "body"}

hierarchy_level_1: Dict[int, str] = {0: "background", 1: "upper_body", 2: "lower_body"}

# Hierarchy labels mapping
level_0: Dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
level_1: Dict[int, int] = {0: 0, 1: 1, 2: 1, 4: 1, 6: 1, 3: 2, 5: 2}
level_2: Dict[int, int] = {k: k for k in hierarchy.keys()}

# Number of classes at each hierarchical level
level_to_num_classes: Dict[int, int] = {0: 2, 1: 3, 2: 7}

# Mapping from level string to index
level_str_to_level_idx: Dict[str, int] = {
    "mask_level_0": 0,
    "mask_level_1": 1,
    "mask_level_2": 2,
}


def decode_segmap_sequence(
    label_masks: List[np.ndarray], n_classes: int
) -> torch.Tensor:
    """
    Decode a sequence of segmentation masks into RGB format.

    Args:
        label_masks (List[np.ndarray]): List of arrays of class labels.
        n_classes (int): Number of classes.

    Returns:
        torch.Tensor: RGB tensor of shape (B, C, H, W) where B is batch size, C is number of channels.
    """
    rgb_masks = [decode_segmap(label_mask, n_classes) for label_mask in label_masks]
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Decode a single segmentation mask into an RGB image.

    Args:
        label_mask (np.ndarray): An array of class labels.
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: The decoded RGB image of shape (M, N, 3).
    """
    label_colours = get_color_map(n_classes)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb


def encode_segmap(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Encode an RGB segmentation mask into class labels.

    Args:
        mask (np.ndarray): Raw RGB mask of shape (M, N, 3).
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Class map of shape (M, N) where each value is the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_color_map(n_classes)):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    return label_mask.astype(int)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image by reversing the normalization process.

    Args:
        image (torch.Tensor): A normalized image tensor.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = image * torch.tensor(std).view(-1, 1, 1)
    image = image + torch.tensor(mean).view(-1, 1, 1)
    return image


def get_color_map(n_classes: int) -> np.ndarray:
    """
    Generate a color map for visualizing segmentation masks.

    Args:
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: Array of shape (n_classes, 3) where each row is an RGB color.
    """
    colors = np.asarray(
        [
            [0, 0, 0],  # background
            [128, 0, 0],  # low_hand
            [0, 128, 0],  # torso
            [128, 128, 0],  # low_leg
            [0, 0, 128],  # head
            [128, 0, 128],  # up_leg
            [0, 128, 128],  # up_hand
        ]
    )
    return colors[:n_classes]
