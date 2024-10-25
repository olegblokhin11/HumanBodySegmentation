import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2


class RandomHorizontalFlip:
    """
    Randomly flip the image and mask horizontally with a probability of 0.5.
    """

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Flip the image and mask horizontally with a probability of 0.5.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Flipped image and mask, or original if not flipped.
        """
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return image, mask


class RandomScaleCrop:
    """
    Randomly scale the image and mask, then crop to a fixed size.
    """

    def __init__(
        self,
        scale_factor: Tuple[float, float],
        base_size: int,
        crop_size: int,
        fill: int = 0,
    ) -> None:
        """
        Initialize RandomScaleCrop.

        Args:
            scale_factor (Tuple[float, float]): Scale factor for random scaling (min_coef, max_coef).
            base_size (int): Base size for random scaling.
            crop_size (int): Target size for cropping.
            fill (int): Fill value for padding the mask (default: 0).
        """
        self.scale_factor = scale_factor
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply random scaling and cropping to the image and mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Scaled and cropped image and mask.
        """
        # Random scale within the specified factor range (short edge scaling)
        scale = random.uniform(self.scale_factor[0], self.scale_factor[1])
        short_size = int(self.base_size * scale)

        # Resize based on the shorter edge
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Pad if needed
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        # Random crop to crop_size
        w, h = image.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return image, mask


class FixScaleCrop:
    """
    Scale the image and mask to a fixed size, then center crop.
    """

    def __init__(self, crop_size: int) -> None:
        """
        Initialize FixScaleCrop.

        Args:
            crop_size (int): Target size for cropping.
        """
        self.crop_size = crop_size

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply fixed scaling and center cropping to the image and mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Scaled and cropped image and mask.
        """
        w, h = image.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Center crop
        w, h = image.size
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0))
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return image, mask


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation, and hue of an image, while leaving the mask unchanged.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ) -> None:
        """
        Initialize ColorJitter using torchvision's ColorJitter.

        Args:
            brightness (float): How much to jitter brightness.
                                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
            contrast (float): How much to jitter contrast.
                              contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
            saturation (float): How much to jitter saturation.
                                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
            hue (float): How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue].
        """
        self.color_jitter = v2.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply color jitter to the image and return both the transformed image and the original mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Color-jittered image and unchanged mask.
        """
        # Apply color jitter from torchvision to the image only
        image = self.color_jitter(image)
        return image, mask


class RandomRotation:
    """
    Randomly rotate the image and mask by a degree chosen from a specified range.
    """

    def __init__(self, degrees: int = 15) -> None:
        """
        Initialize RandomRotation.

        Args:
            degrees (int): Range of degrees to select from [-degrees, degrees].
                           The rotation angle will be randomly selected from this range.
        """
        self.degrees = degrees

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply random rotation to the image and mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Rotated image and mask.
        """
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        mask = mask.rotate(angle, resample=Image.NEAREST, expand=False)
        return image, mask


class Normalize:
    """
    Normalize a tensor image with mean and standard deviation.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Initialize Normalize.

        Args:
            mean (Tuple[float, float, float]): Means for each channel.
            std (Tuple[float, float, float]): Standard deviations for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the image with mean and standard deviation.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The corresponding mask.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalized image and unchanged mask.
        """
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)
        image -= self.mean
        image /= self.std

        return image, mask


class ToTensor:
    """
    Convert PIL images or numpy arrays in the sample to Tensors.
    """

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert image and mask to tensor format.

        Args:
            image (np.ndarray): The input image array.
            mask (np.ndarray): The corresponding mask array.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and mask as tensors.
        """
        # Convert image: H x W x C (PIL) -> C x H x W (Tensor)
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask
