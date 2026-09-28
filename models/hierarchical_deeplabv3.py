import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)


class HierarchicalDeepLabV3(nn.Module):
    """
    A hierarchical DeepLabV3 model for multi-level semantic segmentation.

    This model generates segmentation masks at three different hierarchical levels using a shared
    backbone.
    """

    def __init__(
        self,
        num_classes_level_0: int,
        num_classes_level_1: int,
        num_classes_level_2: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the HierarchicalDeepLabV3 model.

        Args:
            num_classes_level_0 (int): Number of classes for level 0 segmentation.
            num_classes_level_1 (int): Number of classes for level 1 segmentation.
            num_classes_level_2 (int): Number of classes for level 2 segmentation.
            backbone (str): The backbone architecture to use ('resnet50', 'resnet101', 'mobilenet').
            Default is 'resnet50'.
            pretrained (bool): Whether to initialize the backbone with COCO-pretrained weights.
                Set to False when the weights are about to be overwritten by a checkpoint,
                which avoids an unnecessary download. Default is True.
        """
        super().__init__()

        # Validate the selected backbone
        assert backbone in [
            "resnet50",
            "resnet101",
            "mobilenet",
        ], "Invalid backbone name"

        # Initialize the base model based on the selected backbone.
        #
        # aux_loss is passed explicitly (torchvision would otherwise enable it only
        # when pretrained weights are requested). The auxiliary classifier is never
        # used by forward(), but it is part of the module structure, so keeping it
        # enabled in every configuration lets a checkpoint trained with pretrained
        # weights load into a model built with pretrained=False.
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.base_model = deeplabv3_resnet50(weights=weights, progress=True, aux_loss=True)
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            self.base_model = deeplabv3_resnet101(weights=weights, progress=True, aux_loss=True)
        elif backbone == "mobilenet":
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.base_model = deeplabv3_mobilenet_v3_large(
                weights=weights, progress=True, aux_loss=True
            )

        # Remove the last classifier layer from the base model
        self.base_model.classifier[-1] = nn.Identity()

        # Define custom classifiers for each hierarchical level
        self.classifier_level_0 = nn.Conv2d(256, num_classes_level_0, kernel_size=1)
        self.classifier_level_1 = nn.Conv2d(256, num_classes_level_1, kernel_size=1)
        self.classifier_level_2 = nn.Conv2d(256, num_classes_level_2, kernel_size=1)

    def forward(self, input: torch.Tensor) -> dict:
        """
        Forward pass through the HierarchicalDeepLabV3 model.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            dict: A dictionary containing segmentation masks for each hierarchical level:
                - 'mask_level_0': Segmentation output for level 0.
                - 'mask_level_1': Segmentation output for level 1.
                - 'mask_level_2': Segmentation output for level 2.
        """
        # Extract features using the backbone
        features = self.base_model.backbone(input)["out"]
        x = self.base_model.classifier(features)

        # Generate outputs for each hierarchical level
        out_level_0 = self.classifier_level_0(x)
        out_level_1 = self.classifier_level_1(x)
        out_level_2 = self.classifier_level_2(x)

        # Upsample the outputs to match the input size
        out_level_0 = F.interpolate(
            out_level_0, size=input.shape[2:], mode="bilinear", align_corners=True
        )
        out_level_1 = F.interpolate(
            out_level_1, size=input.shape[2:], mode="bilinear", align_corners=True
        )
        out_level_2 = F.interpolate(
            out_level_2, size=input.shape[2:], mode="bilinear", align_corners=True
        )

        return {
            "mask_level_0": out_level_0,
            "mask_level_1": out_level_1,
            "mask_level_2": out_level_2,
        }
