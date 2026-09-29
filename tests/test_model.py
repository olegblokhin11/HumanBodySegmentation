"""Tests for the hierarchical DeepLabV3 model: output shapes and construction."""

import pytest
import torch

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from utils.data_utils import level_to_num_classes

BACKBONES = ["resnet50", "resnet101", "mobilenet"]


def build_model(backbone: str = "resnet50") -> HierarchicalDeepLabV3:
    """
    Build a model with random weights.

    pretrained=False keeps the tests offline and fast; the checkpoint is what actually
    supplies weights in real use.
    """
    return HierarchicalDeepLabV3(
        num_classes_level_0=level_to_num_classes[0],
        num_classes_level_1=level_to_num_classes[1],
        num_classes_level_2=level_to_num_classes[2],
        backbone=backbone,
        pretrained=False,
    ).eval()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_returns_logits_for_every_level(backbone):
    """One forward pass produces three logit maps, one per hierarchy level."""
    model = build_model(backbone)
    inputs = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(inputs)

    assert set(outputs) == {"mask_level_0", "mask_level_1", "mask_level_2"}
    for level_idx, key in enumerate(["mask_level_0", "mask_level_1", "mask_level_2"]):
        expected = (1, level_to_num_classes[level_idx], 64, 64)
        assert tuple(outputs[key].shape) == expected


def test_upsampling_follows_the_input_resolution():
    """Non-square inputs must come back at their own height and width, not transposed."""
    model = build_model("resnet50")
    inputs = torch.randn(2, 3, 64, 96)

    with torch.no_grad():
        outputs = model(inputs)

    # Height and width are different on purpose: swapping them would still produce a
    # plausible-looking tensor of the wrong shape.
    for level_idx, key in enumerate(["mask_level_0", "mask_level_1", "mask_level_2"]):
        assert tuple(outputs[key].shape) == (2, level_to_num_classes[level_idx], 64, 96)


def test_unknown_backbone_is_rejected():
    """A typo in the config should fail immediately instead of building a wrong model."""
    with pytest.raises(AssertionError):
        build_model("resnet18")


def test_auxiliary_classifier_exists_without_pretrained_weights():
    """
    The auxiliary head is part of the module structure in every configuration.

    torchvision only creates it when pretrained weights are requested, but checkpoints trained
    that way contain its parameters. Building with pretrained=False must therefore still accept
    such a checkpoint, which is exactly what loading one relies on.
    """
    model = build_model("resnet50")

    assert hasattr(model.base_model, "aux_classifier")
    assert model.base_model.aux_classifier is not None
