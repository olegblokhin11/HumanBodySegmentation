import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt
from PIL import Image

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from utils.cli import add_checkpoint_arg, build_parser, load_config, resolve_checkpoint
from utils.data_utils import (
    decode_segmap,
    get_color_map,
    hierarchy,
    hierarchy_level_0,
    hierarchy_level_1,
    level_to_num_classes,
)
from utils.dataset import PascalPartDataset
from utils.device import get_device

LEVELS = {
    "mask_level_0": hierarchy_level_0,
    "mask_level_1": hierarchy_level_1,
    "mask_level_2": hierarchy,
}


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, backbone: str):
    """
    Build the model and load its weights from a checkpoint.

    Cached so the checkpoint is read once per session instead of on every rerun.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        backbone (str): Backbone name, read from the configuration file.

    Returns:
        Tuple[HierarchicalDeepLabV3, str]: The model in eval mode and the device it lives on.
    """
    device = get_device()
    model = HierarchicalDeepLabV3(
        num_classes_level_0=level_to_num_classes[0],
        num_classes_level_1=level_to_num_classes[1],
        num_classes_level_2=level_to_num_classes[2],
        backbone=backbone,
        # Weights come from the checkpoint, so skip downloading pretrained ones.
        pretrained=False,
    )
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply the validation transform to a raw image and add a batch dimension.

    Args:
        image (Image.Image): The uploaded image.

    Returns:
        torch.Tensor: Input tensor of shape (1, C, H, W).
    """
    transform = PascalPartDataset.get_transform(mode="val")
    mask = image.copy()
    transformed_image, _ = transform(image, mask)
    return transformed_image.unsqueeze(0)


def postprocess_mask(output: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Turn raw logits into a colour-coded RGB mask.

    Args:
        output (torch.Tensor): Logits of shape (1, num_classes, H, W).
        num_classes (int): Number of classes at this level.

    Returns:
        np.ndarray: RGB mask.
    """
    output = output.argmax(1).squeeze().cpu().numpy()
    return decode_segmap(output, num_classes)


def display_color_legend(color_map: np.ndarray, class_dict: dict) -> None:
    """
    Display the colour legend for a segmentation mask.

    Args:
        color_map (np.ndarray): Colour map for the segmentation classes.
        class_dict (dict): Dictionary mapping class indices to class names.
    """
    fig, ax = plt.subplots(figsize=(4, 2))
    handles = []
    labels = []

    for idx, color in enumerate(color_map):
        patch = plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color / 255, markersize=10
        )
        handles.append(patch)
        labels.append(class_dict.get(idx, f"Class {idx}"))

    ax.legend(handles, labels, loc="center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    """Render the Streamlit application."""
    parser = add_checkpoint_arg(
        build_parser("Visualise hierarchical human body segmentation results.")
    )
    # Streamlit injects its own arguments, so ignore anything we do not recognise.
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    try:
        checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    model, device = load_model(checkpoint_path, config["network"]["backbone"])

    st.title("Human Body Segmentation App")
    st.write("Upload an image to see the segmentation results for each level.")
    st.caption(f"Checkpoint: `{checkpoint_path}`")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    cols = st.columns(3)
    with cols[1]:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    st.write("### Segmentation Results")
    cols = st.columns(3)

    for idx, (level, class_dict) in enumerate(LEVELS.items()):
        num_classes = len(class_dict)
        segmented_image = postprocess_mask(outputs[level], num_classes)

        with cols[idx]:
            st.write(f"**{level.replace('_', ' ').title()}**")
            st.image(
                segmented_image,
                caption=f"Segmented Image - {level}",
                use_column_width=True,
            )
            display_color_legend(get_color_map(num_classes), class_dict)


if __name__ == "__main__":
    main()
