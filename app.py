import streamlit as st
import torch
from matplotlib import pyplot as plt
from PIL import Image

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from train import get_device
from utils.data_utils import (
    decode_segmap,
    get_color_map,
    hierarchy,
    hierarchy_level_0,
    hierarchy_level_1,
    level_to_num_classes,
)
from utils.dataset import PascalPartDataset


# Function to load the model
def load_model(checkpoint_path):
    device = get_device()
    model = HierarchicalDeepLabV3(
        num_classes_level_0=level_to_num_classes[0],
        num_classes_level_1=level_to_num_classes[1],
        num_classes_level_2=level_to_num_classes[2],
        backbone="resnet50",
    )
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


# Function to preprocess input image
def preprocess_image(image):
    transform = PascalPartDataset.get_transform(mode="val")

    mask = image.copy()

    transformed_image, _ = transform(image, mask)
    return transformed_image.unsqueeze(0)


# Function to post-process the output mask
def postprocess_mask(output, num_classes):
    # Convert model output to numpy image with the shape (H, W, C)
    output = output.argmax(1).squeeze().cpu().numpy()
    mask = decode_segmap(output, num_classes)
    return mask


def display_color_legend(color_map, class_dict):
    """
    Display the color legend for the segmentation masks.

    Args:
        color_map (np.ndarray): Color map for the segmentation classes.
        class_dict (Dict[int, str]): Dictionary mapping class indices to class names.
    """
    fig, ax = plt.subplots(figsize=(4, 2))
    handles = []
    labels = []

    for idx, color in enumerate(color_map):
        # Create a colored patch for each class
        patch = plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color / 255, markersize=10
        )
        handles.append(patch)
        label = class_dict.get(idx, f"Class {idx}")
        labels.append(label)

    # Show legend with the correct class names and colors
    ax.legend(handles, labels, loc="center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit App
def main():
    # Load configuration
    checkpoint_path = "./checkpoints/deeplab_resnet50_lr_0.001_batch_4_2/checkpoint_best.pth.tar"  # Update this path as needed

    # Load the model
    model, device = load_model(checkpoint_path)

    # Streamlit interface
    st.title("Image Segmentation App")
    st.write("Upload an image to see the segmentation results for each level.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Create three columns for displaying original image at center
        cols = st.columns(3)
        with cols[1]:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        input_tensor = preprocess_image(image).to(device)

        # Get the segmented output from the model
        with torch.no_grad():
            outputs = model(input_tensor)

        # Post-process and display segmentation results for each level
        st.write("### Segmentation Results")

        # Show legend for each hierarchical level
        levels = {
            "mask_level_0": hierarchy_level_0,
            "mask_level_1": hierarchy_level_1,
            "mask_level_2": hierarchy,
        }

        # Create three columns for displaying segmented images
        cols = st.columns(3)

        for idx, (level, class_dict) in enumerate(levels.items()):
            num_classes = len(class_dict)
            segmented_image = postprocess_mask(outputs[level], num_classes)

            # Place each image in a separate column
            with cols[idx]:
                st.write(f"**{level.replace('_', ' ').title()}**")
                st.image(
                    segmented_image,
                    caption=f"Segmented Image - {level}",
                    use_column_width=True,
                )

                # Show color legend for the segmentation classes
                color_map = get_color_map(num_classes)
                display_color_legend(color_map, class_dict)


if __name__ == "__main__":
    main()
