"""Render qualitative predictions for the README.

Produces a grid comparing, for a few validation samples, the input image and the
ground-truth masks against the model predictions at all three hierarchy levels.

Usage:
    python scripts/visualize_predictions.py --num-samples 3 --output docs/fig_predictions.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierarchical_deeplabv3 import HierarchicalDeepLabV3
from utils.cli import add_checkpoint_arg, build_parser, load_config, resolve_checkpoint
from utils.data_utils import decode_segmap, denormalize_image, level_to_num_classes
from utils.dataset import PascalPartDataset
from utils.device import get_device

LEVEL_KEYS = ["mask_level_0", "mask_level_1", "mask_level_2"]


def to_displayable(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor into a displayable HWC array."""
    image = denormalize_image(image_tensor).clamp(0, 1)
    return image.cpu().numpy().transpose(1, 2, 0)


def build_model(config, checkpoint_path):
    """Create the model and load pretrained weights from a checkpoint."""
    device = get_device()
    model = HierarchicalDeepLabV3(
        num_classes_level_0=level_to_num_classes[0],
        num_classes_level_1=level_to_num_classes[1],
        num_classes_level_2=level_to_num_classes[2],
        backbone=config["network"]["backbone"],
        pretrained=False,
    )
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def select_samples(
    dataset,
    num_samples: int,
    min_body_fraction: float,
    max_body_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick samples where the person actually occupies a useful part of the frame.

    Most validation crops are almost entirely background, which makes for an
    uninformative figure. Samples are therefore ranked by the fraction of
    non-background pixels in the ground-truth mask and picked from a target band.

    Args:
        dataset: The dataset to draw samples from.
        num_samples (int): How many samples to return.
        min_body_fraction (float): Lower bound of the preferred body-pixel fraction.
        max_body_fraction (float): Upper bound of the preferred body-pixel fraction.
        rng (np.random.Generator): Random generator used to break ties.

    Returns:
        np.ndarray: Indices of the selected samples.
    """
    fractions = np.array(
        [float((np.load(mask_path) > 0).mean()) for mask_path in dataset.mask_paths]
    )

    in_band = np.flatnonzero((fractions >= min_body_fraction) & (fractions <= max_body_fraction))
    if len(in_band) < num_samples:
        print(
            f"Only {len(in_band)} samples fall in the "
            f"[{min_body_fraction}, {max_body_fraction}] body-fraction band; "
            "falling back to the most populated samples."
        )
        in_band = np.argsort(fractions)[-max(num_samples * 10, num_samples) :]

    # Shuffle first so ties are broken randomly, then take the most populated ones.
    shuffled = rng.permutation(in_band)
    ranked = shuffled[np.argsort(fractions[shuffled])[::-1]]
    return ranked[:num_samples]


def render(
    config,
    checkpoint_path,
    num_samples,
    split,
    output_path,
    seed,
    min_body_fraction,
    max_body_fraction,
):
    """Build the comparison grid and save it to ``output_path``."""
    model, device = build_model(config, checkpoint_path)
    dataset = PascalPartDataset(config, mode=split)

    rng = np.random.default_rng(seed)
    indices = select_samples(
        dataset=dataset,
        num_samples=num_samples,
        min_body_fraction=min_body_fraction,
        max_body_fraction=max_body_fraction,
        rng=rng,
    )
    print(f"Selected samples: {list(indices)}")

    columns = ["Input"]
    for level_idx in range(3):
        columns += [f"GT L{level_idx}", f"Pred L{level_idx}"]

    fig, axes = plt.subplots(
        num_samples, len(columns), figsize=(2.1 * len(columns), 2.4 * num_samples)
    )
    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        sample = dataset[int(index)]
        image = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)

        axes[row, 0].imshow(to_displayable(sample["image"]))
        axes[row, 0].set_ylabel(
            f"sample {int(index)}", fontsize=8, rotation=0, labelpad=28, va="center"
        )

        for level_idx, key in enumerate(LEVEL_KEYS):
            num_classes = level_to_num_classes[level_idx]

            gt = sample[key].cpu().numpy()
            pred = outputs[key].argmax(dim=1).squeeze(0).cpu().numpy()

            axes[row, 1 + level_idx * 2].imshow(decode_segmap(gt, num_classes))
            axes[row, 2 + level_idx * 2].imshow(decode_segmap(pred, num_classes))

        for col in range(len(columns)):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=10)

    fig.suptitle(
        f"Hierarchical segmentation on the {split} split "
        f"({config['network']['backbone']} backbone)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = add_checkpoint_arg(
        build_parser("Render qualitative segmentation predictions for the README.")
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", default="docs/fig_predictions.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-body-fraction",
        type=float,
        default=0.10,
        help="Prefer samples where at least this share of pixels is non-background.",
    )
    parser.add_argument(
        "--max-body-fraction",
        type=float,
        default=0.60,
        help="Prefer samples where at most this share of pixels is non-background.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)

    render(
        config=config,
        checkpoint_path=checkpoint_path,
        num_samples=args.num_samples,
        split=args.split,
        output_path=args.output,
        seed=args.seed,
        min_body_fraction=args.min_body_fraction,
        max_body_fraction=args.max_body_fraction,
    )


if __name__ == "__main__":
    main()
