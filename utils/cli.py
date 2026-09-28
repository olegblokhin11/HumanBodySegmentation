import argparse
import os
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = "./configs/baseline_heavy.yml"

CHECKPOINT_FILENAMES = ("model_best.pth.tar", "checkpoint_best.pth.tar")


def load_config(config_path: str) -> dict[str, Any]:
    """
    Read a YAML configuration file into a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The parsed configuration.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_parser(description: str) -> argparse.ArgumentParser:
    """
    Create an argument parser with the options shared by all entry points.

    Args:
        description (str): Description shown in the ``--help`` output.

    Returns:
        argparse.ArgumentParser: Parser with the common arguments registered.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser


def add_checkpoint_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register the ``--checkpoint`` option on an existing parser.

    Args:
        parser (argparse.ArgumentParser): Parser to extend.

    Returns:
        argparse.ArgumentParser: The same parser, for chaining.
    """
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a model checkpoint (.pth.tar). If omitted, the newest "
            "model_best.pth.tar / checkpoint_best.pth.tar under the configured "
            "checkpoint_dir is used."
        ),
    )
    return parser


def resolve_checkpoint(config: dict[str, Any], checkpoint_path: Optional[str]) -> str:
    """
    Resolve the checkpoint to load, searching the configured directory when needed.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.
        checkpoint_path (Optional[str]): Explicit path from the command line.

    Returns:
        str: Path to an existing checkpoint file.

    Raises:
        FileNotFoundError: If no usable checkpoint can be located.
    """
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
        return checkpoint_path

    checkpoint_dir = config["training"]["checkpoint_dir"]
    candidates = []
    for root, _, filenames in os.walk(checkpoint_dir):
        for filename in filenames:
            if filename in CHECKPOINT_FILENAMES:
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under '{checkpoint_dir}'. Train a model first, or "
            "download the provided checkpoint and pass it via --checkpoint "
            "(see the README)."
        )

    return max(candidates, key=os.path.getmtime)
