import torch


def get_device() -> str:
    """
    Determine the appropriate device for training and inference.

    Prefers CUDA, then Apple MPS, and falls back to CPU.

    Returns:
        str: The device to be used ('cuda', 'mps', or 'cpu').
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device
