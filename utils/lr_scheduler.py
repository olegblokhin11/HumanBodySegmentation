import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer


def get_lr_scheduler(config: dict, num_iters_per_epoch: int, optimizer: Optimizer):
    """
    Creates a learning rate scheduler based on the provided configuration.

    The function supports four types of schedulers:
    - StepLR: Reduces the learning rate by a factor `gamma` every `step_size` epochs.
    - CosineAnnealingLR: Follows a cosine decay schedule until reaching a minimum value `eta_min`.
    - PolynomialLR: Decreases the learning rate following a polynomial decay formula.
    - CyclicLR: Cycles the learning rate between `base_lr` and `max_lr` over a defined cycle length.

    Additionally, a warm-up phase can be added using `SequentialLR` to smoothly increase the learning rate
    at the beginning of training.

    Args:
        config (dict): Configuration dictionary containing the scheduler setup. Example structure:
            {
                "training": {
                    "epochs": int,  # Total number of training epochs
                    "lr_scheduler": {
                        "mode": str,  # One of ['step', 'cos', 'poly', 'cyclic']
                        "step_size": int,  # StepLR specific, number of epochs before reducing LR
                        "gamma": float,  # StepLR specific, decay factor
                        "ep_max": int,  # CosineAnnealing specific, max epochs until constant LR
                        "eta_min": float,  # CosineAnnealing specific, minimum learning rate
                        "power": float,  # PolynomialLR specific, polynomial power
                        "base_lr": float,  # CyclicLR specific, minimum learning rate
                        "max_lr": float,  # CyclicLR specific, maximum learning rate
                        "step_size_up": int,  # CyclicLR specific, iterations to reach max_lr
                        "cyclic_mode": str,  # CyclicLR specific, mode ('triangular', 'triangular2', 'exp_range')
                        "use_warmup": bool,  # Whether to use a warm-up phase
                        "warmup_epochs": int  # Number of epochs for the warm-up phase (optional)
                    }
                }
            }
        num_iters_per_epoch (int): Number of iterations (batches) per epoch.
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.

    Returns:
        torch.optim.lr_scheduler: Configured learning rate scheduler.
    """
    # Retrieve the mode and verify it's valid
    mode = config["training"]["lr_scheduler"]["mode"]
    assert mode in [
        "step",
        "cos",
        "poly",
        "cyclic",
    ], "Invalid LR scheduler mode. Choose from 'step', 'cos', 'poly', or 'cyclic'."

    # Extract scheduler configuration and total training epochs
    num_epochs = config["training"]["epochs"]
    lr_scheduler_config = config["training"]["lr_scheduler"]

    # Initialize the main scheduler based on the selected mode
    if mode == "step":
        step_size = lr_scheduler_config["step_size"] * num_iters_per_epoch
        gamma = lr_scheduler_config["gamma"]
        main_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif mode == "cos":
        t_max = lr_scheduler_config["ep_max"] * num_iters_per_epoch
        eta_min = lr_scheduler_config["eta_min"]  # Minimum learning rate
        main_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    elif mode == "poly":
        total_iters = num_epochs * num_iters_per_epoch
        power = lr_scheduler_config["power"]  # Polynomial power
        main_scheduler = lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_iters, power=power
        )

    elif mode == "cyclic":
        base_lr = lr_scheduler_config["base_lr"]
        max_lr = lr_scheduler_config["max_lr"]
        step_size_up = lr_scheduler_config["step_size_up"] * num_iters_per_epoch
        cyclic_mode = lr_scheduler_config["cyclic_mode"]
        main_scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=cyclic_mode,
        )

    # Check if a warm-up phase is required
    if lr_scheduler_config.get("use_warmup", False):
        warmup_epochs = lr_scheduler_config.get(
            "warmup_epochs", 5
        )  # Default to 5 epochs
        warmup_iters = warmup_epochs * num_iters_per_epoch
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters
        )

        # Combine warm-up and main scheduler using SequentialLR
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters],
        )
    else:
        scheduler = main_scheduler

    return scheduler
