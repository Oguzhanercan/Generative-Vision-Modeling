# core/schedulers.py
def vq_beta_scheduler(epoch: int, config: dict) -> float:
    """
    Schedules the VQ beta value based on the current epoch.

    Args:
        epoch (int): Current epoch number.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        float: Updated VQ beta value.
    """
    initial_beta = config["model"]["vq_beta"]
    max_beta = 1.0
    warmup_epochs = config.get("model", {}).get("vq_beta_warmup", 10)
    if epoch < warmup_epochs:
        return initial_beta + (max_beta - initial_beta) * (epoch / warmup_epochs)
    return max_beta