import torch.nn.functional as F

def ddpm_noise_loss(e, e_pred):
    # Added: Check shapes before computing loss
    if e.shape != e_pred.shape:
        raise ValueError(
            f"Noise tensor size mismatch: e_pred {e_pred.shape}, e {e.shape}"
        )
    mse_loss = F.mse_loss(e_pred, e)
    return mse_loss