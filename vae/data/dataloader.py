# data/dataloader.py
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np

def get_dataloaders(config):
    """
    Create train and validation dataloaders with input size validation and sample limiting.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: (train_loader, val_loader)
    
    Raises:
        ValueError: If image_size is incompatible with architecture.
    """
    image_size = config["data"]["image_size"]
    hidden_dims = config["model"]["hidden_dims"]
    num_layers = len(hidden_dims)
    
    # Validate image_size
    min_latent_size = image_size // (2 ** num_layers)
    required_divisor = 2 ** num_layers
    if image_size % required_divisor != 0:
        raise ValueError(
            f"image_size {image_size}x{image_size} must be divisible by {required_divisor}x{required_divisor} "
            f"for {num_layers} layers (hidden_dims={hidden_dims})."
        )
    if min_latent_size < 1:
        raise ValueError(
            f"image_size {image_size}x{image_size} with {num_layers} layers "
            f"reduces spatial size to {min_latent_size}x{min_latent_size}."
        )
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(config["data"]["dataset_path"], transform=transform)
    
    # Limit samples
    limit_samples = config["training"]["limit_samples"]
    if limit_samples > 0 and limit_samples < len(dataset):
        indices = np.random.choice(len(dataset), limit_samples, replace=False)
    else:
        indices = np.arange(len(dataset))
    
    # Train-validation split
    val_split = config["data"]["val_split"]
    num_val = int(len(indices) * val_split)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[num_val:], indices[:num_val]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], sampler=train_sampler, num_workers=4
    )
    val_loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], sampler=val_sampler, num_workers=4
    )
    
    return train_loader, val_loader