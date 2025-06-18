from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import torch  # Added: Import torch for tensor size checking

def get_dataloaders(args):  # Modified: Changed parameter to args for YAML compatibility
    """
    Create train and validation dataloaders with input size validation and sample limiting.
    
    Args:
        args: Configuration object from YAML.
    
    Returns:
        tuple: (train_loader, val_loader)
    
    Raises:
        ValueError: If image_size is incompatible with architecture or tensor size mismatch.
    """
    image_size = args.data.image_size  # Modified: Access image_size from YAML
    data_channel = args.data.data_channel  # Added: Access data_channel for size checking

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Modified: Explicitly use tuple for clarity
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # normalize to [-1, 1]
    ])
    
    dataset = datasets.ImageFolder(args.data.dataset_path, transform=transform)  # Modified: Use dataset_path from YAML
    
    limit_samples = args.training.limit_samples  # Modified: Use limit_samples from YAML
    if limit_samples > 0 and limit_samples < len(dataset):
        indices = np.random.choice(len(dataset), limit_samples, replace=False)
    else:
        indices = np.arange(len(dataset))
    
    val_split = args.data.val_split  # Modified: Use val_split from YAML
    num_val = int(len(indices) * val_split)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[num_val:], indices[:num_val]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset, batch_size=args.training.batch_size, sampler=train_sampler, num_workers=4  # Modified: Use batch_size from YAML
    )
    val_loader = DataLoader(
        dataset, batch_size=args.training.batch_size, sampler=val_sampler, num_workers=4  # Modified: Use batch_size from YAML
    )  # Modified: Fixed typo in closing brace for val_loader
    
    # Added: Verify the first batch's shape to catch size mismatches early
    for images, _ in train_loader:
        expected_shape = (args.training.batch_size, data_channel, image_size, image_size)
        actual_shape = images.shape
        print(f"First batch shape: {actual_shape}")  # Added: Log batch shape for debugging
        if actual_shape[1:] != expected_shape[1:]:  # Allow batch size to vary for last batch
            raise ValueError(
                f"Image tensor size mismatch: expected {expected_shape[1:]}, got {actual_shape[1:]}"
            )
        break
    
    return train_loader, val_loader