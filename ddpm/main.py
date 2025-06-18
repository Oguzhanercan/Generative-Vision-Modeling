import yaml  # Added: Import for YAML parsing
from types import SimpleNamespace  # Added: For converting dict to object
from data.get_dataloader import get_dataloaders  # Added: Import dataloader function
from core.trainer import Trainer  # Added: Import Trainer class
from models.unet import UNet  # Added: Import UNet model
import torch


def load_config(config_path):
    # Modified: Updated to cast YAML values to correct types explicitly
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Training parameters
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["device"] = str(config["training"]["device"])
    config["training"]["log_interval"] = int(config["training"]["log_interval"])
    config["training"]["output_dir"] = str(config["training"]["output_dir"])
    config["training"]["limit_samples"] = int(config["training"]["limit_samples"])
    config["training"]["n_best"] = int(config["training"]["n_best"])

    # Data parameters
    config["data"]["dataset_path"] = str(config["data"]["dataset_path"])
    config["data"]["image_size"] = int(config["data"]["image_size"])
    config["data"]["val_split"] = float(config["data"]["val_split"])
    config["data"]["data_channel"] = int(config["data"]["data_channel"])

    # DDPM parameters
    config["ddpm"]["steps"] = int(config["ddpm"]["steps"])
    config["ddpm"]["beta_start"] = float(config["ddpm"]["beta_start"])
    config["ddpm"]["beta_end"] = float(config["ddpm"]["beta_end"])
    config["ddpm"]["beta_schedule"] = str(config["ddpm"]["beta_schedule"])
    config["ddpm"]["model_type"] = str(config["ddpm"]["model_type"])

    # Convert dict to nested namespace for dot access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    return dict_to_namespace(config)

def main():
    config = load_config("configs/config.yaml")  # Added: Load YAML config
    train_loader, val_loader = get_dataloaders(config)  # Modified: Pass YAML config
    model = UNet(in_channels=config.data.data_channel, out_channels=config.data.data_channel, base_channels=64, time_emb_dim=256)  # Modified: Use data_channel from YAML
    trainer = Trainer(model, config)  # Modified: Pass YAML config
    trainer.train(train_loader, val_loader)  # Added: Run training

if __name__ == "__main__":
    main()  # Added: Entry point