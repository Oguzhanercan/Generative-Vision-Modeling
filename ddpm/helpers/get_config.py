import yaml

def load_config(config_path):
    """
    Load and parse the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model parameters

    
    # Training parameters
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    config["training"]["log_interval"] = int(config["training"]["log_interval"])
    config["training"]["limit_samples"] = int(config["training"]["limit_samples"])
    config["training"]["n_best"] = int(config["training"]["n_best"])
    config["training"]["output_dir"] = config["training"].get("output_dir", "runs")
    
    # Data parameters
    config["data"]["image_size"] = int(config["data"]["image_size"])
    config["data"]["val_split"] = float(config["data"]["val_split"])
    


    config["ddpm"]["steps"] = int(config["ddpm"]["steps"])
    config["ddpm"]["beta_start"] = float(config["ddpm"]["beta_start"])
    config["ddpm"]["beta_end"] = float(config["ddpm"]["beta_end"])
    config["ddpm"]["beta_schedule"] = config["ddpm"].get("beta_schedule", "linear")
    return config