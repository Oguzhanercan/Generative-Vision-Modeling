# vis_tools/vis.py
import yaml
import graphviz
import torch
import torch.nn as nn
from typing import List

# Re-define the blocks and VAE/AE components for parameter calculation and visualization
def swish(x):
    return x * torch.sigmoid(x)

class AttnBlockVis(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None
    def forward(self, x):
        return x

class ResnetBlockVis(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, num_groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None
    def forward(self, x):
        return x

class DownsampleVis(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        return x

class UpsampleVis(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return x

class DynamicBlockVis(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_type: str, num_groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_type = block_type
        if block_type == "resnet":
            self.block = ResnetBlockVis(in_channels, out_channels, num_groups=num_groups)
        elif block_type == "attention":
            self.block = AttnBlockVis(in_channels, out_channels, num_groups=num_groups)
    def forward(self, x):
        return self.block(x)

class VectorQuantizerVis(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    def forward(self, z):
        return z, torch.tensor(0.0), None

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def visualize_architecture_graph(config, output_path="vae_architecture.png"):
    """
    Visualizes the VAE or AE architecture using graphviz with detailed shapes and parameters.

    Args:
        config (dict): Configuration dictionary containing model_type ("vae" or "ae").
        output_path (str): Path to save the visualization (should include .png extension).
    """
    model_type = config["model"]["model_type"]
    dot = graphviz.Digraph(comment=f'{model_type.upper()} Architecture', format='png', graph_attr={'rankdir': 'TB'})
    node_attr = {'shape': 'box', 'style': 'filled'}
    colors = {'Attention Block': 'lightblue', 'Resnet Block': 'lightgreen', 'Conv': 'yellow',
              'Downsample': 'orange', 'Upsample': 'purple', 'VQ': 'pink', 'Input': 'gray95',
              'Output': 'gray95', 'Latent': 'lightgray'}

    input_size = config["model"]["input_size"]
    input_dim = config["model"]["input_dim"]
    hidden_dims = config["model"]["hidden_dims"]
    block_configs = config["model"]["block_configs"]
    downsample_positions = sorted(config["model"]["downsample_positions"])
    num_downsamples = len(downsample_positions)
    latent_dim = config["model"]["latent_dim"]
    use_vq = config["model"]["use_vq"]
    vq_embedding_dim = config["model"]["vq_embedding_dim"]
    vq_num_embeddings = config["model"]["vq_num_embeddings"]
    num_groups = config["model"].get("num_groups", 32)

    # Encoder
    current_shape = [1, 3, input_size, input_size]
    dot.node('input', f'Input\nShape: {current_shape}', fillcolor=colors['Input'], **node_attr)

    initial_conv = nn.Conv2d(3, input_dim, kernel_size=3, padding=1)
    params = count_parameters(initial_conv)
    next_shape = [current_shape[0], input_dim, current_shape[2], current_shape[3]]
    dot.node('initial_conv', f'Initial Conv\nShape: {current_shape} -> {next_shape}\nParams: {params}', fillcolor=colors['Conv'], **node_attr)
    dot.edge('input', 'initial_conv')
    current_shape = next_shape

    encoder_blocks = []
    block_counter = 0
    downsample_counter = 0
    for i in range(len(block_configs)):
        block_type_str = block_configs[i % len(block_configs)]["type"].capitalize()
        color_key = block_type_str + " Block" if block_type_str in ["Attention", "Resnet"] else block_type_str
        block_name = f'encoder_block_{block_counter}'
        block = DynamicBlockVis(current_shape[1], hidden_dims[min(i, len(hidden_dims)-1)], block_configs[i % len(block_configs)]["type"], num_groups=num_groups)
        params = count_parameters(block)
        next_shape = [current_shape[0], hidden_dims[min(i, len(hidden_dims)-1)], current_shape[2], current_shape[3]]
        dot.node(block_name, f'{block_type_str} Block\nShape: {current_shape} -> {next_shape}\nParams: {params}', fillcolor=colors.get(color_key, 'white'), **node_attr)
        if encoder_blocks:
            dot.edge(encoder_blocks[-1], block_name)
        else:
            dot.edge('initial_conv', block_name)
        encoder_blocks.append(block_name)
        current_shape = next_shape
        block_counter += 1
        if downsample_counter < num_downsamples and i == downsample_positions[downsample_counter]:
            downsample = DownsampleVis(current_shape[1])
            params = count_parameters(downsample.conv)
            current_shape = [current_shape[0], current_shape[1], current_shape[2] // 2, current_shape[3] // 2]
            downsample_name = f'downsample_{downsample_counter}'
            dot.node(downsample_name, f'Downsample\nShape: {next_shape} -> {current_shape}\nParams: {params}', fillcolor=colors['Downsample'], **node_attr)
            dot.edge(encoder_blocks[-1], downsample_name)
            encoder_blocks.append(downsample_name)
            downsample_counter += 1

    # Latent Space Output based on model_type
    if model_type == "vae":
        mu_conv = nn.Conv2d(current_shape[1], latent_dim, 1)
        params_mu = count_parameters(mu_conv)
        mu_shape = [current_shape[0], latent_dim, current_shape[2], current_shape[3]]
        dot.node('mu', f'Conv (mu)\nShape: {current_shape} -> {mu_shape}\nParams: {params_mu}', fillcolor=colors['Conv'], **node_attr)
        logvar_conv = nn.Conv2d(current_shape[1], latent_dim, 1)
        params_logvar = count_parameters(logvar_conv)
        logvar_shape = [current_shape[0], latent_dim, current_shape[2], current_shape[3]]
        dot.node('logvar', f'Conv (logvar)\nShape: {current_shape} -> {logvar_shape}\nParams: {params_logvar}', fillcolor=colors['Conv'], **node_attr)
        dot.edge(encoder_blocks[-1], 'mu')
        dot.edge(encoder_blocks[-1], 'logvar')
        current_shape = mu_shape  # For the next layer
    else:  # "ae"
        latent_conv = nn.Conv2d(current_shape[1], latent_dim, 1)
        params_latent = count_parameters(latent_conv)
        latent_shape = [current_shape[0], latent_dim, current_shape[2], current_shape[3]]
        dot.node('latent_direct', f'Conv (z)\nShape: {current_shape} -> {latent_shape}\nParams: {params_latent}', fillcolor=colors['Conv'], **node_attr)
        dot.edge(encoder_blocks[-1], 'latent_direct')
        current_shape = latent_shape

    # Latent Space and VQ
    if use_vq:
        dot.node('latent_vq_in', f'Latent Space\nShape: {current_shape}', fillcolor=colors['Latent'], **node_attr)
        if model_type == "vae":
            dot.edge('mu', 'latent_vq_in')
            dot.edge('logvar', 'latent_vq_in')
        else:  # "ae"
            dot.edge('latent_direct', 'latent_vq_in')
        vq_layer = VectorQuantizerVis(vq_num_embeddings, vq_embedding_dim)
        params_vq = count_parameters(vq_layer.embedding)
        vq_shape = [current_shape[0], vq_embedding_dim, current_shape[2], current_shape[3]]
        dot.node('vq', f'VectorQuantizer\nShape: {current_shape} -> {vq_shape}\nParams: {params_vq}', fillcolor=colors['VQ'], **node_attr)
        dot.edge('latent_vq_in', 'vq')
        current_shape = vq_shape
    else:
        dot.node('latent', f'Latent Space\nShape: {current_shape}', fillcolor=colors['Latent'], **node_attr)
        if model_type == "vae":
            dot.edge('mu', 'latent')
            dot.edge('logvar', 'latent')
        else:  # "ae"
            dot.edge('latent_direct', 'latent')

    # Decoder
    decoder_initial_conv = nn.Conv2d(current_shape[1], hidden_dims[-1], 1)
    params_dec_init = count_parameters(decoder_initial_conv)
    current_shape = [current_shape[0], hidden_dims[-1], current_shape[2], current_shape[3]]
    dot.node('decoder_initial_conv', f'Initial Conv\nShape: {mu_shape if (model_type == "vae" and not use_vq) else (latent_shape if (model_type == "ae" and not use_vq) else vq_shape)} -> {current_shape}\nParams: {params_dec_init}', fillcolor=colors['Conv'], **node_attr)
    dot.edge('latent' if not use_vq else 'vq', 'decoder_initial_conv')

    decoder_blocks = []
    upsample_counter = num_downsamples - 1
    block_counter = 0
    for i in range(len(block_configs)):
        in_channels = current_shape[1]
        out_channels = hidden_dims[max(len(hidden_dims) - 1 - i, 0)]
        block_type_str = block_configs[i % len(block_configs)]["type"].capitalize()
        color_key = block_type_str + " Block" if block_type_str in ["Attention", "Resnet"] else block_type_str
        block_name = f'decoder_block_{block_counter}'
        block = DynamicBlockVis(in_channels, out_channels, block_configs[i % len(block_configs)]["type"], num_groups=num_groups)
        params = count_parameters(block)
        next_shape = [current_shape[0], out_channels, current_shape[2], current_shape[3]]
        dot.node(block_name, f'{block_type_str} Block\nShape: {current_shape} -> {next_shape}\nParams: {params}', fillcolor=colors.get(color_key, 'white'), **node_attr)
        if decoder_blocks:
            dot.edge(decoder_blocks[-1], block_name)
        else:
            dot.edge('decoder_initial_conv', block_name)
        decoder_blocks.append(block_name)
        current_shape = next_shape
        block_counter += 1
        if upsample_counter >= 0 and i == len(block_configs) - 1 - upsample_counter:
            upsample = UpsampleVis(current_shape[1])
            params = count_parameters(upsample.conv)
            current_shape = [current_shape[0], current_shape[1], current_shape[2] * 2, current_shape[3] * 2]
            upsample_name = f'upsample_{upsample_counter}'
            dot.node(upsample_name, f'Upsample\nShape: {next_shape} -> {current_shape}\nParams: {params}', fillcolor=colors['Upsample'], **node_attr)
            dot.edge(decoder_blocks[-1], upsample_name)
            decoder_blocks.append(upsample_name)
            upsample_counter -= 1

    decoder_final_conv = nn.Conv2d(current_shape[1], 3, kernel_size=3, padding=1)
    params_dec_final = count_parameters(decoder_final_conv)
    output_shape = [current_shape[0], 3, current_shape[2], current_shape[3]]
    dot.node('output', f'Output\nShape: {current_shape} -> {output_shape}\nParams: {params_dec_final}', fillcolor=colors['Output'], **node_attr)
    dot.edge(decoder_blocks[-1], 'output')

    # Remove the .png extension from output_path for rendering
    output_base = output_path.rsplit('.png', 1)[0]
    dot.render(output_base, view=False, cleanup=True)
    print(f"{model_type.upper()} Architecture graph saved to {output_path}")

def visualize_training_pipeline_graph(config, output_path="vae_training_pipeline.png"):
    """
    Visualizes the VAE or AE training pipeline using graphviz.

    Args:
        config (dict): Configuration dictionary containing model_type ("vae" or "ae").
        output_path (str): Path to save the visualization (should include .png extension).
    """
    model_type = config["model"]["model_type"]
    dot = graphviz.Digraph(comment=f'{model_type.upper()} Training Pipeline', format='png', graph_attr={'rankdir': 'LR'})

    use_perceptual = config["loss"]["use_perceptual"]
    use_lpips = config["loss"]["use_lpips"]
    use_ssim = config["loss"]["use_ssim"]
    use_vq = config["model"]["use_vq"]
    reconstruction_loss_type = config["loss"]["reconstruction_loss"].upper()

    # Data flow
    dot.node('input', 'Input Image', shape='box')
    dot.node('model', f'{model_type.upper()} Model', shape='box')
    dot.node('recon', 'Reconstructed Image', shape='box')
    dot.edge('input', 'model')
    dot.edge('model', 'recon')

    # Loss calculations
    with dot.subgraph(name='cluster_losses') as c:
        c.attr(label='Loss Functions')
        c.node('recon_loss', f'Reconstruction Loss\n({reconstruction_loss_type})', shape='box')
        if model_type == "vae":
            c.node('kl_loss', 'KL Divergence Loss', shape='box')
        if use_vq:
            c.node('vq_loss', 'VQ Loss', shape='box')
        if use_perceptual:
            c.node('percep_loss', 'Perceptual Loss', shape='box')
        if use_lpips:
            c.node('lpips_loss', 'LPIPS Loss', shape='box')
        if use_ssim:
            c.node('ssim_loss', 'SSIM Loss', shape='box')
        c.node('total_loss', f'Total {model_type.upper()} Loss', shape='box')

        # Connections to Reconstruction Loss
        c.edge('recon', 'recon_loss')
        c.edge('input', 'recon_loss')

        # Connection to KL Loss (only for VAE)
        if model_type == "vae":
            c.node('encoder_output', 'Encoder Output\n(mu, logvar)', shape='ellipse')
            c.edge('model', 'encoder_output')
            c.edge('encoder_output', 'kl_loss')
        else:
            c.node('encoder_output', 'Encoder Output\n(z)', shape='ellipse')
            c.edge('model', 'encoder_output')

        # Connection to VQ Loss
        if use_vq:
            c.edge('model', 'vq_loss')

        # Connections to Perceptual Loss
        if use_perceptual:
            c.node('vgg', 'VGG Network', shape='ellipse')
            c.edge('recon', 'vgg')
            c.edge('input', 'vgg')
            c.edge('vgg', 'percep_loss')

        # Connections to LPIPS Loss
        if use_lpips:
            c.node('lpips', 'LPIPS Metric', shape='ellipse')
            c.edge('recon', 'lpips')
            c.edge('input', 'lpips')
            c.edge('lpips', 'lpips_loss')

        # Connections to SSIM Loss
        if use_ssim:
            c.node('ssim', 'SSIM Metric', shape='ellipse')
            c.edge('recon', 'ssim')
            c.edge('input', 'ssim')
            c.edge('ssim', 'ssim_loss')

        # Total Loss aggregation
        c.edge('recon_loss', 'total_loss')
        if model_type == "vae":
            c.edge('kl_loss', 'total_loss')
        if use_vq:
            c.edge('vq_loss', 'total_loss')
        if use_perceptual:
            c.edge('percep_loss', 'total_loss')
        if use_lpips:
            c.edge('lpips_loss', 'total_loss')
        if use_ssim:
            c.edge('ssim_loss', 'total_loss')

    # Remove the .png extension from output_path for rendering
    output_base = output_path.rsplit('.png', 1)[0]
    dot.render(output_base, view=False, cleanup=True)
    print(f"{model_type.upper()} Training Pipeline graph saved to {output_path}")

def visualize_attn_block(output_path="attn_block.png"):
    # (Unchanged from your original code)
    dot = graphviz.Digraph(comment='Attention Block', format='png', graph_attr={'rankdir': 'TB'})
    node_attr = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}

    dot.node('input', 'Input\nShape: BxCxHxW', **node_attr)
    dot.node('norm', 'GroupNorm', **node_attr)
    dot.edge('input', 'norm')
    dot.node('q_proj', 'Conv2d (Q)\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.node('k_proj', 'Conv2d (K)\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.node('v_proj', 'Conv2d (V)\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.edge('norm', 'q_proj')
    dot.edge('norm', 'k_proj')
    dot.edge('norm', 'v_proj')
    dot.node('reshape_q', 'Reshape\nBxCxHxW -> Bx1x(HxW)xC', **node_attr)
    dot.node('reshape_k', 'Reshape\nBxCxHxW -> Bx1x(HxW)xC', **node_attr)
    dot.node('reshape_v', 'Reshape\nBxCxHxW -> Bx1x(HxW)xC', **node_attr)
    dot.edge('q_proj', 'reshape_q')
    dot.edge('k_proj', 'reshape_k')
    dot.edge('v_proj', 'reshape_v')
    dot.node('attention', 'Scaled Dot-Product Attention', **node_attr)
    dot.edge('reshape_q', 'attention')
    dot.edge('reshape_k', 'attention')
    dot.edge('reshape_v', 'attention')
    dot.node('reshape_out', 'Reshape\nBx1x(HxW)xC -> BxCxHxW', **node_attr)
    dot.edge('attention', 'reshape_out')
    dot.node('proj_out', 'Conv2d (Out)\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.edge('reshape_out', 'proj_out')
    dot.node('residual', 'Residual Connection (+)', **{'shape': 'ellipse'})
    dot.edge('proj_out', 'residual')
    dot.edge('input', 'residual')
    dot.node('output', 'Output\nShape: BxCxHxW', **node_attr)
    dot.edge('residual', 'output')

    output_base = output_path.rsplit('.png', 1)[0]
    dot.render(output_base, view=False, cleanup=True)
    print(f"Attention Block visualization saved to {output_path}")

def visualize_resnet_block(output_path="resnet_block.png"):
    # (Unchanged from your original code)
    dot = graphviz.Digraph(comment='ResNet Block', format='png', graph_attr={'rankdir': 'TB'})
    node_attr = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgreen'}

    dot.node('input', 'Input\nShape: BxCxHxW', **node_attr)
    dot.node('norm1', 'GroupNorm', **node_attr)
    dot.edge('input', 'norm1')
    dot.node('swish1', 'Swish', **node_attr)
    dot.edge('norm1', 'swish1')
    dot.node('conv1', 'Conv2d\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.edge('swish1', 'conv1')
    dot.node('norm2', 'GroupNorm', **node_attr)
    dot.edge('conv1', 'norm2')
    dot.node('swish2', 'Swish', **node_attr)
    dot.edge('norm2', 'swish2')
    dot.node('conv2', 'Conv2d\nShape: BxCxHxW -> BxCxHxW', **node_attr)
    dot.edge('swish2', 'conv2')
    dot.node('residual', 'Residual Connection (+)', **{'shape': 'ellipse'})
    dot.edge('conv2', 'residual')
    dot.edge('input', 'residual', style='dashed', label='Shortcut')
    dot.node('output', 'Output\nShape: BxCxHxW', **node_attr)
    dot.edge('residual', 'output')

    output_base = output_path.rsplit('.png', 1)[0]
    dot.render(output_base, view=False, cleanup=True)
    print(f"ResNet Block visualization saved to {output_path}")

if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/config.yaml not found. Please make sure the config file is in the correct location.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config/config.yaml: {e}")
        exit()

    visualize_architecture_graph(config)
    visualize_training_pipeline_graph(config)
    visualize_attn_block()
    visualize_resnet_block()