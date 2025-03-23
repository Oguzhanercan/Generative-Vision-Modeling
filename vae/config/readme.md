


## Choosing Hyperparameters for Model Creation

Configuring the VAE model in `config/config.yaml` requires careful selection of hyperparameters to ensure compatibility and optimal performance. Below are guidelines for choosing key hyperparameters under the `model`, `data`, and related sections to avoid issues like dimension mismatches and to achieve good training results.

### 1. Image Size and Downsampling
- **Parameter**: `data.image_size`, `model.downsample_positions`
- **Constraint**: The `image_size` must be divisible by `2^len(downsample_positions)` to ensure the spatial dimensions remain valid after downsampling.
  - Example: If `downsample_positions: [0, 1]` (2 downsamples), `image_size` must be divisible by `2^2 = 4`. For `image_size: 256`, this is valid (256 / 4 = 64).
- **Recommendation**: Start with a power of 2 (e.g., 256, 128) for `image_size`. Adjust `downsample_positions` based on the desired latent spatial size. For example, 2 downsamples reduce 256 to 64, which is manageable for most datasets.

### 2. Hidden Dimensions and Block Configurations
- **Parameter**: `model.hidden_dims`, `model.block_configs`
- **Constraint**: `hidden_dims` defines the channel sizes for each block in the encoder and decoder. The number of blocks is determined by `len(block_configs)`.
  - If `len(block_configs) > len(hidden_dims)`, the last `hidden_dims` value is reused.
  - Example: `hidden_dims: [64, 128, 256]` with `block_configs` of length 3 works fine.
- **Recommendation**: Choose `hidden_dims` to progressively increase channels in the encoder (e.g., `[64, 128, 256]`) to capture more features. Use a mix of `resnet` and `attention` blocks in `block_configs` for better feature extraction. For smaller models, reduce the number of blocks or channels.

### 3. Downsample Positions
- **Parameter**: `model.downsample_positions`
- **Constraint**: Indices in `downsample_positions` must be less than `len(block_configs)`.
  - Example: With `block_configs` of length 3 (indices 0, 1, 2), `downsample_positions: [0, 1]` is valid.
- **Recommendation**: Distribute downsamples early in the encoder (e.g., `[0, 1]`) to reduce spatial dimensions quickly, allowing the model to focus on higher-level features. Avoid too many downsamples if your `image_size` is small.

### 4. Latent Dimensions and Vector Quantization (VQ)
- **Parameter**: `model.latent_dim`, `model.vq_embedding_dim`, `model.vq_num_embeddings`, `model.use_vq`
- **Constraint**: If `use_vq: true`, ensure `vq_embedding_dim` matches `latent_dim`. `vq_num_embeddings` should be large enough to represent the latent space.
  - Example: `latent_dim: 128`, `vq_embedding_dim: 128`, `vq_num_embeddings: 512`.
- **Recommendation**: Start with a `latent_dim` of 128 or 256 for a good balance between reconstruction quality and compression. For VQ, `vq_num_embeddings` in the range of 512 to 2048 works well for most tasks. Adjust `vq_beta` (e.g., 0.25 to 1.0) to balance commitment and embedding losses.

### 5. Data Channels
- **Parameter**: `data.data_channel`
- **Constraint**: Must match the input data format (e.g., 3 for RGB, 1 for grayscale).
- **Recommendation**: Set to 3 for RGB images, which is the most common case. Ensure your dataset matches this setting.

### 6. Loss Weights and Types
- **Parameter**: `loss.use_perceptual`, `loss.use_lpips`, `loss.use_ssim`, `loss.beta_recon`, `loss.beta_kl`, `loss.reconstruction_loss`
- **Constraint**: `reconstruction_loss` must be either `"mse"` or `"l1"`. Weights should be positive.
- **Recommendation**:
  - Use `reconstruction_loss: "l1"` for sharper reconstructions, or `"mse"` for smoother results.
  - Set `beta_recon` and `beta_kl` to balance reconstruction quality and latent space regularization (e.g., `beta_recon: 0.5`, `beta_kl: 1.0`).
  - Enable `use_perceptual`, `use_lpips`, and `use_ssim` for better perceptual quality, with weights around 1.0. Adjust weights based on the importance of each loss term.

## 7. Memory Optimization Options

The `optimize_memory_usage` section in `config/config.yaml` provides options to reduce GPU memory usage during training. These options can help stabilize memory consumption, especially if you observe large memory spikes (e.g., from 4800 MB to 14800 MB per iteration). Below are the available flags and their effects:

### `optimize_memory_usage` Configuration

- **`use_gradient_checkpointing` (bool, default: false)**:
  - When enabled, gradient checkpointing is applied to the VAE model. This trades computation for memory by recomputing intermediate activations during the backward pass instead of storing them.
  - **Effect**: Reduces peak memory usage significantly, especially for deep models with many layers.
  - **Trade-off**: Increases computation time slightly due to recomputation.
  - **Recommendation**: Enable if memory usage is a concern, especially with large models or high resolutions.

- **`use_mixed_precision` (bool, default: false)**:
  - When enabled, mixed precision training is used, performing most computations in float16 instead of float32.
  - **Effect**: Halves the memory usage for activations and can speed up training on compatible GPUs (e.g., NVIDIA GPUs with Tensor Cores).
  - **Trade-off**: May introduce small numerical differences; ensure your GPU supports mixed precision (e.g., NVIDIA Volta, Turing, or Ampere architectures).
  - **Recommendation**: Enable for a quick memory reduction without altering the model architecture.

- **`perceptual_loss_batch_size` (int, default: 0)**:
  - Specifies the batch size for computing perceptual losses (e.g., VGG-based or LPIPS losses). Set to 0 to disable (process the entire batch at once).
  - **Effect**: Reduces peak memory usage by processing perceptual losses in smaller batches (e.g., 4 images at a time).
  - **Trade-off**: Slightly increases computation time due to multiple forward passes.
  - **Recommendation**: Set to a small value (e.g., 4) if perceptual losses are causing memory spikes.

### Example Configuration
```yaml
optimize_memory_usage:
  use_gradient_checkpointing: true
  use_mixed_precision: true
  perceptual_loss_batch_size: 4

# training section
training:
  epochs: int (e.g., 200) - Number of training epochs
  batch_size: int (e.g., 64) - Batch size for training
  learning_rate: float (e.g., 1e-4) - Learning rate for VAE optimizer
  device: str (e.g., "cuda") - Device to run on
  log_interval: int (e.g., 100) - Interval (in batches) for logging
  checkpoint_dir: str (e.g., "./checkpoints") - Directory to save checkpoints
  output_dir: str (e.g., "./outputs") - Directory for outputs
  limit_samples: int (e.g., 10000) - Limit dataset size (-1 for no limit)
  n_best: int (e.g., 5) - Number of best checkpoints to keep

# data section
data:
  dataset_path: str (e.g., "/home/oguzhan/Downloads/images/") - Path to dataset
  image_size: int (e.g., 64) - Size to resize images
  val_split: float (e.g., 0.1) - Fraction of dataset for validation

# model section
model:
  input_dim: int (e.g., 64) - Initial channel dimension
  hidden_dims: list[int] (e.g., [64, 128, 256]) - Hidden channel dimensions
  block_configs: list[dict] (e.g., [{type: "resnet"}, {type: "attention"}, {type: "resnet"}]) - Block types
  downsample_positions: list[int] (e.g., [0, 1]) - Positions to downsample
  input_size: int (e.g., 64) - Input image size
  latent_dim: int (e.g., 32) - Latent dimension (if not using VQ)
  vq_embedding_dim: int (e.g., 32) - VQ embedding dimension
  vq_num_embeddings: int (e.g., 512) - Number of VQ embeddings
  vq_beta: float (e.g., 0.25) - VQ commitment loss weight
  use_vq: bool (e.g., true) - Whether to use vector quantization
  use_ema: bool (e.g., true) - Use EMA for VQ updates
  ema_decay: float (e.g., 0.99) - EMA decay rate
  constant_variance: bool (e.g., false) - Use constant variance for KL
  num_groups: int (e.g., 32) - Number of groups for GroupNorm

# loss section
loss:
  use_perceptual: bool (e.g., false) - Use perceptual loss
  perceptual_weight: float (e.g., 1.0) - Weight for perceptual loss
  use_lpips: bool (e.g., false) - Use LPIPS loss
  lpips_weight: float (e.g., 1.0) - Weight for LPIPS loss
  use_ssim: bool (e.g., false) - Use SSIM loss
  ssim_weight: float (e.g., 1.0) - Weight for SSIM loss
  beta_recon: float (e.g., 0.5) - Weight for reconstruction loss
  kl_annealing_epochs: int (e.g., 20) - Epochs for KL annealing
  beta_kl: float (e.g., 1.0) - Weight for KL divergence
  reconstruction_loss: str (e.g., "mse") - Type of reconstruction loss ("mse" or "l1")

# gan section
gan:
  enabled: bool (e.g., false) - Enable GAN training
  discriminator_lr: float (e.g., 2e-4) - Learning rate for discriminator
  gan_weight: float (e.g., 1.0) - Weight for generator loss

