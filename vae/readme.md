














# Inference
Usage Examples
Basic Inference:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth

Custom Test Dataset:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth --test_path /path/to/test/data

With TensorBoard and More Batches:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth --num_batches 10 --tensorboard_dir inference_tensorboard


## Training Experiments and Checkpoints

| Experiment ID | Epochs | Batch Size | Learning Rate | Image Size | Data Channels | Hidden Dims | Block Configs | Latent Dim | Use VQ | VQ Embedding Dim | VQ Num Embeddings | Use Perceptual | Perceptual Weight | Use LPIPS | LPIPS Weight | Use SSIM | SSIM Weight | Beta Recon | Beta KL | Reconstruction Loss | GAN Enabled | GAN Weight | Checkpoint Link                                  |
|---------------|--------|------------|---------------|------------|---------------|-------------|---------------|------------|--------|------------------|-------------------|----------------|-------------------|-----------|--------------|----------|-------------|------------|---------|---------------------|-------------|------------|--------------------------------------------------|
| Experiment 1  | 200    | 16         | 1e-4          | 256        | 3             | [64, 128, 256] | resnet,attn,resnet | 128        | false  | -                | -                 | true           | 1.0               | true      | 1.0          | true     | 1.0         | 0.5        | 1.0     | l1                  | false       | -          | [Link to Checkpoint 1](YOUR_GOOGLE_DRIVE_LINK_1) |
| Experiment 2  | 300    | 32         | 5e-5          | 256        | 3             | [128, 256, 512] | resnet,attn,resnet | 256        | true   | 256              | 512               | true           | 1.0               | true      | 1.0          | false    | -           | 0.5        | 1.0     | mse                 | false       | -          | [Link to Checkpoint 2](YOUR_GOOGLE_DRIVE_LINK_2) |
| Experiment 3  | 200    | 16         | 1e-4          | 128        | 3             | [64, 128]      | resnet,resnet     | 64         | false  | -                | -                 | false          | -                 | false     | -            | false    | -           | 0.5        | 1.0     | l1                  | true        | 0.5        | [Link to Checkpoint 3](YOUR_GOOGLE_DRIVE_LINK_3) |
| ...           | ...    | ...        | ...           | ...        | ...           | ...         | ...           | ...        | ...    | ...              | ...               | ...            | ...               | ...       | ...          | ...      | ...         | ...        | ...     | ...                 | ...         | ...        | ...                                              |