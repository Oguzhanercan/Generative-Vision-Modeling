














# Inference
Usage Examples
Basic Inference:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth

Custom Test Dataset:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth --test_path /path/to/test/data

With TensorBoard and More Batches:

python inference.py --config config/config.yaml --checkpoint checkpoints/vae_step_100_valloss_1.2345.pth --num_batches 10 --tensorboard_dir inference_tensorboard


## Training Experiments and Checkpoints

You can reach the relative files with this [link](https://drive.google.com/drive/u/0/folders/1F7dsMHlm52RYLvepCr7Kcbc500n_Vgdd). 
