import torch
import os
import shutil
import datetime
from models.ddpm import DDPM_Noise
from models.unet import UNet
from losses.loss import ddpm_noise_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from heapq import nlargest
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.training.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.training.T_max)
        self.device = args.training.device
        self.model.to(self.device)
        self.model.train()
        self.ddpm = DDPM_Noise(model, args)
        
        # Create run folder with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_directory = os.path.join(args.training.output_dir, f"run_{timestamp}")
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Copy config.yaml to run folder
        config_path = "configs/config.yaml"
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(self.log_directory, "config.yaml"))
        else:
            print(f"Warning: config.yaml not found at {config_path}, not copied to run folder")
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(os.path.join(self.log_directory, "tensorboard"))
        
        self.best_models = []

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.log_directory, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        self.best_models.append((val_loss, checkpoint_path))
        if len(self.best_models) > self.args.training.n_best:
            self.best_models = nlargest(self.args.training.n_best, self.best_models)
            for _, path in self.best_models:
                if path not in [p for _, p in self.best_models]:
                    if os.path.exists(path):
                        os.remove(path)
        print(f"Model saved at epoch {epoch}, Validation Loss: {val_loss:.4f}")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(self.device)
                batch_size = images.size(0)
                t = torch.randint(0, self.args.ddpm.steps, (batch_size,), device=self.device).long()
                e_pred, e_gt = self.ddpm.forward(images, t)
                loss = ddpm_noise_loss(e_gt, e_pred)
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        for epoch in range(self.args.training.epochs):
            self.model.train()
            for i, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                batch_size = images.size(0)
                t = torch.randint(0, self.args.ddpm.steps, (batch_size,), device=self.device).long()
                e_pred, e_gt = self.ddpm.forward(images, t)
                loss = ddpm_noise_loss(e_gt, e_pred)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if i % self.args.training.log_interval == 0:
                    print(f"Epoch [{epoch}/{self.args.training.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    global_step = epoch * len(train_loader) + i
                    self.writer.add_scalar("train/loss", loss.item(), global_step)
            
            self.scheduler.step()
            val_loss = self.validate(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{self.args.training.epochs}], Validation Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("learning_rate", current_lr, epoch)
            
            self.save_checkpoint(epoch, val_loss)
            if epoch % self.args.training.vis_interval == 0:
                self.visualize(epoch, val_loader)
        
        self.writer.close()
        print("Training complete.")

    def visualize(self, epoch, val_loader):
        self.model.eval()
        with torch.no_grad():
            # Get ground truth images from val_loader
            vis_batch_size = 4
            gt_images = next(iter(val_loader))[0][:vis_batch_size].to(self.device)
            
            # Add noise to GT images up to t=900
            t_start = 900
            t_tensor = torch.full((vis_batch_size,), t_start, device=self.device, dtype=torch.long)
            x_start = self.ddpm.add_noise(gt_images, t_tensor)
            
            # Denoise the noisy GT images
            samples, x0_predictions = self.ddpm.sample(x_start)
            samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            samples = samples.clamp(0, 1)
            for t in x0_predictions:
                x0_predictions[t] = (x0_predictions[t] + 1) / 2  # Denormalize x_0 predictions
                x0_predictions[t] = x0_predictions[t].clamp(0, 1)

            # Denormalize GT images for visualization
            gt_images = (gt_images + 1) / 2
            gt_images = gt_images.clamp(0, 1)

            # Create table visualization: 4 rows (samples), 11 columns (t=900, 800, ..., 0, Final, GT)
            num_samples = min(4, samples.size(0))
            timesteps = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
            fig, axes = plt.subplots(num_samples, 11, figsize=(22, 8))
            
            for i in range(num_samples):
                # Plot x_0 predictions for each timestep
                for j, t in enumerate(timesteps):
                    ax = axes[i, j] if num_samples > 1 else axes[j]
                    ax.imshow(x0_predictions[t][i].cpu().numpy().transpose(1, 2, 0))
                    ax.axis('off')
                    if i == 0:
                        ax.set_title(f"t={t}")
                
                # Plot final sample
                ax = axes[i, 9] if num_samples > 1 else axes[9]
                ax.imshow(samples[i].cpu().numpy().transpose(1, 2, 0))
                ax.axis('off')
                if i == 0:
                    ax.set_title("Final")
                
                # Plot ground truth
                ax = axes[i, 10] if num_samples > 1 else axes[10]
                ax.imshow(gt_images[i].cpu().numpy().transpose(1, 2, 0))
                ax.axis('off')
                if i == 0:
                    ax.set_title("GT")
                
                # Add row label
                if num_samples > 1:
                    axes[i, 0].set_ylabel(f"Sample {i+1}", rotation=0, labelpad=40, fontsize=12)

            plt.suptitle(f"Epoch {epoch} Denoising Process with Ground Truth Reconstruction")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(self.log_directory, f"epoch_{epoch}_denoising_table.png"))
            self.writer.add_figure("visualization/denoising_table", fig, epoch)
            plt.close()
        
        self.model.train()
        print(f"Visualized denoising reconstruction of ground truth at epoch {epoch}")