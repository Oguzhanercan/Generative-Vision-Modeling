import torch

class DDPM_Noise:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.alphas_cumprod, self.betas, self.alphas = self.precompute_alphas(args.ddpm.steps)
        self.alphas_cumprod = self.alphas_cumprod.to(args.training.device)
        self.betas = self.betas.to(args.training.device)
        self.alphas = self.alphas.to(args.training.device)

    def precompute_alphas(self, num_steps):
        betas = torch.linspace(self.args.ddpm.beta_start, self.args.ddpm.beta_end, num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod, betas, alphas

    def forward(self, x, t):
        expected_shape = (x.shape[0], self.args.data.data_channel, self.args.data.image_size, self.args.data.image_size)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input x shape mismatch: expected {expected_shape}, got {x.shape}"
            )
        e = torch.randn_like(x)
        if x.shape[2:] != (self.args.data.image_size, self.args.data.image_size):
            raise ValueError(
                f"Input image size mismatch: expected ({self.args.data.image_size}, {self.args.data.image_size}), "
                f"got {x.shape[2:]}"
            )
        alphas_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = alphas_t**0.5 * x + (1 - alphas_t) ** 0.5 * e
        e_pred = self.model(x_t, t)
        if e_pred.shape != e.shape:
            raise ValueError(
                f"Predicted noise shape mismatch: expected {e.shape}, got {e_pred.shape}"
            )
        return e_pred, e

    def predict_x0(self, x_t, t, e_pred):
        if x_t.shape != e_pred.shape:
            raise ValueError(
                f"x_t and e_pred shape mismatch: x_t {x_t.shape}, e_pred {e_pred.shape}"
            )
        alphas_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_0 = (x_t - (1 - alphas_t) ** 0.5 * e_pred) / alphas_t ** 0.5
        return x_0

    def reverse(self, x_t, t):
        expected_shape = (x_t.shape[0], self.args.data.data_channel, self.args.data.image_size, self.args.data.image_size)
        if x_t.shape != expected_shape:
            raise ValueError(
                f"reverse x_t shape mismatch: expected {expected_shape}, got {x_t.shape}"
            )
        e_pred = self.model(x_t, t)
        if e_pred.shape != x_t.shape:
            raise ValueError(
                f"reverse e_pred shape mismatch: expected {x_t.shape}, got {e_pred.shape}"
            )
        x_0 = self.predict_x0(x_t, t, e_pred)

        t_prev = torch.clamp(t - 1, min=0)
        alphas_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        alphas_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        mu = (alphas_t_prev ** 0.5 * betas_t / (1 - alphas_t)) * x_0 + \
             (alpha_t ** 0.5 * (1 - alphas_t_prev) / (1 - alphas_t)) * x_t

        sigma = ((1 - alphas_t_prev) / (1 - alphas_t) * betas_t) ** 0.5
        z = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        x_t_1 = mu + sigma * z
        return x_t_1

    def sample(self, x_start):
        self.model.eval()
        with torch.no_grad():
            x_t = x_start
            x0_predictions = {}
            timesteps_to_log = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]
            
            for t in range(self.args.ddpm.steps - 1, -1, -1):
                t_tensor = torch.full((x_start.shape[0],), t, device=self.args.training.device, dtype=torch.long)
                e_pred = self.model(x_t, t_tensor)
                x_0 = self.predict_x0(x_t, t_tensor, e_pred)
                
                if t in timesteps_to_log:
                    x0_predictions[t] = x_0.clamp(-1, 1)
                
                if t > 0:
                    x_t = self.reverse(x_t, t_tensor)
            
            x_t = x_t.clamp(-1, 1)
            return x_t, x0_predictions

    def add_noise(self, x, t):
        """
        Apply forward diffusion to an image up to timestep t.
        Args:
            x: Input image tensor (batch_size, channels, height, width).
            t: Target timestep (scalar or tensor of shape (batch_size,)).
        Returns:
            Noisy image at timestep t.
        """
        expected_shape = (x.shape[0], self.args.data.data_channel, self.args.data.image_size, self.args.data.image_size)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input x shape mismatch: expected {expected_shape}, got {x.shape}"
            )
        e = torch.randn_like(x)
        alphas_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = alphas_t**0.5 * x + (1 - alphas_t) ** 0.5 * e
        return x_t