"""Denoising Diffusion Probabilistic Model (DDPM) with 28x28 time-conditioned UNet and EMA."""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding projected through a 2-layer SiLU MLP."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual convolutional block conditioned on diffusion timestep embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight multi-head spatial self-attention for 7x7 bottleneck feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return x + self.proj(out)


class UNet28(nn.Module):
    """Compact 28x28 UNet predicting additive Gaussian noise epsilon_theta(x_t, t)."""

    def __init__(self, in_channels: int = 1, base_ch: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        emb_dim = time_dim * 2
        self.time_mlp = SinusoidalTimeEmbedding(time_dim)

        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, emb_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)  # 28 -> 14

        self.down2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 14 -> 7

        self.mid1 = ResBlock(base_ch * 2, base_ch * 4, emb_dim)
        self.attn = SelfAttention2d(base_ch * 4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.dec1 = ResBlock(base_ch * 2, base_ch, emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h0 = self.init_conv(x)
        h1 = self.down1(h0, t_emb)          # 28x28, C
        h2 = self.down2(self.pool1(h1), t_emb)  # 14x14, 2C
        m = self.mid2(self.attn(self.mid1(self.pool2(h2), t_emb)), t_emb)  # 7x7, 4C

        u2 = self.up2(m)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)
        return self.out_conv(u1)


class EMA:
    """Exponential Moving Average tracker for stable diffusion sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)


class GaussianDiffusion(nn.Module):
    """DDPM forward noising and ancestral reverse denoising sampler."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        c1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        c2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return c1 * x0 + c2 * noise

    def p_losses(self, x0_unit: torch.Tensor) -> torch.Tensor:
        """Compute supervised noise-prediction MSE loss on images scaled to [-1, 1]."""
        x0 = x0_unit * 2.0 - 1.0
        bsz = x0.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        denoiser: Optional[nn.Module] = None,
        record_steps: Optional[List[int]] = None,
        use_amp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run T-step reverse diffusion from x_T ~ N(0, I) -> x_0 in [0, 1]."""
        net = denoiser if denoiser is not None else self.model
        net.eval()
        x = torch.randn(batch_size, 1, 28, 28, device=device)
        trajectory: Dict[int, torch.Tensor] = {}
        record_set = set(record_steps or [])

        if self.timesteps - 1 in record_set:
            trajectory[self.timesteps - 1] = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()

        amp_enabled = use_amp and device.type == "cuda"
        for step in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                eps_pred = net(x, t_batch).float()
            beta_t = self.betas[step]
            sqrt_recip_alpha = self.sqrt_recip_alphas[step]
            sqrt_one_minus_abar = self.sqrt_one_minus_alphas_cumprod[step]

            mean = sqrt_recip_alpha * (x - (beta_t / sqrt_one_minus_abar) * eps_pred)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(self.posterior_variance[step]) * noise
            else:
                x = mean

            if step in record_set:
                trajectory[step] = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()

        x_unit = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        return x_unit, trajectory
