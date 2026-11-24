"""DDPM training pipeline with EMA weights and denoising trajectory capture."""

import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.ddpm import EMA, GaussianDiffusion, UNet28


def train_ddpm(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    """Train 28x28 UNet DDPM using noise-prediction MSE loss and EMA."""
    print("\n=== Training Diffusion Family (DDPM) ===")
    unet = UNet28(in_channels=cfg.in_channels, base_ch=64, time_dim=128).to(device)
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=cfg.ddpm_timesteps,
        beta_start=cfg.ddpm_beta_start,
        beta_end=cfg.ddpm_beta_end,
    ).to(device)
    ema = EMA(unet, decay=cfg.ddpm_ema_decay)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-4)

    train_losses = []
    epoch_times = []

    amp_enabled = device.type == "cuda"
    for epoch in range(1, cfg.ddpm_epochs + 1):
        unet.train()
        running_loss = 0.0
        total_n = 0
        t0 = time.perf_counter()

        for x, _ in loaders["train"]:
            x = x.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss = diffusion.p_losses(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            ema.update(unet)
            running_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        ep_loss = running_loss / total_n
        train_losses.append(ep_loss)

        if epoch % 5 == 0 or epoch == cfg.ddpm_epochs:
            print(f"  [DDPM] Epoch {epoch:02d}/{cfg.ddpm_epochs} | Noise MSE Loss: {ep_loss:.5f} | {epoch_times[-1]:.1f}s", flush=True)

    torch.save(unet.state_dict(), cfg.checkpoint_dir / "ddpm_unet.pt")
    torch.save(ema.shadow.state_dict(), cfg.checkpoint_dir / "ddpm_ema.pt")

    return {
        "diffusion": diffusion,
        "ema_model": ema.shadow,
        "train_losses": train_losses,
        "mean_epoch_time": sum(epoch_times) / len(epoch_times),
    }
