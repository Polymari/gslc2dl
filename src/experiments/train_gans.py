"""Unified minimax training pipeline for Vanilla GAN and DCGAN."""

import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.gans import (
    DCGANDiscriminator,
    DCGANGenerator,
    VanillaDiscriminator,
    VanillaGenerator,
)


def _train_single_gan(
    name: str,
    generator: nn.Module,
    discriminator: nn.Module,
    latent_dim: int,
    epochs: int,
    train_loader: DataLoader,
    device: torch.device,
    ckpt_prefix: str,
    real_label_val: float = 1.0,
) -> Dict[str, Any]:
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    g_losses, d_losses = [], []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        running_g, running_d = 0.0, 0.0
        total_n = 0
        t0 = time.perf_counter()

        for real_x, _ in train_loader:
            bsz = real_x.size(0)
            real_x = real_x.to(device) * 2.0 - 1.0  # Scale [0, 1] -> [-1, 1]
            real_d_targets = torch.full((bsz,), real_label_val, device=device)
            real_g_targets = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            # 1. Update Discriminator: max log(D(x)) + log(1 - D(G(z)))
            opt_d.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_x = generator(z).detach()
            d_loss_real = criterion(discriminator(real_x), real_d_targets)
            d_loss_fake = criterion(discriminator(fake_x), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # 2. Update Generator: max log(D(G(z)))
            opt_g.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            gen_x = generator(z)
            g_loss = criterion(discriminator(gen_x), real_g_targets)
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item() * bsz
            running_g += g_loss.item() * bsz
            total_n += bsz

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        ep_g = running_g / total_n
        ep_d = running_d / total_n
        g_losses.append(ep_g)
        d_losses.append(ep_d)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{name}] Epoch {epoch:02d}/{epochs} | D_loss: {ep_d:.4f} | G_loss: {ep_g:.4f}", flush=True)

    torch.save(generator.state_dict(), f"{ckpt_prefix}_G.pt")
    torch.save(discriminator.state_dict(), f"{ckpt_prefix}_D.pt")
    return {
        "generator": generator,
        "discriminator": discriminator,
        "g_losses": g_losses,
        "d_losses": d_losses,
        "mean_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def train_dcgan_only(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    """Train DCGAN with one-sided label smoothing (0.9) for stabilized minimax equilibrium."""
    dcgan_g = DCGANGenerator(latent_dim=cfg.dcgan_latent_dim).to(device)
    dcgan_d = DCGANDiscriminator().to(device)
    return _train_single_gan(
        name="DCGAN",
        generator=dcgan_g,
        discriminator=dcgan_d,
        latent_dim=cfg.dcgan_latent_dim,
        epochs=cfg.dcgan_epochs,
        train_loader=loaders["train"],
        device=device,
        ckpt_prefix=str(cfg.checkpoint_dir / "dcgan"),
        real_label_val=0.9,
    )


def train_all_gans(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Train Vanilla MLP GAN and DCGAN on Fashion-MNIST."""
    print("\n=== Training Adversarial Family (Vanilla GAN, DCGAN) ===")
    vanilla_g = VanillaGenerator(latent_dim=cfg.gan_latent_dim).to(device)
    vanilla_d = VanillaDiscriminator().to(device)

    results = {
        "vanilla_gan": _train_single_gan(
            name="Vanilla GAN",
            generator=vanilla_g,
            discriminator=vanilla_d,
            latent_dim=cfg.gan_latent_dim,
            epochs=cfg.gan_epochs,
            train_loader=loaders["train"],
            device=device,
            ckpt_prefix=str(cfg.checkpoint_dir / "vanilla_gan"),
            real_label_val=1.0,
        ),
        "dcgan": train_dcgan_only(cfg, loaders, device),
    }
    return results
