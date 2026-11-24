"""High-resolution generation of the 5 mandatory comparative figures (300 DPI)."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import FASHION_MNIST_CLASSES_ID


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig1_reconstructions(
    gt: torch.Tensor,
    labels: torch.Tensor,
    vanilla_recon: torch.Tensor,
    sparse_recon: torch.Tensor,
    vae_recon: torch.Tensor,
    out_path: Path,
) -> None:
    """Figur 1: Side-by-side Test Set reconstruction grid (8 samples x 4 rows)."""
    rows = [
        ("Citra Asli\n(Ground Truth)", gt),
        ("Vanilla\nAutoencoder", vanilla_recon),
        ("Sparse AE\n(L1 Penalty)", sparse_recon),
        ("Variational\nAE (VAE)", vae_recon),
    ]
    n_cols = gt.size(0)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(13.5, 7.2))
    for r_idx, (row_title, imgs) in enumerate(rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx, 0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                cls_name = FASHION_MNIST_CLASSES_ID[int(labels[c_idx])]
                ax.set_title(cls_name, fontsize=9, fontweight="bold", pad=5)
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=9, fontweight="bold", rotation=90, labelpad=10)
    fig.suptitle(
        "Figur 1: Perbandingan Rekonstruksi Side-by-Side pada Test Set Fashion-MNIST",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, out_path)


def plot_fig2_random_sampling(
    samples_dict: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    """Figur 2: 4x4 random generative sampling panels for VAE, Vanilla GAN, DCGAN, DDPM."""
    panels = [
        ("Panel A: VAE (z ~ N(0, I))", samples_dict["vae"][:16]),
        ("Panel B: Vanilla GAN (z ~ N(0, I))", samples_dict["vanilla_gan"][:16]),
        ("Panel C: DCGAN (z ~ N(0, I))", samples_dict["dcgan"][:16]),
        ("Panel D: DDPM (Reverse Denoising x_T -> x_0)", samples_dict["ddpm"][:16]),
    ]
    fig = plt.figure(figsize=(12.5, 11.5))
    outer = fig.add_gridspec(2, 2, wspace=0.14, hspace=0.18)

    for idx, (title, imgs) in enumerate(panels):
        r, c = divmod(idx, 2)
        inner = outer[r, c].subgridspec(4, 4, wspace=0.04, hspace=0.04)
        ax_title = fig.add_subplot(outer[r, c])
        ax_title.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax_title.axis("off")
        for k in range(16):
            ax = fig.add_subplot(inner[k // 4, k % 4])
            ax.imshow(imgs[k, 0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Figur 2: Komparasi Sampel Generatif Acak (Kisi 4x4 per Model)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    _save_fig(fig, out_path)


def plot_fig3_mlp_proof(
    mlp_res: Dict[str, Any],
    vanilla_ae_recon: torch.Tensor,
    out_path: Path,
) -> None:
    """Figur 3: 4-panel empirical proof of MLP Classifier weight & latent limitations."""
    fig = plt.figure(figsize=(14.0, 10.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 2.0, 2.0, 1.35], hspace=0.52)

    # Panel 1: Ground Truth (8 images)
    gs1 = gs[0].subgridspec(1, 8, wspace=0.08)
    ax1_hdr = fig.add_subplot(gs[0])
    ax1_hdr.set_title("(1) Citra Asli Test Set (Ground Truth)", fontsize=10.5, fontweight="bold", loc="left", pad=22)
    ax1_hdr.axis("off")
    for i in range(8):
        ax = fig.add_subplot(gs1[0, i])
        ax.imshow(mlp_res["sample_x"][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(FASHION_MNIST_CLASSES_ID[int(mlp_res["sample_y"][i])], fontsize=8)
        ax.axis("off")

    # Panel 2: Direct Transpose W^T (from y_c and from z_MLP)
    gs2 = gs[1].subgridspec(2, 8, wspace=0.08, hspace=0.12)
    ax2_hdr = fig.add_subplot(gs[1])
    ax2_hdr.set_title(
        "(2) Proyeksi Balik Langsung Transpos Bobot W^T (Atas: dari One-Hot y_c | Bawah: dari Latent z_MLP)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax2_hdr.axis("off")
    for i in range(8):
        ax_top = fig.add_subplot(gs2[0, i])
        ax_top.imshow(mlp_res["sample_onehot_transpose"][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_top.axis("off")
        ax_bot = fig.add_subplot(gs2[1, i])
        ax_bot.imshow(mlp_res["sample_latent_transpose"][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_bot.axis("off")

    # Panel 3: Frozen Latent Decoder (z_MLP) vs Vanilla AE (z_AE)
    gs3 = gs[2].subgridspec(2, 8, wspace=0.08, hspace=0.12)
    ax3_hdr = fig.add_subplot(gs[2])
    ax3_hdr.set_title(
        "(3) Rekonstruksi Frozen Latent Decoder z_MLP (Atas) vs. Vanilla Autoencoder z_AE (Bawah)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax3_hdr.axis("off")
    for i in range(8):
        ax_top = fig.add_subplot(gs3[0, i])
        ax_top.imshow(mlp_res["sample_frozen_recon"][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_top.axis("off")
        ax_bot = fig.add_subplot(gs3[1, i])
        ax_bot.imshow(vanilla_ae_recon[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_bot.axis("off")

    # Panel 4: Random prior sampling z_rand ~ N(mu_z, diag(sigma_z^2))
    gs4 = gs[3].subgridspec(1, 12, wspace=0.06)
    ax4_hdr = fig.add_subplot(gs[3])
    ax4_hdr.set_title(
        "(4) Uji Random Prior Sampling z_rand ~ N(mu_z, diag(sigma_z^2)) pada Ruang Laten MLP Classifier",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax4_hdr.axis("off")
    for i in range(12):
        ax = fig.add_subplot(gs4[0, i])
        ax.imshow(mlp_res["prior_samples"][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(
        "Figur 3: Bukti Empiris Batasan Matriks Bobot (W^T) dan Ruang Laten MLP Classifier",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    _save_fig(fig, out_path)


def plot_fig4_latent_interpolation(
    interp_rows: List[tuple[str, torch.Tensor]],
    alphas: List[float],
    out_path: Path,
) -> None:
    """Figur 4: Linear latent interpolation across Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN."""
    n_rows = len(interp_rows)
    n_cols = len(alphas)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 8.8))

    for r_idx, (model_label, imgs) in enumerate(interp_rows):
        for c_idx, alpha in enumerate(alphas):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx, 0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"alpha = {alpha:.1f}", fontsize=10, fontweight="bold", pad=6)
            if c_idx == 0:
                ax.set_ylabel(model_label, fontsize=9.5, fontweight="bold", rotation=90, labelpad=10)

    fig.suptitle(
        "Figur 4: Interpolasi Ruang Laten Linier z(alpha) = (1 - alpha) z_A + alpha z_B Antarmodel",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, out_path)


def plot_fig5_losses_and_ddpm_trajectory(
    histories: Dict[str, Any],
    ddpm_trajectory: Dict[int, torch.Tensor],
    out_path: Path,
) -> None:
    """Figur 5: Training loss curves for all 6 models + step-by-step DDPM denoising trajectory."""
    fig = plt.figure(figsize=(14.0, 10.8))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.08], hspace=0.46)

    # Row 1: Autoencoders (Vanilla AE, Sparse AE, VAE)
    top_gs = outer[0].subgridspec(1, 3, wspace=0.25)
    for idx, (key, title, ylabel) in enumerate([
        ("vanilla_ae", "Vanilla Autoencoder (MSE)", "MSE Loss"),
        ("sparse_ae", "Sparse Autoencoder (MSE + L1)", "Total Loss"),
        ("vae", "Variational Autoencoder (ELBO)", "ELBO Loss"),
    ]):
        ax = fig.add_subplot(top_gs[0, idx])
        tr = histories[key]["train_losses"]
        va = histories[key]["val_losses"]
        epochs = np.arange(1, len(tr) + 1)
        ax.plot(epochs, tr, label="Train", linewidth=1.8, color="#1f77b4")
        ax.plot(epochs, va, label="Val", linewidth=1.8, linestyle="--", color="#ff7f0e")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Row 2: Vanilla GAN, DCGAN, DDPM
    mid_gs = outer[1].subgridspec(1, 3, wspace=0.25)
    for idx, (key, title) in enumerate([
        ("vanilla_gan", "Vanilla GAN (G vs D Loss)"),
        ("dcgan", "DCGAN (G vs D Loss)"),
    ]):
        ax = fig.add_subplot(mid_gs[0, idx])
        g_l = histories[key]["g_losses"]
        d_l = histories[key]["d_losses"]
        epochs = np.arange(1, len(g_l) + 1)
        ax.plot(epochs, d_l, label="Discriminator (D)", linewidth=1.8, color="#d62728")
        ax.plot(epochs, g_l, label="Generator (G)", linewidth=1.8, color="#2ca02c")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("BCEWithLogits Loss", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax_ddpm = fig.add_subplot(mid_gs[0, 2])
    ddpm_l = histories["ddpm"]["train_losses"]
    epochs = np.arange(1, len(ddpm_l) + 1)
    ax_ddpm.plot(epochs, ddpm_l, label="Noise MSE", linewidth=1.8, color="#9467bd")
    ax_ddpm.set_title("DDPM (Noise Prediction MSE)", fontsize=10, fontweight="bold")
    ax_ddpm.set_xlabel("Epoch", fontsize=9)
    ax_ddpm.set_ylabel("MSE(eps, eps_theta)", fontsize=9)
    ax_ddpm.grid(alpha=0.3)
    ax_ddpm.legend(fontsize=8)

    # Row 3: DDPM Reverse Denoising Trajectory (2 samples x sorted timesteps T -> 0)
    steps_desc = sorted(ddpm_trajectory.keys(), reverse=True)
    n_steps = len(steps_desc)
    bot_gs = outer[2].subgridspec(2, n_steps, wspace=0.05, hspace=0.08)
    ax_traj_hdr = fig.add_subplot(outer[2])
    ax_traj_hdr.set_title(
        "Sekuens Reverse Diffusion DDPM Langkah demi Langkah (x_T -> x_0)",
        fontsize=10.5,
        fontweight="bold",
        pad=16,
    )
    ax_traj_hdr.axis("off")

    for r in range(2):
        for c, t_val in enumerate(steps_desc):
            ax = fig.add_subplot(bot_gs[r, c])
            ax.imshow(ddpm_trajectory[t_val][r, 0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t_val}", fontsize=8.5, fontweight="bold", pad=4)

    fig.suptitle(
        "Figur 5: Kurva Training Loss Keenam Model dan Trajektori Reverse Denoising DDPM",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    _save_fig(fig, out_path)
