"""Quantitative evaluation metrics: MSE, PSNR, SSIM, InceptionV3 FID, and latency."""

import math
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import models


def compute_reconstruction_metrics(
    recons: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute Test MSE, PSNR (dB), and SSIM on [0, 1] tensors of shape (N, 1, 28, 28)."""
    recons = recons.clamp(0.0, 1.0).to(device)
    targets = targets.clamp(0.0, 1.0).to(device)

    per_img_mse = ((recons - targets) ** 2).mean(dim=(1, 2, 3))
    mse = per_img_mse.mean().item()
    psnr = (10.0 * torch.log10(1.0 / per_img_mse.clamp(min=1e-10))).mean().item()
    ssim = _compute_ssim_batch(recons, targets).mean().item()
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.unsqueeze(0).unsqueeze(0)


def _compute_ssim_batch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute per-image SSIM in [0, 1] using a 2D Gaussian window."""
    window = _gaussian_window(window_size, 1.5, img1.device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return ssim_map.mean(dim=(1, 2, 3))


class InceptionFeatureExtractor(nn.Module):
    """Pretrained ImageNet InceptionV3 pool3 (2048-d) feature extractor for FID."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.fc = nn.Identity()
        inception.eval()
        for p in inception.parameters():
            p.requires_grad_(False)
        self.model = inception.to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        # Upsample [0, 1] grayscale -> 299x299 RGB and apply ImageNet normalization expected by Inception3
        x_rgb = x_unit.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        x_resized = F.interpolate(x_rgb, size=(299, 299), mode="bilinear", align_corners=False)
        x_norm = (x_resized - self.mean) / self.std
        return self.model(x_norm)


def _extract_features(extractor: InceptionFeatureExtractor, images: torch.Tensor, device: torch.device, batch_size: int = 128) -> np.ndarray:
    feats = []
    for i in range(0, images.size(0), batch_size):
        batch = images[i : i + batch_size].to(device)
        feats.append(extractor(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    extractor: InceptionFeatureExtractor,
    device: torch.device,
) -> float:
    """Compute Frechet Inception Distance between real and generated [0, 1] image tensors."""
    real_feats = _extract_features(extractor, real_images, device)
    fake_feats = _extract_features(extractor, fake_images, device)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def measure_inference_latency(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
) -> Tuple[float, float]:
    """Measure mean and std inference latency in milliseconds per batch of 64."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings_ms)
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def evaluate_autoencoder_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_vae: bool = False,
) -> Dict[str, Any]:
    """Collect reconstructions, metrics, and latent sparsity stats on Test Set."""
    model.eval()
    all_x, all_recon, all_z = [], [], []
    for x, _ in test_loader:
        x = x.to(device)
        if is_vae:
            mu, _ = model.encode_params(x)
            recon = model.decode(mu)
            z = mu
        else:
            recon, z = model(x)
        all_x.append(x.cpu())
        all_recon.append(recon.cpu())
        all_z.append(z.cpu())

    x_cat = torch.cat(all_x, dim=0)
    recon_cat = torch.cat(all_recon, dim=0)
    z_cat = torch.cat(all_z, dim=0)
    metrics = compute_reconstruction_metrics(recon_cat, x_cat, device)
    metrics["latent_l1_mean"] = z_cat.abs().sum(dim=1).mean().item()
    metrics["latent_zero_frac"] = (z_cat.abs() < 1e-3).float().mean().item()
    return {"metrics": metrics, "x": x_cat, "recon": recon_cat, "z": z_cat}
