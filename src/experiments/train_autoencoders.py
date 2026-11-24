"""Unified training pipeline for Vanilla AE, Sparse AE, and VAE."""

import time
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.autoencoders import SparseAutoencoder, VanillaAutoencoder, VariationalAutoencoder


def _run_ae_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, _ in loader:
            x = x.to(device)
            if is_train:
                optimizer.zero_grad()
            loss = loss_fn(model, x)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
    return total_loss / total_n


def _train_single_ae(
    name: str,
    model: nn.Module,
    loss_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    epochs: int,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    ckpt_path: str,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    epoch_times = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss = _run_ae_epoch(model, loaders["train"], loss_fn, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        va_loss = _run_ae_epoch(model, loaders["val"], loss_fn, None, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{name}] Epoch {epoch:02d}/{epochs} | Train Loss: {tr_loss:.5f} | Val Loss: {va_loss:.5f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), ckpt_path)
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mean_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def sparse_ae_loss(cfg: ExperimentConfig, model: SparseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Compute MSE reconstruction loss plus mean L1 sparsity penalty on z."""
    recon, z = model(x)
    return F.mse_loss(recon, x) + cfg.sparse_l1_lambda * z.abs().mean()


def train_sparse_ae_only(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    """Train only Sparse Autoencoder and save its checkpoint."""
    model = SparseAutoencoder(latent_dim=cfg.sparse_latent_dim).to(device)
    return _train_single_ae(
        name="Sparse AE",
        model=model,
        loss_fn=lambda m, x: sparse_ae_loss(cfg, m, x),
        epochs=cfg.sparse_ae_epochs,
        loaders=loaders,
        device=device,
        ckpt_path=str(cfg.checkpoint_dir / "sparse_ae.pt"),
    )


def train_all_autoencoders(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Train Vanilla AE, Sparse AE, and VAE under standardized protocol."""
    print("\n=== Training Autoencoder Family (Vanilla AE, Sparse AE, VAE) ===")

    vanilla_ae = VanillaAutoencoder(latent_dim=cfg.ae_latent_dim).to(device)
    sparse_ae = SparseAutoencoder(latent_dim=cfg.sparse_latent_dim).to(device)
    vae = VariationalAutoencoder(latent_dim=cfg.vae_latent_dim).to(device)

    def vanilla_loss(m: VanillaAutoencoder, x: torch.Tensor) -> torch.Tensor:
        recon, _ = m(x)
        return F.mse_loss(recon, x)

    def vae_loss(m: VariationalAutoencoder, x: torch.Tensor) -> torch.Tensor:
        recon, mu, logvar = m(x)
        loss, _, _ = VariationalAutoencoder.elbo_loss(recon, x, mu, logvar)
        return loss

    results: Dict[str, Dict[str, Any]] = {}
    specs: Tuple[Tuple[str, str, nn.Module, Callable[[Any, torch.Tensor], torch.Tensor], int], ...] = (
        ("vanilla_ae", "Vanilla AE", vanilla_ae, vanilla_loss, cfg.ae_epochs),
        ("sparse_ae", "Sparse AE", sparse_ae, lambda m, x: sparse_ae_loss(cfg, m, x), cfg.sparse_ae_epochs),
        ("vae", "VAE", vae, vae_loss, cfg.vae_epochs),
    )
    for key, label, model, fn, epochs in specs:
        results[key] = _train_single_ae(
            name=label,
            model=model,
            loss_fn=fn,
            epochs=epochs,
            loaders=loaders,
            device=device,
            ckpt_path=str(cfg.checkpoint_dir / f"{key}.pt"),
        )
    return results
