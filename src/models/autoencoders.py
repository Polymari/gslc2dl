"""Autoencoder family: Vanilla AE, Sparse AE (L1), and Variational Autoencoder (VAE)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp_decoder(latent_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid(),
    )


class VanillaAutoencoder(nn.Module):
    """Deterministic MLP Autoencoder (784 -> 256 -> 64 -> 256 -> 784)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = _build_mlp_decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class SparseAutoencoder(nn.Module):
    """Sparse MLP Autoencoder with ReLU bottleneck and L1 sparsity penalty."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = _build_mlp_decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class VariationalAutoencoder(nn.Module):
    """Probabilistic Variational Autoencoder with diagonal Gaussian latent space."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _build_mlp_decoder(latent_dim, hidden_dim, input_dim)

    def encode_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent mean mu(x) for interpolation and test-time encoding."""
        mu, _ = self.encode_params(x)
        return mu

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def elbo_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample ELBO loss = BCE_sum + KL_divergence."""
        bsz = x.size(0)
        bce = F.binary_cross_entropy(recon.view(bsz, -1), x.view(bsz, -1), reduction="sum") / bsz
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / bsz
        return bce + kl, bce, kl
