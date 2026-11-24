"""Adversarial generative models: Vanilla MLP GAN and Deep Convolutional GAN (DCGAN)."""

import torch
import torch.nn as nn


def init_dcgan_weights(module: nn.Module) -> None:
    """Initialize Conv/BatchNorm/Linear weights per Radford et al. (2015)."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class VanillaGenerator(nn.Module):
    """MLP Generator mapping z in R^64 -> [-1, 1]^(1x28x28)."""

    def __init__(self, latent_dim: int = 64, output_dim: int = 784) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class VanillaDiscriminator(nn.Module):
    """MLP Discriminator scoring images in [-1, 1]^(1x28x28)."""

    def __init__(self, input_dim: int = 784) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1)).squeeze(1)


class DCGANGenerator(nn.Module):
    """Convolutional Generator mapping z in R^100 -> [-1, 1]^(1x28x28)."""

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(init_dcgan_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Strided convolutional Discriminator for 1x28x28 images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(256 * 4 * 4, 1)
        self.apply(init_dcgan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h.view(h.size(0), -1)).squeeze(1)
