"""MLP Classifier and Frozen Latent Decoder for Section 2.1 limitation proof."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """Discriminative MLP Classifier structurally matching a 784->256->64 encoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return discriminative latent features z_MLP in R^64 prior to classification."""
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mlp = self.extract_features(x)
        return self.fc3(z_mlp)

    @torch.no_grad()
    def project_from_class_onehot(self, y_onehot: torch.Tensor) -> torch.Tensor:
        """Stage 2: Top-down transpose projection from one-hot class vector y_c to pixel space."""
        z_hat = F.relu(y_onehot @ self.fc3.weight)
        h1_hat = F.relu(z_hat @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def project_from_latent(self, z_mlp: torch.Tensor) -> torch.Tensor:
        """Stage 3: Direct transpose projection from extracted z_MLP to pixel space."""
        h1_hat = F.relu(z_mlp @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def weight_orthogonality_diagnostics(self) -> Dict[str, float]:
        """Measure non-orthogonality ||W^T W - I||_F / ||I||_F and condition ratios."""
        diagnostics: Dict[str, float] = {}
        for name, layer in [("W1", self.fc1), ("W2", self.fc2), ("W3", self.fc3)]:
            w = layer.weight  # shape (out_dim, in_dim)
            gram = w.T @ w
            eye = torch.eye(gram.size(0), device=w.device, dtype=w.dtype)
            rel_err = (torch.linalg.norm(gram - eye) / torch.linalg.norm(eye)).item()
            diagnostics[f"{name}_orthogonality_error"] = round(rel_err, 4)
        return diagnostics


class FrozenLatentDecoder(nn.Module):
    """Separate MLP decoder (64->256->784) trained on frozen z_MLP features."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)
