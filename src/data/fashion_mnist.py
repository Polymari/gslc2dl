"""Standardized Fashion-MNIST loading and stratified 80/10/10 splitting."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from src.config import ExperimentConfig


def load_fashion_mnist_tensors(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load combined 70,000 Fashion-MNIST images in [0, 1] and integer labels."""
    train_raw = datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    test_raw = datasets.FashionMNIST(root=str(data_dir), train=False, download=True)

    images = torch.cat([train_raw.data, test_raw.data], dim=0).float() / 255.0
    images = images.unsqueeze(1)  # (70000, 1, 28, 28) in [0, 1]
    labels = torch.cat([train_raw.targets, test_raw.targets], dim=0).long()
    return images, labels


def create_dataloaders(cfg: ExperimentConfig) -> Dict[str, DataLoader]:
    """Build deterministic stratified Train (80%), Val (10%), and Test (10%) DataLoaders."""
    images, labels = load_fashion_mnist_tensors(cfg.data_dir)
    indices = np.arange(len(labels))
    labels_np = labels.numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(cfg.val_ratio + cfg.test_ratio),
        stratify=labels_np,
        random_state=cfg.seed,
    )
    relative_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=labels_np[temp_idx],
        random_state=cfg.seed,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    splits = {
        "train": (train_idx, True),
        "val": (val_idx, False),
        "test": (test_idx, False),
    }

    loaders: Dict[str, DataLoader] = {}
    for split_name, (idx, shuffle) in splits.items():
        ds = TensorDataset(images[idx], labels[idx])
        loaders[split_name] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if shuffle else None,
        )
    return loaders
