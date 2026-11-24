"""Central configuration for standardized experiments and report metadata."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters and paths shared across all models and evaluations."""

    seed: int = 42
    dataset_name: str = "FashionMNIST"
    image_size: int = 28
    in_channels: int = 1
    num_classes: int = 10

    # Stratified 80 / 10 / 10 split of 70,000 total images
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    batch_size: int = 128
    num_workers: int = 0

    # Latent dimensions (aligned with Section 2.1 Tahap 4 & Template Table)
    ae_latent_dim: int = 64
    sparse_latent_dim: int = 64
    sparse_l1_lambda: float = 5e-3
    vae_latent_dim: int = 64
    gan_latent_dim: int = 64
    dcgan_latent_dim: int = 100

    # Training epochs (Balanced Thorough Run on RTX 4060)
    mlp_epochs: int = 20
    frozen_decoder_epochs: int = 30
    ae_epochs: int = 30
    sparse_ae_epochs: int = 30
    vae_epochs: int = 30
    gan_epochs: int = 50
    dcgan_epochs: int = 50
    ddpm_epochs: int = 50

    # DDPM diffusion schedule
    ddpm_timesteps: int = 1000
    ddpm_beta_start: float = 1e-4
    ddpm_beta_end: float = 0.02
    ddpm_ema_decay: float = 0.999

    # Evaluation settings
    latency_batch_size: int = 64
    latency_warmup_runs: int = 10
    latency_measure_runs: int = 50
    fid_num_samples: int = 7000

    # Paths
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    checkpoint_dir: Path = root_dir / "checkpoints"
    results_dir: Path = root_dir / "results"
    figures_dir: Path = root_dir / "figures"


FASHION_MNIST_CLASSES_ID = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
