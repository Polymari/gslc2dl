"""Pre-download Fashion-MNIST and InceptionV3 weights needed for offline runs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchvision import datasets, models
from src.config import ExperimentConfig


def main() -> None:
    cfg = ExperimentConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading Fashion-MNIST...")
    datasets.FashionMNIST(root=str(cfg.data_dir), train=True, download=True)
    datasets.FashionMNIST(root=str(cfg.data_dir), train=False, download=True)
    print("Downloading InceptionV3 pretrained weights...")
    models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    print("All assets downloaded successfully.")


if __name__ == "__main__":
    main()
