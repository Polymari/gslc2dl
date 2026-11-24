"""Section 2.1 Empirical Proof: Theoretical & Empirical Limits of MLP Classifiers."""

import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier


def _evaluate_classifier(model: MLPClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_mlp_proof(
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute Stages 1-5 of Section 2.1 MLP Classifier limitation proof."""
    print("\n=== [Section 2.1] MLP Classifier Limitation Proof ===")
    classifier = MLPClassifier(latent_dim=cfg.ae_latent_dim, num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.mlp_epochs)
    criterion = nn.CrossEntropyLoss()

    clf_losses = []
    best_val_acc = 0.0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, cfg.mlp_epochs + 1):
        classifier.train()
        running_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(loaders["train"].dataset)
        val_acc = _evaluate_classifier(classifier, loaders["val"], device)
        clf_losses.append(epoch_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        if epoch % 5 == 0 or epoch == cfg.mlp_epochs:
            print(f"  [MLP Clf] Epoch {epoch:02d}/{cfg.mlp_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if best_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_acc = _evaluate_classifier(classifier, loaders["test"], device)
    clf_time = time.perf_counter() - t0
    print(f"  [MLP Clf] Final Test Accuracy: {test_acc*100:.2f}% (Target >= 88.0%) | Time: {clf_time:.1f}s")

    # Freeze classifier for Stages 2-5
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # Stage 2: Top-down transpose projection from one-hot class vectors
    one_hot = torch.eye(cfg.num_classes, device=device)
    class_transpose_imgs = classifier.project_from_class_onehot(one_hot).cpu()

    # Collect a fixed sample batch from test set for visual comparisons
    sample_x, sample_y = next(iter(loaders["test"]))
    sample_x = sample_x[:8].to(device)
    sample_y = sample_y[:8].to(device)
    with torch.no_grad():
        sample_z_mlp = classifier.extract_features(sample_x)
        sample_latent_transpose = classifier.project_from_latent(sample_z_mlp).cpu()
        sample_onehot_transpose = classifier.project_from_class_onehot(
            F.one_hot(sample_y, num_classes=cfg.num_classes).float()
        ).cpu()

    # Stage 4: Train Frozen Latent Decoder (64 -> 256 -> 784) on frozen z_MLP
    decoder = FrozenLatentDecoder(latent_dim=cfg.ae_latent_dim).to(device)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    dec_losses = []
    for epoch in range(1, cfg.frozen_decoder_epochs + 1):
        decoder.train()
        running_loss = 0.0
        for x, _ in loaders["train"]:
            x = x.to(device)
            with torch.no_grad():
                z_mlp = classifier.extract_features(x)
            dec_opt.zero_grad()
            recon = decoder(z_mlp)
            loss = F.mse_loss(recon, x)
            loss.backward()
            dec_opt.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(loaders["train"].dataset)
        dec_losses.append(epoch_loss)
        if epoch % 10 == 0 or epoch == cfg.frozen_decoder_epochs:
            print(f"  [Frozen Decoder] Epoch {epoch:02d}/{cfg.frozen_decoder_epochs} | MSE Loss: {epoch_loss:.5f}")

    decoder.eval()
    # Extract all test z_MLP for Stage 5 stats and decoder test MSE
    all_z_test = []
    all_recons_frozen = []
    all_targets = []
    with torch.no_grad():
        for x, _ in loaders["test"]:
            x = x.to(device)
            z_mlp = classifier.extract_features(x)
            recon = decoder(z_mlp)
            all_z_test.append(z_mlp.cpu())
            all_recons_frozen.append(recon.cpu())
            all_targets.append(x.cpu())

    z_test_cat = torch.cat(all_z_test, dim=0)
    recons_frozen_cat = torch.cat(all_recons_frozen, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    # Stage 5: Prior sampling N(mu_z, diag(sigma_z^2)) on MLP latent space
    mu_z = z_test_cat.mean(dim=0, keepdim=True)
    sigma_z = z_test_cat.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_rand = mu_z + sigma_z * torch.randn(16, cfg.ae_latent_dim)
    with torch.no_grad():
        prior_samples = decoder(z_rand.to(device)).cpu()
        sample_frozen_recon = decoder(sample_z_mlp).cpu()

    # Latent sparsity / active dimension statistics of z_MLP
    zero_frac = (z_test_cat == 0).float().mean().item()

    ortho_diag = classifier.weight_orthogonality_diagnostics()

    torch.save(classifier.state_dict(), cfg.checkpoint_dir / "mlp_classifier.pt")
    torch.save(decoder.state_dict(), cfg.checkpoint_dir / "frozen_latent_decoder.pt")

    return {
        "classifier": classifier,
        "frozen_decoder": decoder,
        "test_accuracy": test_acc,
        "val_accuracy": best_val_acc,
        "clf_losses": clf_losses,
        "frozen_dec_losses": dec_losses,
        "orthogonality": ortho_diag,
        "z_mlp_zero_fraction": zero_frac,
        "z_mlp_mu_norm": mu_z.norm().item(),
        "z_mlp_sigma_mean": sigma_z.mean().item(),
        "recons_frozen_all": recons_frozen_cat,
        "targets_all": targets_cat,
        "sample_x": sample_x.cpu(),
        "sample_y": sample_y.cpu(),
        "sample_onehot_transpose": sample_onehot_transpose,
        "sample_latent_transpose": sample_latent_transpose,
        "sample_frozen_recon": sample_frozen_recon,
        "class_transpose_imgs": class_transpose_imgs,
        "prior_samples": prior_samples,
    }
