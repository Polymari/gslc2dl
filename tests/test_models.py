"""Fast unit tests verifying tensor shapes, losses, and mathematical invariants."""

import torch

from src.models.autoencoders import SparseAutoencoder, VanillaAutoencoder, VariationalAutoencoder
from src.models.ddpm import GaussianDiffusion, UNet28
from src.models.gans import DCGANDiscriminator, DCGANGenerator, VanillaDiscriminator, VanillaGenerator
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier


def test_mlp_classifier_and_projections() -> None:
    x = torch.rand(4, 1, 28, 28)
    clf = MLPClassifier()
    logits = clf(x)
    z = clf.extract_features(x)
    assert logits.shape == (4, 10)
    assert z.shape == (4, 64)
    assert clf.project_from_class_onehot(torch.eye(10)).shape == (10, 1, 28, 28)
    assert clf.project_from_latent(z).shape == (4, 1, 28, 28)
    dec = FrozenLatentDecoder()
    assert dec(z).shape == (4, 1, 28, 28)


def test_autoencoder_family_shapes() -> None:
    x = torch.rand(4, 1, 28, 28)
    for cls in (VanillaAutoencoder, SparseAutoencoder):
        m = cls(latent_dim=64)
        recon, z = m(x)
        assert recon.shape == x.shape
        assert z.shape == (4, 64)
        assert 0.0 <= recon.min().item() <= recon.max().item() <= 1.0

    vae = VariationalAutoencoder(latent_dim=64)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape and mu.shape == (4, 64) and logvar.shape == (4, 64)
    total, bce, kl = VariationalAutoencoder.elbo_loss(recon, x, mu, logvar)
    assert total.item() > 0 and bce.item() >= 0 and kl.item() >= 0


def test_gan_family_shapes() -> None:
    vg = VanillaGenerator(64)
    vd = VanillaDiscriminator()
    fake_v = vg(torch.randn(4, 64))
    assert fake_v.shape == (4, 1, 28, 28)
    assert vd(fake_v).shape == (4,)

    dg = DCGANGenerator(100)
    dd = DCGANDiscriminator()
    fake_d = dg(torch.randn(4, 100))
    assert fake_d.shape == (4, 1, 28, 28)
    assert dd(fake_d).shape == (4,)


def test_ddpm_unet_and_diffusion_step() -> None:
    unet = UNet28(in_channels=1, base_ch=32, time_dim=64)
    diffusion = GaussianDiffusion(unet, timesteps=20)
    x = torch.rand(2, 1, 28, 28)
    loss = diffusion.p_losses(x)
    assert torch.isfinite(loss) and loss.item() > 0
    samples, traj = diffusion.sample(2, torch.device("cpu"), record_steps=[19, 0])
    assert samples.shape == (2, 1, 28, 28)
    assert 0.0 <= samples.min().item() <= samples.max().item() <= 1.0
    assert set(traj.keys()) == {19, 0}
