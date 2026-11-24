"""End-to-end resumable training, evaluation, and figure generation pipeline."""

import argparse
import json
from typing import Any, Dict

import torch

from src.config import ExperimentConfig
from src.data.fashion_mnist import create_dataloaders
from src.evaluation.metrics import (
    InceptionFeatureExtractor,
    compute_fid,
    compute_reconstruction_metrics,
    evaluate_autoencoder_on_test,
    measure_inference_latency,
)
from src.evaluation.visualize import (
    plot_fig1_reconstructions,
    plot_fig2_random_sampling,
    plot_fig3_mlp_proof,
    plot_fig4_latent_interpolation,
    plot_fig5_losses_and_ddpm_trajectory,
)
from src.experiments.mlp_proof import run_mlp_proof
from src.experiments.train_autoencoders import train_all_autoencoders, train_sparse_ae_only
from src.experiments.train_ddpm import train_ddpm
from src.experiments.train_gans import train_all_gans, train_dcgan_only
from src.models.autoencoders import SparseAutoencoder, VanillaAutoencoder, VariationalAutoencoder
from src.models.ddpm import GaussianDiffusion, UNet28
from src.models.gans import DCGANGenerator, VanillaGenerator
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier
from src.utils.seed import get_device, set_seed


def _select_distinct_test_samples(loaders: Dict[str, Any], num_classes_to_pick: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    picked_x, picked_y = [], []
    seen = set()
    for x, y in loaders["test"]:
        for i in range(x.size(0)):
            lbl = int(y[i])
            if lbl not in seen:
                seen.add(lbl)
                picked_x.append(x[i : i + 1])
                picked_y.append(y[i : i + 1])
                if len(seen) == num_classes_to_pick:
                    return torch.cat(picked_x, dim=0), torch.cat(picked_y, dim=0)
    raise RuntimeError("Could not find enough distinct classes in test loader.")


def _select_contrasting_pair(loaders: Dict[str, Any], class_a: int = 7, class_b: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    img_a, img_b = None, None
    for x, y in loaders["test"]:
        for i in range(x.size(0)):
            lbl = int(y[i])
            if lbl == class_a and img_a is None:
                img_a = x[i : i + 1]
            elif lbl == class_b and img_b is None:
                img_b = x[i : i + 1]
            if img_a is not None and img_b is not None:
                return img_a, img_b
    raise RuntimeError("Could not locate contrasting class pair.")


@torch.no_grad()
def _generate_samples_batched(gen_fn, total: int, batch_size: int, label: str = "") -> torch.Tensor:
    batches = []
    remaining = total
    done = 0
    while remaining > 0:
        cur = min(batch_size, remaining)
        batches.append(gen_fn(cur).cpu())
        remaining -= cur
        done += cur
        if label:
            print(f"  [{label}] Generated {done}/{total} samples...", flush=True)
    return torch.cat(batches, dim=0)


def _save_json(path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage_fast(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device) -> None:
    """Run MLP Proof, Autoencoders (Vanilla, Sparse, VAE), and GANs (Vanilla, DCGAN)."""
    set_seed(cfg.seed)
    mlp_res = run_mlp_proof(cfg, loaders, device)
    _save_json(
        cfg.results_dir / "stage_mlp.json",
        {
            "test_accuracy": mlp_res["test_accuracy"],
            "val_accuracy": mlp_res["val_accuracy"],
            "orthogonality": mlp_res["orthogonality"],
            "z_mlp_zero_fraction": mlp_res["z_mlp_zero_fraction"],
            "z_mlp_mu_norm": mlp_res["z_mlp_mu_norm"],
            "z_mlp_sigma_mean": mlp_res["z_mlp_sigma_mean"],
            "clf_losses": mlp_res["clf_losses"],
            "frozen_dec_losses": mlp_res["frozen_dec_losses"],
        },
    )

    set_seed(cfg.seed)
    ae_res = train_all_autoencoders(cfg, loaders, device)
    _save_json(
        cfg.results_dir / "stage_ae.json",
        {
            k: {
                "train_losses": v["train_losses"],
                "val_losses": v["val_losses"],
                "epoch_time_s": v["mean_epoch_time"],
            }
            for k, v in ae_res.items()
        },
    )

    set_seed(cfg.seed)
    gan_res = train_all_gans(cfg, loaders, device)
    _save_json(
        cfg.results_dir / "stage_gan.json",
        {
            k: {
                "g_losses": v["g_losses"],
                "d_losses": v["d_losses"],
                "epoch_time_s": v["mean_epoch_time"],
            }
            for k, v in gan_res.items()
        },
    )


def run_stage_sparse_ae(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device) -> None:
    """Retrain only Sparse AE and update results/stage_ae.json."""
    set_seed(cfg.seed)
    res = train_sparse_ae_only(cfg, loaders, device)
    ae_json_path = cfg.results_dir / "stage_ae.json"
    ae_data = _load_json(ae_json_path)
    ae_data["sparse_ae"] = {
        "train_losses": res["train_losses"],
        "val_losses": res["val_losses"],
        "epoch_time_s": res["mean_epoch_time"],
    }
    _save_json(ae_json_path, ae_data)


def run_stage_dcgan(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device) -> None:
    """Retrain only DCGAN and update results/stage_gan.json."""
    set_seed(cfg.seed)
    res = train_dcgan_only(cfg, loaders, device)
    gan_json_path = cfg.results_dir / "stage_gan.json"
    gan_data = _load_json(gan_json_path)
    gan_data["dcgan"] = {
        "g_losses": res["g_losses"],
        "d_losses": res["d_losses"],
        "epoch_time_s": res["mean_epoch_time"],
    }
    _save_json(gan_json_path, gan_data)


def run_stage_ddpm(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device) -> None:
    """Train 28x28 UNet DDPM with EMA and save training history."""
    set_seed(cfg.seed)
    ddpm_res = train_ddpm(cfg, loaders, device)
    _save_json(
        cfg.results_dir / "stage_ddpm.json",
        {
            "train_losses": ddpm_res["train_losses"],
            "epoch_time_s": ddpm_res["mean_epoch_time"],
        },
    )


def run_stage_sample_ddpm(cfg: ExperimentConfig, device: torch.device) -> None:
    """Generate 7,000 DDPM FID samples + latency + denoising trajectory from saved EMA checkpoint."""
    print("\n=== Sampling 7,000 DDPM Images & Measuring DDPM Latency ===", flush=True)
    ema_unet = UNet28(in_channels=cfg.in_channels, base_ch=64, time_dim=128).to(device)
    ema_unet.load_state_dict(torch.load(cfg.checkpoint_dir / "ddpm_ema.pt", map_location=device))
    ema_unet.eval()
    diffusion = GaussianDiffusion(
        model=ema_unet,
        timesteps=cfg.ddpm_timesteps,
        beta_start=cfg.ddpm_beta_start,
        beta_end=cfg.ddpm_beta_end,
    ).to(device)

    lat_ddpm = measure_inference_latency(
        lambda: diffusion.sample(cfg.latency_batch_size, device, denoiser=ema_unet, use_amp=True)[0],
        device,
        warmup=1,
        runs=3,
    )
    print(f"  [DDPM] Latency per batch 64: {lat_ddpm[0]:.2f} +/- {lat_ddpm[1]:.2f} ms", flush=True)

    set_seed(cfg.seed + 7)
    record_steps = [999, 900, 800, 700, 600, 450, 300, 150, 50, 0]
    _, ddpm_traj = diffusion.sample(4, device, denoiser=ema_unet, record_steps=record_steps, use_amp=True)

    set_seed(cfg.seed)
    gen_ddpm_all = _generate_samples_batched(
        lambda n: diffusion.sample(n, device, denoiser=ema_unet, use_amp=True)[0],
        cfg.fid_num_samples,
        batch_size=700,
        label="DDPM",
    )
    torch.save(
        {
            "samples": gen_ddpm_all,
            "trajectory": ddpm_traj,
            "latency_ms_mean": lat_ddpm[0],
            "latency_ms_std": lat_ddpm[1],
        },
        cfg.results_dir / "ddpm_eval_cache.pt",
    )
    print(f"  Saved DDPM cache to {cfg.results_dir / 'ddpm_eval_cache.pt'}", flush=True)


@torch.no_grad()
def run_stage_eval(cfg: ExperimentConfig, loaders: Dict[str, Any], device: torch.device) -> None:
    """Evaluate all checkpoints on Test Set, compute FID, and build the 5 figures."""
    print("\n=== Quantitative Evaluation & 5 Mandatory Figures ===", flush=True)
    stage_mlp = _load_json(cfg.results_dir / "stage_mlp.json")
    stage_ae = _load_json(cfg.results_dir / "stage_ae.json")
    stage_gan = _load_json(cfg.results_dir / "stage_gan.json")
    stage_ddpm = _load_json(cfg.results_dir / "stage_ddpm.json")
    ddpm_cache = torch.load(cfg.results_dir / "ddpm_eval_cache.pt", map_location="cpu")

    # Load all trained models
    classifier = MLPClassifier(latent_dim=cfg.ae_latent_dim, num_classes=cfg.num_classes).to(device)
    classifier.load_state_dict(torch.load(cfg.checkpoint_dir / "mlp_classifier.pt", map_location=device))
    classifier.eval()

    frozen_decoder = FrozenLatentDecoder(latent_dim=cfg.ae_latent_dim).to(device)
    frozen_decoder.load_state_dict(torch.load(cfg.checkpoint_dir / "frozen_latent_decoder.pt", map_location=device))
    frozen_decoder.eval()

    vanilla_ae = VanillaAutoencoder(latent_dim=cfg.ae_latent_dim).to(device)
    vanilla_ae.load_state_dict(torch.load(cfg.checkpoint_dir / "vanilla_ae.pt", map_location=device))
    vanilla_ae.eval()

    sparse_ae = SparseAutoencoder(latent_dim=cfg.sparse_latent_dim).to(device)
    sparse_ae.load_state_dict(torch.load(cfg.checkpoint_dir / "sparse_ae.pt", map_location=device))
    sparse_ae.eval()

    vae_model = VariationalAutoencoder(latent_dim=cfg.vae_latent_dim).to(device)
    vae_model.load_state_dict(torch.load(cfg.checkpoint_dir / "vae.pt", map_location=device))
    vae_model.eval()

    vgan_g = VanillaGenerator(latent_dim=cfg.gan_latent_dim).to(device)
    vgan_g.load_state_dict(torch.load(cfg.checkpoint_dir / "vanilla_gan_G.pt", map_location=device))
    vgan_g.eval()

    dcgan_g = DCGANGenerator(latent_dim=cfg.dcgan_latent_dim).to(device)
    dcgan_g.load_state_dict(torch.load(cfg.checkpoint_dir / "dcgan_G.pt", map_location=device))
    dcgan_g.eval()

    # Recompute MLP proof visual tensors & Frozen Decoder test metrics deterministically
    test_images = torch.cat([x for x, _ in loaders["test"]], dim=0)
    all_z_test, all_frozen_recons = [], []
    for x, _ in loaders["test"]:
        z_mlp = classifier.extract_features(x.to(device))
        all_z_test.append(z_mlp.cpu())
        all_frozen_recons.append(frozen_decoder(z_mlp).cpu())
    z_test_cat = torch.cat(all_z_test, dim=0)
    frozen_recons_cat = torch.cat(all_frozen_recons, dim=0)
    frozen_metrics = compute_reconstruction_metrics(frozen_recons_cat, test_images, device)

    sample_x, sample_y = next(iter(loaders["test"]))
    sample_x_dev, sample_y_dev = sample_x[:8].to(device), sample_y[:8].to(device)
    sample_z_mlp = classifier.extract_features(sample_x_dev)
    sample_onehot_transpose = classifier.project_from_class_onehot(
        torch.nn.functional.one_hot(sample_y_dev, num_classes=cfg.num_classes).float()
    ).cpu()
    sample_latent_transpose = classifier.project_from_latent(sample_z_mlp).cpu()
    sample_frozen_recon = frozen_decoder(sample_z_mlp).cpu()

    set_seed(cfg.seed)
    mu_z = z_test_cat.mean(dim=0, keepdim=True)
    sigma_z = z_test_cat.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_rand = mu_z + sigma_z * torch.randn(16, cfg.ae_latent_dim)
    mlp_prior_samples = frozen_decoder(z_rand.to(device)).cpu()

    transpose_onehot_metrics = compute_reconstruction_metrics(sample_onehot_transpose, sample_x[:8], device)
    transpose_latent_metrics = compute_reconstruction_metrics(sample_latent_transpose, sample_x[:8], device)

    eval_vanilla_ae = evaluate_autoencoder_on_test(vanilla_ae, loaders["test"], device)
    eval_sparse_ae = evaluate_autoencoder_on_test(sparse_ae, loaders["test"], device)
    eval_vae = evaluate_autoencoder_on_test(vae_model, loaders["test"], device, is_vae=True)

    # Latency measurements (ms / batch 64)
    dummy_x64 = torch.rand(cfg.latency_batch_size, 1, 28, 28, device=device)
    dummy_z_vae = torch.randn(cfg.latency_batch_size, cfg.vae_latent_dim, device=device)
    dummy_z_gan = torch.randn(cfg.latency_batch_size, cfg.gan_latent_dim, device=device)
    dummy_z_dcgan = torch.randn(cfg.latency_batch_size, cfg.dcgan_latent_dim, device=device)

    lat_vanilla_ae = measure_inference_latency(
        lambda: vanilla_ae(dummy_x64)[0], device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_sparse_ae = measure_inference_latency(
        lambda: sparse_ae(dummy_x64)[0], device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_vae = measure_inference_latency(
        lambda: vae_model.decode(dummy_z_vae), device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_vgan = measure_inference_latency(
        lambda: (vgan_g(dummy_z_gan) + 1.0) / 2.0, device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_dcgan = measure_inference_latency(
        lambda: (dcgan_g(dummy_z_dcgan) + 1.0) / 2.0, device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )

    print("Computing InceptionV3 FID scores (7,000 test vs 7,000 generated samples)...", flush=True)
    extractor = InceptionFeatureExtractor(device)
    set_seed(cfg.seed)
    gen_vae_all = _generate_samples_batched(
        lambda n: vae_model.decode(torch.randn(n, cfg.vae_latent_dim, device=device)),
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_vgan_all = _generate_samples_batched(
        lambda n: (vgan_g(torch.randn(n, cfg.gan_latent_dim, device=device)) + 1.0) / 2.0,
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_dcgan_all = _generate_samples_batched(
        lambda n: (dcgan_g(torch.randn(n, cfg.dcgan_latent_dim, device=device)) + 1.0) / 2.0,
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_ddpm_all = ddpm_cache["samples"]

    fid_vae = compute_fid(test_images, gen_vae_all, extractor, device)
    fid_vgan = compute_fid(test_images, gen_vgan_all, extractor, device)
    fid_dcgan = compute_fid(test_images, gen_dcgan_all, extractor, device)
    fid_ddpm = compute_fid(test_images, gen_ddpm_all, extractor, device)

    print("Generating 5 mandatory high-resolution figures...", flush=True)
    distinct_x, distinct_y = _select_distinct_test_samples(loaders, 8)
    dx = distinct_x.to(device)
    plot_fig1_reconstructions(
        distinct_x,
        distinct_y,
        vanilla_ae(dx)[0].cpu(),
        sparse_ae(dx)[0].cpu(),
        vae_model.decode(vae_model.encode(dx)).cpu(),
        cfg.figures_dir / "fig1_reconstruction_grid.png",
    )

    plot_fig2_random_sampling(
        {
            "vae": gen_vae_all[:16],
            "vanilla_gan": gen_vgan_all[:16],
            "dcgan": gen_dcgan_all[:16],
            "ddpm": gen_ddpm_all[:16],
        },
        cfg.figures_dir / "fig2_generative_sampling_4x4.png",
    )

    plot_fig3_mlp_proof(
        {
            "sample_x": sample_x[:8],
            "sample_y": sample_y[:8],
            "sample_onehot_transpose": sample_onehot_transpose,
            "sample_latent_transpose": sample_latent_transpose,
            "sample_frozen_recon": sample_frozen_recon,
            "prior_samples": mlp_prior_samples,
        },
        vanilla_ae(sample_x_dev)[0].cpu(),
        cfg.figures_dir / "fig3_mlp_limitation_proof.png",
    )

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    img_a, img_b = _select_contrasting_pair(loaders, class_a=7, class_b=3)
    img_a, img_b = img_a.to(device), img_b.to(device)

    def _interp_encoder_model(m):
        za, zb = m.encode(img_a), m.encode(img_b)
        zs = torch.cat([(1.0 - a) * za + a * zb for a in alphas], dim=0)
        return m.decode(zs).cpu()

    def _interp_generator_model(g, z_dim: int):
        gen = torch.Generator(device=device).manual_seed(cfg.seed + 99)
        za = torch.randn(1, z_dim, device=device, generator=gen)
        zb = torch.randn(1, z_dim, device=device, generator=gen)
        zs = torch.cat([(1.0 - a) * za + a * zb for a in alphas], dim=0)
        return ((g(zs) + 1.0) / 2.0).cpu()

    interp_rows = [
        ("Vanilla AE", _interp_encoder_model(vanilla_ae)),
        ("Sparse AE", _interp_encoder_model(sparse_ae)),
        ("VAE", _interp_encoder_model(vae_model)),
        ("Vanilla GAN", _interp_generator_model(vgan_g, cfg.gan_latent_dim)),
        ("DCGAN", _interp_generator_model(dcgan_g, cfg.dcgan_latent_dim)),
    ]
    plot_fig4_latent_interpolation(interp_rows, alphas, cfg.figures_dir / "fig4_latent_interpolation.png")

    histories = {
        "vanilla_ae": stage_ae["vanilla_ae"],
        "sparse_ae": stage_ae["sparse_ae"],
        "vae": stage_ae["vae"],
        "vanilla_gan": stage_gan["vanilla_gan"],
        "dcgan": stage_gan["dcgan"],
        "ddpm": stage_ddpm,
    }
    plot_fig5_losses_and_ddpm_trajectory(
        histories, ddpm_cache["trajectory"], cfg.figures_dir / "fig5_loss_curves_and_ddpm_trajectory.png"
    )

    summary = {
        "dataset": {
            "name": cfg.dataset_name,
            "train_size": len(loaders["train"].dataset),
            "val_size": len(loaders["val"].dataset),
            "test_size": len(loaders["test"].dataset),
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "mlp_proof": {
            "test_accuracy": stage_mlp["test_accuracy"],
            "val_accuracy": stage_mlp["val_accuracy"],
            "orthogonality": stage_mlp["orthogonality"],
            "z_mlp_zero_fraction": stage_mlp["z_mlp_zero_fraction"],
            "z_mlp_mu_norm": stage_mlp["z_mlp_mu_norm"],
            "z_mlp_sigma_mean": stage_mlp["z_mlp_sigma_mean"],
            "transpose_onehot_metrics": transpose_onehot_metrics,
            "transpose_latent_metrics": transpose_latent_metrics,
            "frozen_decoder_metrics": frozen_metrics,
        },
        "models": {
            "vanilla_ae": {
                "latent_dim": cfg.ae_latent_dim,
                "epoch_time_s": stage_ae["vanilla_ae"]["epoch_time_s"],
                "latency_ms_mean": lat_vanilla_ae[0],
                "latency_ms_std": lat_vanilla_ae[1],
                **eval_vanilla_ae["metrics"],
                "fid": None,
            },
            "sparse_ae": {
                "latent_dim": cfg.sparse_latent_dim,
                "l1_lambda": cfg.sparse_l1_lambda,
                "epoch_time_s": stage_ae["sparse_ae"]["epoch_time_s"],
                "latency_ms_mean": lat_sparse_ae[0],
                "latency_ms_std": lat_sparse_ae[1],
                **eval_sparse_ae["metrics"],
                "fid": None,
            },
            "vae": {
                "latent_dim": cfg.vae_latent_dim,
                "epoch_time_s": stage_ae["vae"]["epoch_time_s"],
                "latency_ms_mean": lat_vae[0],
                "latency_ms_std": lat_vae[1],
                **eval_vae["metrics"],
                "fid": fid_vae,
            },
            "vanilla_gan": {
                "latent_dim": cfg.gan_latent_dim,
                "epoch_time_s": stage_gan["vanilla_gan"]["epoch_time_s"],
                "latency_ms_mean": lat_vgan[0],
                "latency_ms_std": lat_vgan[1],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_vgan,
            },
            "dcgan": {
                "latent_dim": cfg.dcgan_latent_dim,
                "epoch_time_s": stage_gan["dcgan"]["epoch_time_s"],
                "latency_ms_mean": lat_dcgan[0],
                "latency_ms_std": lat_dcgan[1],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_dcgan,
            },
            "ddpm": {
                "latent_dim": "28x28",
                "timesteps": cfg.ddpm_timesteps,
                "epoch_time_s": stage_ddpm["epoch_time_s"],
                "latency_ms_mean": ddpm_cache["latency_ms_mean"],
                "latency_ms_std": ddpm_cache["latency_ms_std"],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_ddpm,
            },
        },
        "histories": histories,
    }
    metrics_path = cfg.results_dir / "metrics.json"
    _save_json(metrics_path, summary)
    print(f"\nSaved complete quantitative summary to {metrics_path}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "histories"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative Models Comparative Study Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "fast", "sparse_ae", "dcgan", "ddpm", "sample_ddpm", "eval"],
        default="all",
        help="Run entire pipeline ('all') or a specific resumable stage.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    device = get_device()
    for d in [cfg.data_dir, cfg.checkpoint_dir, cfg.results_dir, cfg.figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Running Comparative Study (stage={args.stage}) on device: {device}", flush=True)
    loaders = create_dataloaders(cfg)

    if args.stage in ("all", "fast"):
        run_stage_fast(cfg, loaders, device)
    if args.stage == "sparse_ae":
        run_stage_sparse_ae(cfg, loaders, device)
    if args.stage == "dcgan":
        run_stage_dcgan(cfg, loaders, device)
    if args.stage in ("all", "ddpm"):
        run_stage_ddpm(cfg, loaders, device)
    if args.stage in ("all", "sample_ddpm"):
        run_stage_sample_ddpm(cfg, device)
    if args.stage in ("all", "eval"):
        run_stage_eval(cfg, loaders, device)


if __name__ == "__main__":
    main()
