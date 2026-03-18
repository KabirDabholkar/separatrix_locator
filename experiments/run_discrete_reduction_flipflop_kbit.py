from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from separatrix_locator.dynamics.flipflop_task import FlipFlopDataset
from separatrix_locator.dynamics.rnn_training import train_flipflop_rnn, build_rnn_model
from separatrix_locator.reduction import EncoderNN, LatentDynamicsNN, LatentEigenfunctionNN
from separatrix_locator.reduction.training_discrete import DiscreteReductionTrainConfig, train_discrete_reduction
from separatrix_locator.reduction.trajectory_utils import rollout_rnn_hidden_states, extract_one_step_pairs

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def default_rhs_function(psi: torch.Tensor, u_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Same squashed RHS used elsewhere in the repo: psi -> psi - psi^3.
    return psi - psi ** 3


@torch.no_grad()
def compute_latent_bounds(z: torch.Tensor, quantile_low: float = 0.05, quantile_high: float = 0.95):
    lo = torch.quantile(z, quantile_low, dim=0)
    hi = torch.quantile(z, quantile_high, dim=0)
    # Pad bounds to make plots nicer.
    pad = (hi - lo).clamp_min(1e-6) * 0.25
    return lo - pad, hi + pad


@torch.no_grad()
def visualize_latent_dynamics_2d(
    encoder: nn.Module,
    latent_dynamics: nn.Module,
    x_t_sample: torch.Tensor,
    u_vis: torch.Tensor,
    save_path: Path,
    device: torch.device,
    grid_res: int = 25,
):
    encoder.eval()
    latent_dynamics.eval()

    z_sample = encoder(x_t_sample.to(device))
    z_lo, z_hi = compute_latent_bounds(z_sample)
    z_lo = z_lo.cpu()
    z_hi = z_hi.cpu()

    z1 = torch.linspace(float(z_lo[0]), float(z_hi[0]), grid_res)
    z2 = torch.linspace(float(z_lo[1]), float(z_hi[1]), grid_res)
    Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")

    Z = torch.stack([Z1, Z2], dim=-1).reshape(-1, 2).to(device)
    u_batch = u_vis.to(device).view(1, -1).repeat(Z.shape[0], 1)

    z_next = latent_dynamics(Z, u_batch)
    v = (z_next - Z).reshape(grid_res, grid_res, 2).cpu().numpy()

    U = v[..., 0]
    V = v[..., 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.streamplot(z1.numpy(), z2.numpy(), U, V, density=1.2, linewidth=1.0, color="red", arrowsize=1.2)

    # Overlay projected training points (subsample).
    z_scatter = z_sample[: min(2000, z_sample.shape[0])].cpu().numpy()
    ax.scatter(z_scatter[:, 0], z_scatter[:, 1], s=6, alpha=0.25, c="black")

    ax.set_title(f"Latent dynamics (2D streamplot)")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def visualize_latent_dynamics_3d_quiver(
    encoder: nn.Module,
    latent_dynamics: nn.Module,
    x_t_sample: torch.Tensor,
    u_vis: torch.Tensor,
    save_path: Path,
    device: torch.device,
    grid_res: int = 4,
):
    encoder.eval()
    latent_dynamics.eval()

    z_sample = encoder(x_t_sample.to(device))
    z_lo, z_hi = compute_latent_bounds(z_sample)
    z_lo = z_lo.cpu()
    z_hi = z_hi.cpu()

    z1 = torch.linspace(float(z_lo[0]), float(z_hi[0]), grid_res)
    z2 = torch.linspace(float(z_lo[1]), float(z_hi[1]), grid_res)
    z3 = torch.linspace(float(z_lo[2]), float(z_hi[2]), grid_res)

    Z1, Z2, Z3 = torch.meshgrid(z1, z2, z3, indexing="ij")
    Z = torch.stack([Z1, Z2, Z3], dim=-1).reshape(-1, 3).to(device)
    u_batch = u_vis.to(device).view(1, -1).repeat(Z.shape[0], 1)

    z_next = latent_dynamics(Z, u_batch)
    v = (z_next - Z).cpu().numpy()
    Z_np = Z.cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(
        Z_np[:, 0],
        Z_np[:, 1],
        Z_np[:, 2],
        v[:, 0],
        v[:, 1],
        v[:, 2],
        length=0.2,
        normalize=False,
        linewidth=0.8,
        color="red",
        alpha=0.8,
    )

    z_scatter = z_sample[: min(5000, z_sample.shape[0])].cpu().numpy()
    ax.scatter(z_scatter[:, 0], z_scatter[:, 1], z_scatter[:, 2], s=6, alpha=0.25, c="black")

    ax.set_title("Latent dynamics (3D quiver; v = g(z,u) - z)")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def train_or_load_rnn(
    *,
    k: int,
    hidden_size: int,
    n_trials: int,
    n_time: int,
    p: float,
    random_seed: int,
    epochs: int,
    lr: float,
    mse_threshold: float,
    max_epochs: int,
    cell_type: str,
    device: torch.device,
    checkpoint_path: Path,
):
    def eval_mse(model_: nn.Module, inputs_np, targets_np) -> float:
        model_.eval()
        inputs = torch.from_numpy(inputs_np).float().to(device)
        targets = torch.from_numpy(targets_np).float().to(device)
        with torch.no_grad():
            outputs = model_(inputs)
            mse = torch.nn.functional.mse_loss(outputs, targets).item()
        return float(mse)

    dataset = FlipFlopDataset(
        n_trials=n_trials,
        repeats=1,
        n_bits=k,
        n_time=n_time,
        p=p,
        random_seed=random_seed,
    )

    # Important: train/evaluate on a FIXED dataset batch to make learning stable
    # and to allow checkpoint validation via an MSE threshold.
    inputs_np, targets_np = dataset()

    def fixed_dataset():
        return inputs_np, targets_np

    # 1) Try loading checkpoint and validate MSE.
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model = build_rnn_model(
            input_size=k,
            output_size=k,
            hidden_size=hidden_size,
            cell_type=cell_type,
        ).to(device)
        model.load_state_dict(state)

        mse = eval_mse(model, inputs_np, targets_np)
        print(f"[RNN checkpoint] k={k} loaded '{checkpoint_path.name}' mse={mse:.6f} (threshold={mse_threshold})")
        if mse <= mse_threshold:
            model.eval()
            return model

    # 2) Train/retrain until MSE < threshold (or max_epochs reached).
    from separatrix_locator.dynamics.rnn_training import OptimizerConfig, TrainingConfig

    epochs_to_try = epochs
    while True:
        model = build_rnn_model(
            input_size=k,
            output_size=k,
            hidden_size=hidden_size,
            cell_type=cell_type,
        ).to(device)

        optimizer_config = OptimizerConfig(lr=lr, weight_decay=0.0)
        training_config = TrainingConfig(
            epochs=epochs_to_try,
            log_interval=200,
            device=str(device),
            save_dir=None,
            save_checkpoint=False,
            save_loss_plot=False,
        )

        result = train_flipflop_rnn(
            dataset=fixed_dataset,
            input_size=k,
            output_size=k,
            hidden_size=hidden_size,
            criterion=None,
            optimizer_config=optimizer_config,
            training_config=training_config,
            cell_type=cell_type,
            progress_callback=None,
        )

        model = result.model.to(device)
        mse = eval_mse(model, inputs_np, targets_np)
        print(
            f"[RNN train] k={k} epochs={epochs_to_try} mse={mse:.6f} (threshold={mse_threshold})"
        )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        if mse <= mse_threshold or epochs_to_try >= max_epochs:
            if mse > mse_threshold:
                raise RuntimeError(
                    f"RNN training failed to reach mse<{mse_threshold} for k={k}. "
                    f"Got mse={mse} at epochs={epochs_to_try} (max_epochs={max_epochs})."
                )
            return model

        epochs_to_try = min(int(epochs_to_try * 2), max_epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k-list", type=str, default="2,3")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--rnn-epochs", type=int, default=2000)
    parser.add_argument("--rnn-lr", type=float, default=5e-4)
    parser.add_argument("--rnn-cell", type=str, default="GRU")
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--n-time", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rnn-mse-threshold", type=float, default=1e-2)
    parser.add_argument("--rnn-max-epochs", type=int, default=8000)
    parser.add_argument("--train-rnn-only", action="store_true")
    parser.add_argument("--reduction-epochs", type=int, default=2000)
    parser.add_argument("--reduction-lr", type=float, default=1e-3)
    parser.add_argument("--latent-dynamics-weight", type=float, default=1.0)
    parser.add_argument("--koopman-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float, default=1.0)
    parser.add_argument(
        "--geometry-loss",
        type=str,
        default="stress",
        choices=["stress", "pca"],
        help="Geometry loss in the latent space: distance stress or PCA-supervised regression.",
    )
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--u-vis", type=str, default="zeros", help="zeros or mean or +/-1 pattern like 1,-1,1")
    parser.add_argument("--save-dir", type=str, default="results/discrete_reduction_flipflop")
    args = parser.parse_args()

    device = torch.device(args.device)
    k_list = [int(s) for s in args.k_list.split(",") if s.strip()]

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for k in k_list:
        out_dir = save_root / f"{k}bit_latent{k}_hidden{args.hidden_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rnn_ckpt = out_dir / f"rnn_{k}bit_{args.rnn_cell}_hidden{args.hidden_size}.pt"

        # 1) Train/load discrete-time RNN.
        rnn_model = train_or_load_rnn(
            k=k,
            hidden_size=args.hidden_size,
            n_trials=args.n_trials,
            n_time=args.n_time,
            p=args.p,
            random_seed=args.seed,
            epochs=args.rnn_epochs,
            lr=args.rnn_lr,
            mse_threshold=args.rnn_mse_threshold,
            max_epochs=args.rnn_max_epochs,
            cell_type=args.rnn_cell,
            device=device,
            checkpoint_path=rnn_ckpt,
        )

        if args.train_rnn_only:
            continue

        # 2) Generate one-step dataset (x_t, x_next, u_t).
        dataset = FlipFlopDataset(
            n_trials=args.n_trials,
            repeats=1,
            n_bits=k,
            n_time=args.n_time,
            p=args.p,
            random_seed=args.seed + 1,
        )
        u_seq_np, _ = dataset()
        u_seq = torch.from_numpy(u_seq_np).to(device=device, dtype=torch.float32)  # (T, B, k)

        x_seq = rollout_rnn_hidden_states(rnn_model, u_seq, device=device)  # (T, B, D_hidden)
        x_t, x_next, u_t = extract_one_step_pairs(x_seq, u_seq)  # (N, D), (N, D), (N, k)

        # Subsample reduction training data for speed (configurable later).
        max_pairs = min(20000, x_t.shape[0])
        if max_pairs < x_t.shape[0]:
            sel = torch.randperm(x_t.shape[0], device=device)[:max_pairs]
            x_t_train = x_t[sel]
            x_next_train = x_next[sel]
            u_t_train = u_t[sel]
        else:
            x_t_train, x_next_train, u_t_train = x_t, x_next, u_t

        # 3) Build reduction networks.
        encoder = EncoderNN(input_dim=args.hidden_size, latent_dim=k, hidden_dim=256, num_layers=3).to(device)
        latent_dynamics = LatentDynamicsNN(latent_dim=k, u_dim=k, hidden_dim=256, num_layers=3).to(device)
        eigenfunction = LatentEigenfunctionNN(latent_dim=k, hidden_dim=256, num_layers=3).to(device)

        # Geometry-loss setup (either stress or PCA-supervision).
        pca_components = None
        pca_mean = None
        geometry_loss_type = args.geometry_loss
        if geometry_loss_type == "pca":
            x_all = torch.cat([x_t_train, x_next_train], dim=0).detach().cpu().numpy()
            pca = PCA(n_components=k)
            pca.fit(x_all)
            pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
            pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
            print(f"[PCA] fit for k={k}: components={pca.components_.shape} mean={pca.mean_.shape}")

        # Choose u_vis for visualization.
        if args.u_vis.lower() == "zeros":
            u_vis = torch.zeros(k, dtype=torch.float32)
        elif args.u_vis.lower() == "mean":
            u_vis = u_t_train.mean(dim=0).detach().cpu().to(torch.float32)
        else:
            # Parse pattern like "1,-1,1"
            parts = [float(x) for x in args.u_vis.split(",")]
            if len(parts) != k:
                raise ValueError(f"Expected u-vis with {k} entries, got {len(parts)}")
            u_vis = torch.tensor(parts, dtype=torch.float32)

        # 4) Train discrete-time reduction.
        train_cfg = DiscreteReductionTrainConfig(
            n_epochs=args.reduction_epochs,
            batch_size=512,
            lr=args.reduction_lr,
            weight_state=args.latent_dynamics_weight,
            weight_koopman=args.koopman_weight,
            weight_stress=args.stress_weight,
            kappa=args.kappa,
            geometry_loss_type=geometry_loss_type,
        )

        history = train_discrete_reduction(
            x_t=x_t_train,
            x_next=x_next_train,
            u_t=u_t_train,
            encoder=encoder,
            latent_dynamics=latent_dynamics,
            eigenfunction=eigenfunction,
            rhs_function=default_rhs_function,
            config=train_cfg,
            device=device,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"weight_decay": 1e-4},
            pca_components=pca_components,
            pca_mean=pca_mean,
        )

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "latent_dynamics": latent_dynamics.state_dict(),
                "eigenfunction": eigenfunction.state_dict(),
                "history": history,
                "train_cfg": asdict(train_cfg),
            },
            out_dir / "reduction_model.pt",
        )

        # 5) Visualize latent dynamics.
        x_sample = x_t_train[: min(5000, x_t_train.shape[0])].detach().cpu()
        if k == 2:
            visualize_latent_dynamics_2d(
                encoder=encoder,
                latent_dynamics=latent_dynamics,
                x_t_sample=x_sample,
                u_vis=u_vis,
                save_path=out_dir / "latent_dynamics_2d_streamplot.png",
                device=device,
                grid_res=25,
            )
        elif k == 3:
            visualize_latent_dynamics_3d_quiver(
                encoder=encoder,
                latent_dynamics=latent_dynamics,
                x_t_sample=x_sample,
                u_vis=u_vis,
                save_path=out_dir / "latent_dynamics_3d_quiver.png",
                device=device,
                grid_res=4,
            )

        # Plot training loss curves.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["loss_total"], label="total")
        ax.plot(history["loss_state"], label="state")
        ax.plot(history["loss_koopman"], label="koopman")
        if geometry_loss_type == "pca":
            ax.plot(history["loss_stress"], label="pca-supervision")
        else:
            ax.plot(history["loss_stress"], label="stress")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log)")
        ax.set_title(f"Discrete reduction losses ({k}bit, geometry={geometry_loss_type})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "reduction_losses.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()

