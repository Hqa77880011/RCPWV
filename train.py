import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import LowRankResidualSurrogate


VALIDATION_BOUNDARY_WEIGHT = 9.0


def load_split(path: Path) -> TensorDataset:

    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        aerial = archive["aerial_grid"]
    if (
        features.ndim != 5
        or features.shape[1:] != (2, 5, 128, 128)
        or aerial.shape != (len(features), 2, 128, 128)
        or len(features) == 0
    ):
        raise ValueError(f"Invalid optical feature/target shapes: {path}")
    if not np.isfinite(features).all() or not np.isfinite(aerial).all():
        raise ValueError(f"Non-finite feature or target values: {path}")
    if np.min(features[:, :, :4]) < -1e-7 or aerial.min() < -1e-7:
        raise ValueError(f"Weighted intensities must be nonnegative: {path}")
    source_count = len(features)
    return TensorDataset(
        torch.from_numpy(
            features.reshape(-1, 5, 128, 128).astype(np.float32, copy=False)
        ),
        torch.from_numpy(aerial.reshape(-1, 128, 128).astype(np.float32, copy=False)),
        torch.tensor([0, 1], dtype=torch.long).repeat(source_count),
    )


def aerial_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.225,
    boundary_weight: float = 9.0,
    boundary_width: float = 0.04,
) -> tuple[torch.Tensor, dict]:

    prediction, target = prediction.float(), target.float()
    error = prediction - target
    weights = 1 + boundary_weight * torch.exp(
        -(target - threshold).abs() / boundary_width
    )
    per_image = (weights * error.square()).mean(dim=(1, 2)) / weights.mean(dim=(1, 2))
    return per_image.mean(), {
        "mse": error.square().mean(),
        "mae": error.abs().mean(),
    }


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    boundary_weight: float,
    boundary_width: float,
) -> dict:

    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for features, target, conditions in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model.forward_features(features, conditions)
                loss, components = aerial_loss(
                    prediction,
                    target,
                    boundary_weight=boundary_weight,
                    boundary_width=boundary_width,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite aerial loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            foreground = prediction > model.physics.threshold
            actual = target > model.physics.threshold
            intersection = (foreground & actual).sum(dim=(1, 2))
            union = (foreground | actual).sum(dim=(1, 2))
            iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
            count = len(features)
            totals += torch.stack(
                [
                    loss.detach() * count,
                    components["mse"].detach() * count,
                    components["mae"].detach() * count,
                    iou.sum(),
                    prediction.new_tensor(count),
                ]
            )
    loss, mse, mae, iou, count = totals.cpu().tolist()
    return {
        "weighted_aerial_mse": loss / count,
        "aerial_mse": mse / count,
        "aerial_mae": mae / count,
        "grid_128_iou_descriptive_only": iou / count,
        "condition_samples": int(count),
        "source_layouts": int(count // 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the low-rank optical residual on cached train/validation features."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--kernels", type=int, choices=(0, 1, 4), default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--boundary-weight", type=float, default=9.0)
    parser.add_argument("--boundary-width", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.threads) < 1:
        parser.error("Training counts must be positive")
    if args.lr <= 0 or args.boundary_weight < 0 or args.boundary_width <= 0:
        parser.error("Invalid learning rate or boundary weighting")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pt").exists() or (args.output / "epochs.jsonl").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_data = load_split(args.data / "train.npz")
    val_data = load_split(args.data / "val.npz")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = LowRankResidualSurrogate(args.kernels, args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_type": (
            "bandlimited_mask_CNN_only"
            if args.kernels == 0
            else "fixed_lowrank_optics_plus_learned_aerial_residual"
        ),
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_source_layouts": len(train_data) // 2,
        "validation_source_layouts": len(val_data) // 2,
        "condition_axis": ["focus", "defocus"],
        "feature_channels": "first K weighted mode intensities, bandlimited mask, condition flag",
        "supervision": "24-mode aerial_grid at nominal dose; never used as model input",
        "grid_size": 128,
        "canvas_size": 2048,
        "pixel_size_nm": 1.0,
        "input_frequency_radius": 17,
        "output_frequency_radius": 34,
        "dose_law": "output intensity multiplied by dose squared",
        "residual_constraint": "softplus positive on native grid before fixed bandwidth projection",
        "loss": "per-image normalized threshold-weighted aerial MSE",
        "validation_boundary_weight": VALIDATION_BOUNDARY_WEIGHT,
        "selection": "lowest validation weighted aerial MSE",
        "uses_calibration_or_test": False,
        "training_feature_cache": "used for training only; runtime must start from original mask",
        "comparison_requirement": "full 24-mode band-limited exact simulator is the timing baseline",
    }
    print(json.dumps({"event": "start", "config": config}), flush=True)
    best_loss, best_epoch, best_metrics = float("inf"), 0, {}
    with (args.output / "epochs.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            begin = time.perf_counter()
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                scaler,
                args.boundary_weight,
                args.boundary_width,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                scaler,
                VALIDATION_BOUNDARY_WEIGHT,
                args.boundary_width,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "epoch_seconds": time.perf_counter() - begin,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else 0,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if val_metrics["weighted_aerial_mse"] < best_loss:
                best_loss = val_metrics["weighted_aerial_mse"]
                best_epoch, best_metrics = epoch, val_metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_loss,
                        "best_val_metrics": best_metrics,
                    },
                    args.output / "best.pt",
                )
            scheduler.step()
    summary = {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
        if device.type == "cuda"
        else 0,
        "checkpoint": str((args.output / "best.pt").resolve()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
