import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


NIBBLE_POPCOUNT = np.array(
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4], dtype=np.uint8
)


def _packed_area_counts(mask):

    counts = np.empty((2048, 512), dtype=np.uint8)
    counts[:, 0::2] = NIBBLE_POPCOUNT[mask >> 4]
    counts[:, 1::2] = NIBBLE_POPCOUNT[mask & 15]

    return counts.reshape(512, 4, 512).sum(1, dtype=np.uint8)


def build_classifier_input(masks: torch.Tensor, specs: list[dict]) -> torch.Tensor:

    if masks.ndim == 3:
        masks = masks[:, None]
    batch, _, height, width = masks.shape
    if len(specs) != batch:
        raise ValueError("Exactly one terminal specification is required per mask")
    result = torch.zeros((batch, 4, height, width), device=masks.device)
    result[:, :1] = masks
    x = torch.arange(width, device=masks.device, dtype=torch.float32)
    y = torch.arange(height, device=masks.device, dtype=torch.float32)
    for index, spec in enumerate(specs):
        if spec["kind"] not in {"bridge", "break"} or len(spec["terminals"]) != 2:
            raise ValueError("Expected bridge/break with two fixed terminal ROIs")
        source_h, source_w = spec.get("source_shape", (height, width))
        for channel, (y0, x0, y1, x1) in enumerate(spec["terminals"], start=1):
            if not (0 <= y0 < y1 <= source_h and 0 <= x0 < x1 <= source_w):
                raise ValueError("Terminal lies outside its original image")
            y0, y1 = y0 * height / source_h, y1 * height / source_h
            x0, x1 = x0 * width / source_w, x1 * width / source_w
            coverage_x = (torch.clamp(x + 1, max=x1) - torch.clamp(x, min=x0)).clamp(
                0, 1
            )
            coverage_y = (torch.clamp(y + 1, max=y1) - torch.clamp(y, min=y0)).clamp(
                0, 1
            )
            result[index, channel] = coverage_y[:, None] * coverage_x[None, :]
        if spec["kind"] == "break":
            result[index, 3] = 1
    return result


class MaskDataset(Dataset):
    def __init__(
        self, masks, labels, packed=False, input_size=512, cached_area_counts=False
    ):
        self.masks = masks
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.packed = packed
        self.input_size = input_size
        self.cached_area_counts = cached_area_counts
        self.source_shape = (
            (2048, 2048)
            if cached_area_counts
            else (masks.shape[1], masks.shape[2] * (8 if packed else 1))
        )

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask = self.masks[index]
        if self.cached_area_counts:
            area = mask.astype(np.float32) / 16.0
            return torch.from_numpy(area[None]), self.labels[index], index
        if self.packed and mask.shape == (2048, 256) and self.input_size == 512:
            area = _packed_area_counts(mask).astype(np.float32) / 16.0
            return torch.from_numpy(area[None]), self.labels[index], index
        if self.packed:
            mask = np.unpackbits(mask, axis=-1, bitorder="big")
        height, width = mask.shape
        target_h, target_w = min(height, self.input_size), min(width, self.input_size)
        if height % target_h == 0 and width % target_w == 0:
            mask = mask.reshape(
                target_h, height // target_h, target_w, width // target_w
            )
            mask = mask.mean((1, 3), dtype=np.float32)
        else:
            mask = F.adaptive_avg_pool2d(
                torch.from_numpy(mask.astype(np.float32))[None, None],
                (target_h, target_w),
            )[0, 0].numpy()
        if not self.packed and mask.max() > 1:
            mask = mask / 255.0
        return torch.from_numpy(mask[None]), self.labels[index], index


def load_split(path: Path, condition="focus", input_size=512):

    records = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        packed = "masks_packed" in archive.files
        masks = archive["masks_packed" if packed else "masks"]
    if masks.ndim != 3 or not len(masks) or len(masks) != len(records):
        raise ValueError(f"Input/metadata shape mismatch: {path}")
    if packed and masks.dtype != np.uint8:
        raise ValueError("Packed masks must use uint8, NumPy's default big bit order")
    labels = []
    for record in records:
        key = f"truth_{condition}"
        if key in record:
            truth = record[key]
        elif condition == "focus" and "truth" in record:
            truth = record["truth"]
        else:
            raise ValueError(f"Record {record.get('id')} has no {key} label")
        labels.append(bool(truth["defect"]))
    cached = packed and masks.shape[1:] == (2048, 256) and input_size == 512
    if cached:
        started = time.perf_counter()
        counts = np.empty((len(masks), 512, 512), dtype=np.uint8)
        for index in range(len(masks)):
            counts[index] = _packed_area_counts(masks[index])
        masks = counts
        print(
            json.dumps(
                {
                    "event": "uint8_area_cache",
                    "split": str(path),
                    "samples": len(masks),
                    "shape": list(masks.shape),
                    "mib": masks.nbytes / 1024**2,
                    "seconds": time.perf_counter() - started,
                    "representation": "exact 4x4 set-bit counts 0..16, float32/16 at access",
                }
            ),
            flush=True,
        )
    dataset = MaskDataset(
        masks, labels, packed=packed, input_size=input_size, cached_area_counts=cached
    )
    specs = [
        {**record["spec"], "source_shape": list(dataset.source_shape)}
        for record in records
    ]
    return dataset, specs


class DefectClassifier(nn.Module):
    def __init__(
        self, width: int = 16, pool_size: int = 4, input_size: int = 512
    ) -> None:
        super().__init__()
        if width < 4 or width % 4 or pool_size < 1:
            raise ValueError("width must be a positive multiple of 4; pool_size >= 1")
        self.width = width
        self.pool_size = pool_size
        self.input_size = input_size
        stages = []
        incoming = 5
        for outgoing in (width, 2 * width, 4 * width, 8 * width):
            stages.extend(
                [
                    nn.Conv2d(incoming, outgoing, 5, stride=2, padding=2, bias=False),
                    nn.GroupNorm(4, outgoing),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                    nn.GroupNorm(4, outgoing),
                    nn.SiLU(inplace=True),
                ]
            )
            incoming = outgoing
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flattened = incoming * pool_size**2 + 2
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(flattened, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )
                for _ in range(2)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("Expected B x 4 x H x W classifier inputs")
        inputs = inputs.float()
        kinds = inputs[:, 3, 0, 0].round().long()
        if max(inputs.shape[-2:]) > self.input_size:
            inputs = F.adaptive_avg_pool2d(
                inputs, tuple(min(side, self.input_size) for side in inputs.shape[-2:])
            )
        batch, _, height, width = inputs.shape
        x = (
            torch.arange(width, device=inputs.device, dtype=inputs.dtype) + 0.5
        ) / width
        y = (
            torch.arange(height, device=inputs.device, dtype=inputs.dtype) + 0.5
        ) / height
        terminals = inputs[:, 1:3]
        mass = terminals.sum((2, 3)).clamp_min(1.0)
        center_x = (terminals * x[None, None, None, :]).sum((2, 3)) / mass
        center_y = (terminals * y[None, None, :, None]).sum((2, 3)) / mass
        midpoint_x = center_x.mean(1)
        midpoint_y = center_y.mean(1)
        relative_x = 2.0 * (x[None, None, None, :] - midpoint_x[:, None, None, None])
        relative_y = 2.0 * (y[None, None, :, None] - midpoint_y[:, None, None, None])
        visual = torch.cat(
            (
                2.0 * inputs[:, :1] - 1.0,
                terminals,
                relative_x.expand(batch, 1, height, width),
                relative_y.expand(batch, 1, height, width),
            ),
            dim=1,
        )
        offset = torch.stack(
            (center_x[:, 1] - center_x[:, 0], center_y[:, 1] - center_y[:, 0]), dim=1
        )
        feature = self.pool(self.features(visual)).flatten(1)
        feature = torch.cat((feature, offset.to(feature.dtype)), dim=1)
        logits = torch.cat([head(feature) for head in self.heads], dim=1)
        return logits.gather(1, kinds[:, None]).squeeze(1)


def load_classifier(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DefectClassifier, dict]:

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if config.get("model_type") != "terminal_conditioned_v2":
        raise ValueError("This loader requires a classifier checkpoint")
    model = DefectClassifier(config["width"], config["pool_size"], config["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def summarize_logits(logits, labels, kinds):

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    kinds = np.asarray(kinds)

    def one_subset(subset):
        score, truth = logits[subset], labels[subset]
        n_positive, n_negative = int(truth.sum()), int((~truth).sum())
        if not len(score):
            return {"n": 0, "bce": None, "accuracy": None, "auroc": None}
        auc = None
        if n_positive and n_negative:
            ranks = rankdata(score)
            auc = float(
                (ranks[truth].sum() - n_positive * (n_positive + 1) / 2)
                / (n_positive * n_negative)
            )
        return {
            "n": len(score),
            "n_defects": n_positive,
            "bce": float(np.mean(np.logaddexp(0.0, score) - truth * score)),
            "accuracy": float(np.mean((score > 0) == truth)),
            "auroc": auc,
            "logit_std": float(score.std()),
        }

    result = one_subset(np.ones(len(logits), dtype=bool))
    result["by_kind"] = {
        name: one_subset(kinds == index)
        for index, name in enumerate(("bridge", "break"))
    }
    kind_bces = [
        value["bce"] for value in result["by_kind"].values() if value["bce"] is not None
    ]
    result["mean_kind_bce"] = float(np.mean(kind_bces))
    return result


def run_epoch(model, loader, specs, device, optimizer=None, scaler=None):

    training = optimizer is not None
    model.train(training)
    all_logits, all_labels, all_kinds = [], [], []
    for masks, labels, indices in loader:
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_specs = [specs[index] for index in indices.tolist()]
        inputs = build_classifier_input(masks, batch_specs)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classifier loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        all_logits.extend(logits.detach().float().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_kinds.extend(0 if spec["kind"] == "bridge" else 1 for spec in batch_specs)
    return summarize_logits(all_logits, all_labels, all_kinds)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Terminal-conditioned direct classifier with spatial pooling and kind heads."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--condition", choices=("focus", "defocus"), default="focus")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.threads, args.pool_size) < 1
        or args.lr <= 0
    ):
        parser.error("Counts and learning rate must be positive")
    if args.input_size < 32:
        parser.error("input-size must be at least 32")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pt").exists() or (args.output / "epochs.jsonl").exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_data, train_specs = load_split(
        args.data / "train", args.condition, args.input_size
    )
    val_data, val_specs = load_split(args.data / "val", args.condition, args.input_size)
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
    model = DefectClassifier(args.width, args.pool_size, args.input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_type": "terminal_conditioned_v2",
        "torch_version": str(torch.__version__),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input": "full mask, terminal maps, relative midpoint coordinates; kind selects head only",
        "resize": "full-image area occupancy; packed2048->512 cached as uint8 0..16 counts, float32/16 at access",
        "mask_normalization": "2 * binary_mask - 1",
        "normalization": "GroupNorm(4)",
        "activation": "SiLU",
        "selection": "lowest_validation_mean_kind_BCE",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "uses_calibration_or_test": False,
        "prediction": "defect_logit",
    }
    print(json.dumps({"event": "start", "config": config}), flush=True)
    best_loss, best_epoch, best_metrics = float("inf"), 0, {}
    with (args.output / "epochs.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = run_epoch(
                model, train_loader, train_specs, device, optimizer, scaler
            )
            val_metrics = run_epoch(model, val_loader, val_specs, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if val_metrics["mean_kind_bce"] < best_loss:
                best_loss = val_metrics["mean_kind_bce"]
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
        "peak_gpu_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        ),
        "checkpoint": str((args.output / "best.pt").resolve()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
