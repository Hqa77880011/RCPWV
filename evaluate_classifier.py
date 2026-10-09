import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from classifier import (
    build_classifier_input,
    load_classifier,
    load_split,
    summarize_logits,
)
from metrics import json_finite, summarize_policies


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_geometry(path, profile):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"split", "id", "kind"}.issubset(reader.fieldnames or []):
            raise ValueError("Geometry CSV requires split, id and kind columns")
        indexed = {}
        for row in reader:
            if row["split"] not in {"cal", "test"}:
                continue
            row_profile = row.get("profile") or row.get("condition")
            if row_profile and row_profile != profile:
                continue
            key = (row["split"], row["id"])
            if key in indexed:
                raise ValueError(
                    f"Duplicate geometry row after condition filter: {key}"
                )
            indexed[key] = row
    return indexed


def _csv_boolean(value):
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false", "0", "1", "0.0", "1.0"}:
        raise ValueError(f"Not a boolean defect value: {value}")
    return normalized in {"true", "1", "1.0"}


def score_split(args, split, model, config, device, geometry):

    dataset, specs = load_split(args.data / split, args.condition, config["input_size"])
    records = json.loads((args.data / f"{split}.json").read_text(encoding="utf-8"))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    rows, logits_all, labels_all, kinds_all, batch_seconds = [], [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for masks, labels, indices in loader:
            batch_specs = [specs[index] for index in indices.tolist()]
            _synchronize(device)
            tick = time.perf_counter()
            masks = masks.to(device)
            inputs = build_classifier_input(masks, batch_specs)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
            logits = logits.float().cpu()
            scores = -torch.sigmoid(logits)
            _synchronize(device)
            batch_seconds.append(time.perf_counter() - tick)
            for offset, index in enumerate(indices.tolist()):
                record, spec = records[index], specs[index]
                identifier = record["id"]
                key = (split, identifier)
                defect = bool(labels[offset])
                if args.geometry_csv is not None:
                    if key not in geometry:
                        raise ValueError(f"Missing geometry score for {key}")
                    row = dict(geometry[key])
                    if row["kind"] != spec["kind"]:
                        raise ValueError(f"Geometry kind mismatch for {key}")
                    if "defect" in row and _csv_boolean(row["defect"]) != defect:
                        raise ValueError(
                            f"Geometry/reference defect mismatch for {key}"
                        )
                    for method in args.geometry_methods:
                        row[method] = float(row[method])
                else:
                    row = {}
                row.update(
                    {
                        "split": split,
                        "id": identifier,
                        "kind": spec["kind"],
                        "condition": args.condition,
                        "defect": defect,
                        "profile": args.profile or args.condition,
                        "classifier": float(scores[offset]),
                        "classifier_logit": float(logits[offset]),
                        "classifier_predicted_defect": bool(logits[offset] > 0),
                    }
                )
                rows.append(row)
                logits_all.append(float(logits[offset]))
                labels_all.append(defect)
                kinds_all.append(0 if spec["kind"] == "bridge" else 1)
            if len(rows) % 64 == 0 or len(rows) == len(dataset):
                print(
                    json.dumps(
                        {"split": split, "scored": len(rows), "total": len(dataset)}
                    ),
                    flush=True,
                )
    return rows, {
        "classification": summarize_logits(logits_all, labels_all, kinds_all),
        "inference_seconds": sum(batch_seconds),
        "batch_seconds": batch_seconds,
        "total_seconds_including_lazy_unpack_and_resize": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen classifier and optionally join existing geometry scores."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("focus", "defocus"), default="focus")
    parser.add_argument("--profile", help="Geometry CSV profile; defaults to condition")
    parser.add_argument("--geometry-csv", type=Path)
    parser.add_argument("--geometry-methods", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not 0 < args.alpha < 1:
        parser.error("Positive batch/threads and 0 < alpha < 1 required")
    if args.geometry_methods and args.geometry_csv is None:
        parser.error("geometry-methods requires geometry-csv")
    if any(
        (args.output / name).exists() for name in ("results.json", "per_sample.csv")
    ):
        raise FileExistsError(
            "Use a fresh output directory; previous results are preserved"
        )
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, checkpoint = load_classifier(args.classifier, device)
    geometry = _read_geometry(args.geometry_csv, args.profile or args.condition)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summaries = {}, {}
    for split in ("cal", "test"):
        rows[split], summaries[split] = score_split(
            args, split, model, checkpoint["config"], device, geometry
        )
        np.savez_compressed(
            args.output / f"{split}_scores.npz",
            **{
                key: np.asarray([row[key] for row in rows[split]])
                for key in ("id", "kind", "defect", "classifier", "classifier_logit")
            },
        )
    methods = list(dict.fromkeys([*args.geometry_methods, "classifier"]))
    alphas = sorted({0.01, 0.05, 0.1, args.alpha})
    primary, pooled = summarize_policies(rows["cal"], rows["test"], methods, alphas)
    all_rows = rows["cal"] + rows["test"]
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
    with (args.output / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    results = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "classifier_config": checkpoint["config"],
        "best_epoch": checkpoint["best_epoch"],
        "best_val_metrics": checkpoint.get("best_val_metrics"),
        "alphas": alphas,
        "split_summary": summaries,
        "primary_by_kind_calibrated": primary,
        "pooled_empirical": pooled,
        "notes": [
            "Scores are -sigmoid(defect logit); auto-pass is strictly score > threshold.",
            "Default batch_size=1 matches the online timing path in calibration and test.",
            "Each requested condition uses its own independent calibration split in this command.",
            "A shifted-condition test with frozen nominal thresholds must be reported separately.",
            "Null quantile means +infinity: no automatic pass for that kind.",
            "Packed2048-to512 inputs use exact cached uint8 area counts; no entire float dataset is materialized.",
            "Joined geometry scores are copied read-only; source results are never modified.",
            "Model architecture and checkpoint must be frozen before calling this script.",
        ],
    }
    (args.output / "results.json").write_text(
        json.dumps(json_finite(results), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"results": str(args.output / "results.json")}), flush=True)


if __name__ == "__main__":
    main()
