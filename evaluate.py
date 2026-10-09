import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from metrics import json_finite, summarize_policies
from geometry import evaluate_spec, geometric_margin, uniform_margin
from spectral_simulator import synthesize_intensity


PROFILES = {
    "focus": (0, 1.0),
    "defocus": (1, 1.0),
    "focus_low": (0, 0.98),
    "focus_high": (0, 1.02),
    "defocus_low": (1, 0.98),
    "defocus_high": (1, 1.02),
}
METHODS = [
    "aerial_margin",
    "truncated4_margin",
    "truncated1_margin",
    "spectral24_margin",
    "uniform_margin",
    "input_uniform_margin",
    "global_confidence",
]


def project_spec(spec, size=256):

    scale = 2048 / size
    terminals = [
        [int(np.ceil(value / scale)) for value in box] for box in spec["terminals"]
    ]
    if any(
        not (0 <= y0 < y1 <= size and 0 <= x0 < x1 <= size)
        for y0, x0, y1, x1 in terminals
    ):
        raise ValueError("Terminal must contain a sample on the frozen scoring grid")
    return {
        "kind": spec["kind"],
        "connectivity": spec.get("connectivity", 4),
        "terminals": terminals,
    }


def full_metrics(grids, reference, specs):

    truth = synthesize_intensity(reference) > 0.225
    actual_defects = [
        evaluate_spec(target, spec)["defect"]
        for target, spec in zip(truth.cpu().numpy(), specs)
    ]
    output = {}
    for name, grid in grids.items():
        prediction = synthesize_intensity(grid) > 0.225
        intersections = (prediction & truth).sum((1, 2))
        unions = (prediction | truth).sum((1, 2))
        ious = (
            torch.where(unions > 0, intersections / unions.clamp_min(1), 1.0)
            .cpu()
            .tolist()
        )
        central_prediction = prediction[:, 768:1280, 768:1280]
        central_truth = truth[:, 768:1280, 768:1280]
        central_union = (central_prediction | central_truth).sum((1, 2))
        central_intersection = (central_prediction & central_truth).sum((1, 2))
        central_iou = (
            torch.where(
                central_union > 0,
                central_intersection / central_union.clamp_min(1),
                1.0,
            )
            .cpu()
            .tolist()
        )

        predicted_defects = (
            [
                evaluate_spec(pred, spec)["defect"]
                for pred, spec in zip(prediction.cpu().numpy(), specs)
            ]
            if name == "aerial"
            else [None] * len(specs)
        )
        output[name] = list(zip(ious, predicted_defects, actual_defects, central_iou))
    return output


def evaluate_split(data, split, model, device, profiles, batch_size, score_size):
    records = json.loads((data / f"{split}.json").read_text(encoding="utf-8"))
    with np.load(data / f"{split}.npz", allow_pickle=False) as archive:
        features = archive["features"]
        references = archive["aerial_grid"]

        stride = 2048 // score_size
        packed = archive["masks_packed"]
        sampled_rows = packed[:, ::stride]
        if stride == 8:
            input_coarse = (sampled_rows >> 7) & 1
        else:
            input_coarse = np.unpackbits(sampled_rows, axis=-1)[:, :, ::stride]
        del packed, sampled_rows
    if features.shape != (len(records), 2, 5, 128, 128):
        raise ValueError("Expected N,2,5,128,128 features")
    if references.shape != (len(records), 2, 128, 128):
        raise ValueError("Expected N,2,128,128 reference intensities")
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            stop = min(start + batch_size, len(records))
            specs = [record["spec"] for record in records[start:stop]]
            projected = [project_spec(spec, score_size) for spec in specs]
            input_scores = [
                uniform_margin(input_coarse[index], spec)
                for index, spec in zip(range(start, stop), projected)
            ]
            for profile in profiles:
                condition, dose = PROFILES[profile]
                value = (
                    torch.from_numpy(features[start:stop, condition]).to(device).float()
                )
                reference = (
                    torch.from_numpy(references[start:stop, condition])
                    .to(device)
                    .float()
                    * dose**2
                )
                with torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    corrected = model.forward_features(value, condition)
                if corrected.ndim == 4:
                    corrected = corrected[:, 0]
                corrected = corrected.float() * dose**2
                approximations = {
                    "aerial": corrected,
                    "truncated4": value[:, :4].sum(1) * dose**2,
                    "truncated1": value[:, 0] * dose**2,
                }
                full = full_metrics(approximations, reference, specs)
                coarse = {
                    name: synthesize_intensity(grid, score_size).cpu().numpy()
                    for name, grid in approximations.items()
                }
                coarse["spectral24"] = (
                    synthesize_intensity(reference, score_size).cpu().numpy()
                )
                for offset, record in enumerate(records[start:stop]):
                    truth_defect = full["aerial"][offset][2]
                    if dose == 1.0:
                        stored = record[
                            "truth_focus" if condition == 0 else "truth_defocus"
                        ]["defect"]
                        if truth_defect != stored:
                            raise ValueError(
                                f"Stored truth mismatch: {split}/{record['id']}/{profile}"
                            )
                    elif f"truth_{profile}" in record:
                        if truth_defect != record[f"truth_{profile}"]["defect"]:
                            raise ValueError(
                                f"Stored endpoint truth mismatch: {split}/{record['id']}/{profile}"
                            )
                    row = {
                        "split": split,
                        "id": record["id"],
                        "group": record["group"],
                        "profile": profile,
                        "kind": record["spec"]["kind"],
                        "defect": truth_defect,
                        "predicted_defect": full["aerial"][offset][1],
                        "iou": full["aerial"][offset][0],
                        "critical_iou": full["aerial"][offset][3],
                        "truncated4_iou": full["truncated4"][offset][0],
                        "truncated1_iou": full["truncated1"][offset][0],
                        "truncated4_critical_iou": full["truncated4"][offset][3],
                        "truncated1_critical_iou": full["truncated1"][offset][3],
                    }
                    spec = projected[offset]
                    for name in ["aerial", "truncated4", "truncated1", "spectral24"]:
                        begin = time.perf_counter()
                        row[f"{name}_margin"] = geometric_margin(
                            coarse[name][offset] - 0.225, 1.0, spec
                        )
                        row[f"{name}_margin_seconds"] = time.perf_counter() - begin
                    predicted = coarse["aerial"][offset] > 0.225
                    begin = time.perf_counter()
                    row["uniform_margin"] = uniform_margin(predicted, spec)
                    row["uniform_margin_seconds"] = time.perf_counter() - begin
                    row["input_uniform_margin"] = input_scores[offset]
                    row["input_uniform_margin_seconds"] = 0.0
                    row["global_confidence"] = (
                        float(np.abs(coarse["aerial"][offset] - 0.225).mean())
                        if evaluate_spec(predicted, spec)["compliant"]
                        else -1.0
                    )
                    row["global_confidence_seconds"] = 0.0
                    rows.append(row)
            if stop % 64 == 0 or stop == len(records):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "scored_sources": stop,
                            "total": len(records),
                            "seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Frozen full-context evaluation with a fixed coarse geometric routing score."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles", nargs="+", choices=list(PROFILES), default=["focus", "defocus"]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "focus" not in args.profiles:
        parser.error("focus is required for the nominal policy transfer comparison")
    if args.score_size not in (256, 512, 1024, 2048):
        parser.error("score-size must preserve the physical grid and 9-pixel terminals")
    if (args.output / "results.json").exists():
        raise FileExistsError(f"Choose a fresh evaluation output: {args.output}")
    from model import load_model

    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = evaluate_split(
        args.data, "cal", model, device, args.profiles, args.batch_size, args.score_size
    )
    rows += evaluate_split(
        args.data,
        "test",
        model,
        device,
        args.profiles,
        args.batch_size,
        args.score_size,
    )
    profiles = {}
    for profile in args.profiles:
        cal_rows = [
            row for row in rows if row["split"] == "cal" and row["profile"] == profile
        ]
        test_rows = [
            row for row in rows if row["split"] == "test" and row["profile"] == profile
        ]
        policies, _ = summarize_policies(
            cal_rows, test_rows, METHODS, [0.01, 0.05, 0.1]
        )
        nominal_cal = [
            row for row in rows if row["split"] == "cal" and row["profile"] == "focus"
        ]
        nominal_policies, _ = summarize_policies(
            nominal_cal, test_rows, METHODS, [0.01, 0.05, 0.1]
        )
        profiles[profile] = {
            "n_test_sources": len(test_rows),
            "test_defects": sum(row["defect"] for row in test_rows),
            "mean_iou": {
                key: float(np.mean([row[key] for row in test_rows]))
                for key in [
                    "iou",
                    "critical_iou",
                    "truncated4_iou",
                    "truncated1_iou",
                    "truncated4_critical_iou",
                    "truncated1_critical_iou",
                ]
            },
            "primary_by_kind_calibrated": policies,
            "nominal_calibration_transferred": nominal_policies,
        }
    with (args.output / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "best_epoch": checkpoint["best_epoch"],
        "profiles": profiles,
        "notes": [
            "All full-resolution truth and IoU use all 24 public SOCS modes.",
            "Coarse geometric score is a ranking proxy; no full-resolution deterministic certificate is claimed.",
            "Scores and sample-center terminal mapping were fixed before cal/test evaluation.",
            "Per-profile calibration uses independent cal/test source groups with fixed kind quotas.",
            "Nominal threshold transfer is descriptive under process shift, without exchangeability guarantee.",
            "Seeds and process profiles share source layouts and are not extra independent test sources.",
            "Cached features are for offline evaluation; deployment timing must recompute features from the mask.",
        ],
    }
    (args.output / "results.json").write_text(
        json.dumps(json_finite(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profiles": {key: value["mean_iou"] for key, value in profiles.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
