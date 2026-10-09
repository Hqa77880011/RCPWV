import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from classifier import build_classifier_input, load_classifier
from geometry import evaluate_spec
from spectral_simulator import SpectralSimulator


def online_area_occupancy(mask: np.ndarray) -> np.ndarray:

    if mask.dtype != np.uint8 or mask.shape != (2048, 2048):
        raise ValueError("Expected a full 2048x2048 uint8 mask")

    counts = mask.reshape(512, 4, 512, 4).sum(axis=(1, 3), dtype=np.uint8)
    return counts.astype(np.float32) / 16.0


def main():

    parser = argparse.ArgumentParser(
        description="Time online direct-classifier routing against optimized full SOCS verification."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or not 0 < args.alpha < 1:
        parser.error("samples/repeats must be positive and alpha must lie in (0,1)")
    if args.output.exists():
        raise FileExistsError("Use a fresh benchmark output file")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, checkpoint = load_classifier(args.classifier, device)
    if checkpoint["config"]["input_size"] != 512:
        raise ValueError("This benchmark matches the frozen 512-occupancy classifier")
    reference = SpectralSimulator(device)
    result = json.loads((args.evaluation / "results.json").read_text(encoding="utf-8"))
    policies = result["primary_by_kind_calibrated"]["classifier"][str(args.alpha)]
    if result["config"]["condition"] != "focus":
        raise ValueError("Benchmark requires a frozen nominal-focus evaluation")
    thresholds = {
        kind: math.inf if value is None else float(value)
        for kind, value in policies["thresholds"].items()
    }
    records = json.loads((args.data / "test.json").read_text(encoding="utf-8"))[
        : args.samples
    ]
    if len(records) != args.samples:
        raise ValueError("Requested more sources than the frozen test split contains")
    with np.load(args.data / "test.npz", allow_pickle=False) as archive:
        masks = np.unpackbits(
            archive["masks_packed"][: args.samples], axis=-1, count=2048, bitorder="big"
        )
    if masks.dtype != np.uint8 or masks.shape != (len(records), 2048, 2048):
        raise ValueError("Frozen test masks have unexpected shape or dtype")
    specs = [record["spec"] for record in records]
    classifier_specs = [{**spec, "source_shape": [2048, 2048]} for spec in specs]
    defects = [bool(record["truth_focus"]["defect"]) for record in records]
    saved = {}
    with (args.evaluation / "per_sample.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] != "test"
                or (row.get("profile") or row.get("condition")) != "focus"
            ):
                continue
            if row["id"] in saved:
                raise ValueError(f"Duplicate nominal test score: {row['id']}")
            saved[row["id"]] = row
    expected, saved_scores = [], []
    for record, spec in zip(records, specs):
        row = saved[record["id"]]
        if row["kind"] != spec["kind"]:
            raise ValueError(
                "Frozen classifier kind differs from dataset specification"
            )
        score = float(row["classifier"])
        saved_scores.append(score)
        expected.append(score > thresholds[spec["kind"]])

    def full_reference(index):
        mask = torch.from_numpy(masks[index : index + 1]).to(
            device, dtype=torch.float32
        )
        actual = reference(mask)[0].cpu().numpy()
        return bool(evaluate_spec(actual, specs[index])["defect"])

    def operation(method, index):
        if method == "all_spectral24":
            return False, full_reference(index), None
        area = online_area_occupancy(masks[index])

        area_tensor = torch.from_numpy(area[None, None]).to(device)
        inputs = build_classifier_input(area_tensor, [classifier_specs[index]])
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit = model(inputs)
        score = float(-torch.sigmoid(logit.float().cpu())[0])
        passed = score > thresholds[specs[index]["kind"]]
        if passed:
            return True, False, score
        return False, full_reference(index), score

    methods = ["all_spectral24", "classifier"]
    timings = {method: [] for method in methods}
    outcomes, score_differences = {}, []
    with torch.inference_mode():
        for method in methods:
            for _ in range(3):
                operation(method, 0)
        for repeat in range(args.repeats):
            ordered = (
                methods[repeat % len(methods) :] + methods[: repeat % len(methods)]
            )
            for method in ordered:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                values = [operation(method, index) for index in range(len(records))]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                timings[method].append(elapsed)
                passes = [value[0] for value in values]
                if method == "classifier":
                    mismatches = sum(
                        observed != frozen for observed, frozen in zip(passes, expected)
                    )
                    if mismatches:
                        raise AssertionError(
                            f"Online classifier routing differs from frozen scores: {mismatches}"
                        )
                    score_differences.append(
                        max(
                            abs(value[2] - saved_score)
                            for value, saved_score in zip(values, saved_scores)
                        )
                    )

                if any(
                    (not value[0]) and value[1] != defect
                    for value, defect in zip(values, defects)
                ):
                    raise AssertionError(
                        f"Full reference connectivity mismatch in {method}"
                    )
                counts = {
                    "n_samples": len(records),
                    "n_passed": sum(passes),
                    "reference_calls": len(records) - sum(passes),
                    "n_defects": sum(defects),
                    "n_defects_missed": sum(
                        passed and defect for passed, defect in zip(passes, defects)
                    ),
                }
                if repeat == 0:
                    outcomes[method] = counts
                elif counts != outcomes[method]:
                    raise AssertionError(
                        f"Pipeline outcomes changed across timing repeats: {method}"
                    )
                print(
                    json.dumps(
                        {
                            "repeat": repeat + 1,
                            "method": method,
                            "seconds": elapsed,
                            **counts,
                        }
                    ),
                    flush=True,
                )
    baseline = float(np.median(timings["all_spectral24"]))
    for method in methods:
        median = float(np.median(timings[method]))
        outcomes[method].update(
            {
                "all_seconds": timings[method],
                "median_seconds": median,
                "median_seconds_per_source": median / len(records),
                "speedup_vs_optimized_full_reference": baseline / median,
            }
        )
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methods": outcomes,
        "source_ids": [record["id"] for record in records],
        "condition": "focus",
        "batch_size": 1,
        "classifier_best_epoch": checkpoint["best_epoch"],
        "thresholds": {
            kind: value if math.isfinite(value) else None
            for kind, value in thresholds.items()
        },
        "frozen_decision_mismatches": 0,
        "max_abs_score_difference_vs_frozen_per_repeat": score_differences,
        "scope": (
            "Preloaded uint8 full2048 mask to decision/reference connectivity. Includes online exact "
            "4x4 sum/16 occupancy preprocessing, float32 H2D, original four-channel terminal/kind "
            "construction, AMP classifier, CPU float32 sigmoid, strict threshold, actual spectral24 "
            "fallback from the original full mask, D2H, and CPU full-resolution connectivity. "
            "Excludes file IO, packed-mask loading/unpacking, offline training and calibration."
        ),
        "notes": [
            "Uses the same fixed first test sources and nominal focus condition as benchmark_nominal.py.",
            "Two methods are interleaved in rotating order after three warmup calls per method.",
            "No cached area feature, cached network output or fallback reference output is used online.",
            "Every repeat's pass decisions must equal frozen CSV scores under the strict thresholds.",
            "Raw AMP scores can differ slightly with inference batch size; such differences are reported.",
            "All online reference and rejected-classifier truth checks must match truth_focus.",
            "Speedup uses the optimized 24-mode spectral reference measured in this same run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output.resolve()), "methods": outcomes}),
        flush=True,
    )


if __name__ == "__main__":
    main()
