import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_window import COUNTERS, WindowQuery
from geometry import evaluate_spec, margin_pass
from model import load_model
from simulator import Simulator
from spectral_simulator import SpectralSimulator, synthesize_intensity


def nominal_operation(
    mask, spec, model, fast, original, device, score_size, method, threshold=None
):

    query = WindowQuery(mask, spec, model, fast, device, score_size)
    if method == "all_original24":
        tensor = torch.from_numpy(mask[None]).to(device, dtype=torch.float32)
        query.counts["h2d_transfers"] += 1
        query.counts["input_ffts"] += 1
        query.counts["coherent_ifft_batches"] += (
            24 + original.kernel_chunk - 1
        ) // original.kernel_chunk
        query.counts["coherent_modes_focus"] += 24
        actual = original(tensor)[0].cpu().numpy()
    else:
        if method == "input_uniform_margin":
            passed = query.route(method, threshold)
        elif method == "all_spectral24":
            passed = False
        else:
            grid = query.get_grid(0, method)
            coarse = synthesize_intensity(grid.float(), score_size)[0].cpu().numpy()
            query.counts["coarse_reconstructions"] += 1
            query.counts["coarse_margin_checks"] += 1
            passed = margin_pass(coarse - 0.225, 1.0, query.projected, threshold)
        if passed:
            return True, None, query.counts

        grid = query.get_grid(0, "spectral24_margin")
        actual = (synthesize_intensity(grid.float())[0] > 0.225).cpu().numpy()
        query.counts["full_reconstructions"] += 1
    query.counts["reference_calls"] += 1
    query.counts["full_connectivity_checks"] += 1
    return False, bool(evaluate_spec(actual, spec)["defect"]), query.counts


def main():
    parser = argparse.ArgumentParser(
        description="Time nominal routing with one shared input FFT and reused optical modes."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or not 0 < args.alpha < 1:
        parser.error("samples/repeats must be positive and alpha must be in (0,1)")
    if args.output.exists():
        raise FileExistsError(
            "Nominal timing already exists; retain the saved measurement"
        )
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, _ = load_model(args.model, device)
    original, fast = Simulator(device, 24), SpectralSimulator(device)
    result = json.loads((args.evaluation / "results.json").read_text(encoding="utf-8"))
    if Path(result["config"]["model"]).resolve() != args.model.resolve():
        raise ValueError(
            "Benchmark checkpoint differs from frozen evaluation checkpoint"
        )
    score_size = result["config"]["score_size"]
    policies = result["profiles"]["focus"]["primary_by_kind_calibrated"]
    records = json.loads((args.data / "test.json").read_text(encoding="utf-8"))[
        : args.samples
    ]
    if len(records) != args.samples:
        raise ValueError("Requested more sources than the frozen test split contains")
    with np.load(args.data / "test.npz", allow_pickle=False) as archive:
        masks = np.unpackbits(archive["masks_packed"][: args.samples], axis=-1)
    specs = [record["spec"] for record in records]
    saved = {}
    with (args.evaluation / "per_sample.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test" and row["profile"] == "focus":
                saved[row["id"]] = row
    methods = [
        "all_original24",
        "all_spectral24",
        "aerial_margin",
        "truncated4_margin",
        "spectral24_margin",
        "input_uniform_margin",
    ]
    outcomes, timings = {}, {method: [] for method in methods}
    work = {method: [] for method in methods}

    def threshold_for(method, kind):
        value = policies[method][str(args.alpha)]["thresholds"][kind]
        return float("inf") if value is None else value

    def operation(method, index):
        threshold = (
            None
            if method.startswith("all_")
            else threshold_for(method, specs[index]["kind"])
        )
        return nominal_operation(
            masks[index],
            specs[index],
            model,
            fast,
            original,
            device,
            score_size,
            method,
            threshold,
        )

    defects = [record["truth_focus"]["defect"] for record in records]
    expected = {
        method: [
            float(saved[record["id"]][method])
            > threshold_for(method, record["spec"]["kind"])
            for record in records
        ]
        for method in methods
        if not method.startswith("all_")
    }

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
                begin = time.perf_counter()
                values = [operation(method, index) for index in range(len(records))]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - begin
                timings[method].append(seconds)
                passes = [value[0] for value in values]
                if not method.startswith("all_"):
                    mismatches = sum(a != b for a, b in zip(passes, expected[method]))
                    if mismatches:
                        raise AssertionError(
                            f"Online routing differs from frozen evaluation: {method}/{mismatches}"
                        )
                if any(
                    value[1] is not None and value[1] != defect
                    for value, defect in zip(values, defects)
                ):
                    raise AssertionError(
                        f"Actual reference/fallback connectivity mismatch: {method}"
                    )
                counts = {
                    key: sum(value[2][key] for value in values) for key in COUNTERS
                }
                summary = {
                    "n_samples": len(records),
                    "n_passed": sum(passes),
                    "reference_calls": len(records) - sum(passes),
                    "n_defects": sum(defects),
                    "n_defects_missed": sum(
                        passed and defect for passed, defect in zip(passes, defects)
                    ),
                }
                if method in outcomes and outcomes[method] != summary:
                    raise AssertionError(
                        f"Pipeline outcomes changed across timing repeats: {method}"
                    )
                if work[method] and work[method][0] != counts:
                    raise AssertionError(
                        f"Actual work counts changed across timing repeats: {method}"
                    )
                outcomes[method] = summary
                work[method].append(counts)
                print(
                    json.dumps(
                        {
                            "repeat": repeat + 1,
                            "method": method,
                            "seconds": seconds,
                            **summary,
                            "work_counts": counts,
                        }
                    ),
                    flush=True,
                )
    baseline = float(np.median(timings["all_spectral24"]))
    for method in methods:
        outcomes[method].update(
            {
                "all_seconds": timings[method],
                "median_seconds": float(np.median(timings[method])),
                "speedup_vs_optimized_full_reference": baseline
                / float(np.median(timings[method])),
                "actual_work_by_repeat": work[method],
            }
        )
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methods": outcomes,
        "checks": {
            "strict_frozen_route_match_every_repeat": True,
            "actual_reference_and_fallback_labels_match_every_repeat": True,
            "work_counts_stable_every_repeat": True,
        },
        "scope": "Preloaded uint8 full masks to nominal decision/reference connectivity; includes CPU-to-GPU transfer, one shared input FFT, required optical modes, CNN, coarse CPU geometry, actual fast fallback, full-resolution reference connectivity. Excludes file IO and offline training/calibration.",
        "notes": [
            "All methods use identical fixed first test sources and nominal focus condition.",
            "Routing decisions must exactly match frozen offline scores and strict thresholds.",
            "The optimized full 24-mode reference is the primary speed baseline.",
            "No predicted full-resolution image is needed for an accepted routing query.",
            "Each query computes at most one input FFT and each required mode at most once.",
            "Rejected K4/CNN queries append remaining modes in original order and jointly sum all 24; spectral24 fallback reuses its grid.",
            "Full reconstructions count intensity Fourier synthesis; original24 instead performs full-grid coherent IFFTs in its original chunks.",
            "Every repeat checks strict saved route decisions, all actual reference/fallback labels and stable work counts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(outcomes), flush=True)


if __name__ == "__main__":
    main()
