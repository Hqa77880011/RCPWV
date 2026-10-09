import argparse
import csv
import html
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_window import (
    COUNTERS,
    ENDPOINT_CONFIG,
    METHODS,
    WindowQuery,
    online_classifier_score,
    synchronize,
)
from classifier import load_classifier
from evaluate import project_spec
from public_data import load_public
from geometry import evaluate_spec, geometric_margin, uniform_margin
from model import load_model
from spectral_simulator import SpectralSimulator, synthesize_intensity
from calibrate_window import ENDPOINTS


ROOT = Path(__file__).resolve().parent
SCORE_SIZE = 256
BISECTION_STEPS = 10


def read_thresholds(path, model_path, classifier_path, alpha):

    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["seed"] != 0 or tuple(policy["endpoints"]) != ENDPOINTS:
        raise ValueError("Expected the frozen seed0 four-endpoint window policy")
    evaluation = json.loads(
        (Path(policy["config"]["evaluation"]) / "results.json").read_text(
            encoding="utf-8"
        )
    )
    classifier_result = json.loads(
        Path(policy["config"]["classifier"])
        .with_name("results.json")
        .read_text(encoding="utf-8")
    )
    if Path(evaluation["config"]["model"]).resolve() != model_path.resolve():
        raise ValueError(
            "Model differs from the frozen critical calibration checkpoint"
        )
    if (
        evaluation["config"]["batch_size"] != 1
        or evaluation["config"]["score_size"] != SCORE_SIZE
    ):
        raise ValueError("Require the same batch1/256-grid calibration computation")
    config = classifier_result["config"]
    if Path(config["classifier"]).resolve() != classifier_path.resolve():
        raise ValueError("Classifier differs from frozen calibration")
    if config["batch_size"] != 1 or config["condition"] != "focus":
        raise ValueError("Require frozen nominal-focus classifier scores at batch1")
    thresholds = {}
    for method in METHODS[1:]:
        values = policy["primary_by_kind_calibrated"][method][str(alpha)]["thresholds"]
        thresholds[method] = {
            kind: float("inf") if value is None else float(value)
            for kind, value in values.items()
        }
    return thresholds


@torch.inference_mode()
def offline_layout(mask, record, model, classifier, simulator, device, thresholds):

    specs = record["all_specs"]
    query = WindowQuery(mask, specs[0], model, simulator, device, SCORE_SIZE)
    grids = {
        method: {condition: query.get_grid(condition, method) for condition in (0, 1)}
        for method in ("aerial_margin", "truncated4_margin", "spectral24_margin")
    }
    full_truth = {}
    fields = {method: {} for method in grids}
    for profile, (condition, dose) in zip(ENDPOINTS, ENDPOINT_CONFIG):
        full_truth[profile] = (
            (
                synthesize_intensity(grids["spectral24_margin"][condition] * dose**2)[0]
                > 0.225
            )
            .cpu()
            .numpy()
        )
        for method in grids:
            fields[method][profile] = (
                synthesize_intensity(grids[method][condition] * dose**2, SCORE_SIZE)[0]
                .cpu()
                .numpy()
                - 0.225
            )
    nominal_field = (
        synthesize_intensity(grids["aerial_margin"][0], SCORE_SIZE)[0].cpu().numpy()
        - 0.225
    )
    coarse_mask = mask[::8, ::8]
    rows = []
    for index, spec in enumerate(specs):
        projected = project_spec(spec, SCORE_SIZE)
        endpoint_labels = {
            profile: evaluate_spec(truth, spec)["defect"]
            for profile, truth in full_truth.items()
        }
        scores = {}
        for method in fields:
            scores[method] = min(
                geometric_margin(
                    fields[method][profile], 1.0, projected, max_iter=BISECTION_STEPS
                )
                for profile in ENDPOINTS
            )
        scores["nominal_aerial_only"] = geometric_margin(
            nominal_field, 1.0, projected, max_iter=BISECTION_STEPS
        )
        scores["input_uniform_margin"] = uniform_margin(
            coarse_mask, projected, max_iter=BISECTION_STEPS
        )
        scores["classifier"] = online_classifier_score(
            mask, spec, classifier, device, dict.fromkeys(COUNTERS, 0)
        )
        defect = any(endpoint_labels.values())
        row = {
            "id": record["id"],
            "check_id": f"{record['id']}_{index}",
            "source_group": record["source_group"],
            "source": record["source"],
            "cell": record.get("cell", ""),
            "spec_index": index,
            "kind": spec["kind"],
            "terminals": json.dumps(spec["terminals"]),
            "window_defect": defect,
            **{
                f"defect_{profile}": label for profile, label in endpoint_labels.items()
            },
        }
        for method in METHODS:
            threshold = (
                None if method == "all_spectral24" else thresholds[method][spec["kind"]]
            )
            passed = False if method == "all_spectral24" else scores[method] > threshold
            row[f"{method}_score"] = scores.get(method)
            row[f"{method}_threshold"] = (
                threshold if threshold is not None and np.isfinite(threshold) else None
            )
            row[f"{method}_threshold_finite"] = (
                bool(np.isfinite(threshold)) if threshold is not None else None
            )
            row[f"{method}_passed"] = bool(passed)
            row[f"{method}_missed"] = bool(passed and defect)
        rows.append(row)
    return rows


def outcome_counts(rows, method):

    n = len(rows)
    n_defects = sum(row["window_defect"] for row in rows)
    passed = sum(row[f"{method}_passed"] for row in rows)
    missed = sum(row[f"{method}_missed"] for row in rows)
    healthy = n - n_defects
    correct = passed - missed
    return {
        "n_checks": n,
        "n_window_defects": n_defects,
        "n_healthy": healthy,
        "n_passed": passed,
        "n_referred": n - passed,
        "n_missed_defects": missed,
        "n_healthy_correctly_autoaccepted": correct,
        "descriptive_pass_rate": passed / n if n else None,
        "descriptive_defect_miss_rate": missed / n_defects if n_defects else None,
        "descriptive_healthy_acceptance_rate": correct / healthy if healthy else None,
    }


def correlated_summary(rows, method):

    groups = sorted({row["source_group"] for row in rows})
    by_source = {}
    for group in groups:
        group_rows = [row for row in rows if row["source_group"] == group]
        by_source[group] = {
            "n_layouts": len({row["id"] for row in group_rows}),
            **outcome_counts(group_rows, method),
        }
    macro = {}
    for metric in (
        "descriptive_pass_rate",
        "descriptive_defect_miss_rate",
        "descriptive_healthy_acceptance_rate",
    ):
        values = [
            source[metric]
            for source in by_source.values()
            if source[metric] is not None
        ]
        macro[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "n_source_groups_with_denominator": len(values),
        }
    return {
        **outcome_counts(rows, method),
        "n_layouts": len({row["id"] for row in rows}),
        "n_source_groups": len(groups),
        "source_macro": macro,
        "by_source_group": by_source,
        "by_kind": {
            kind: outcome_counts([row for row in rows if row["kind"] == kind], method)
            for kind in ("bridge", "break")
        },
    }


def write_report(path, result):
    rows = []
    for method, values in result["methods"].items():
        rows.append(
            f"<tr><td>{html.escape(method)}</td>"
            f"<td>{values['n_healthy_correctly_autoaccepted']}/{values['n_healthy']}</td>"
            f"<td>{values['n_missed_defects']}/{values['n_window_defects']}</td>"
            f"<td>{values['median_seconds']:.4f}</td>"
            f"<td>{values['speedup_vs_optimized_window_reference']:.3f}</td></tr>"
        )
    content = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>RCPWV public results</title>'
        "<style>body{font-family:Arial;max-width:1100px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd7dc;text-align:left}</style>"
        "<h1>Public process-window verification</h1>"
        "<p>Frozen critical seed-0 policies at a 5% budget. Public layout variants and specifications share source groups. Event rates and source-group macro rates are descriptive under source shift.</p>"
        "<table><tr><th>Policy</th><th>Healthy accepted</th><th>Defects missed</th><th>Median seconds</th><th>Speedup</th></tr>"
        + "".join(rows)
        + "</table><p>checks.csv contains per-specification endpoint labels, scores and decisions. summary.json contains source-group rates, all timing repeats and work counters.</p></html>"
    )
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Frozen critical-window policies on existing public GDS specifications."
    )
    parser.add_argument(
        "--public-data", type=Path, default=ROOT / "work/data/ordinary/public"
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "work/runs/k4_seed0/best.pt"
    )
    parser.add_argument(
        "--classifier", type=Path, default=ROOT / "work/runs/classifier_seed0/best.pt"
    )
    parser.add_argument(
        "--window-policy",
        type=Path,
        default=ROOT / "work/results/critical/window_seed0.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work/results/public")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.alpha != 0.05 or args.repeats != 3:
        parser.error(
            "This frozen external protocol uses alpha0.05 and exactly3 timing repeats"
        )
    if args.output.exists():
        raise FileExistsError(
            "External window results already exist; preserve the saved evaluation"
        )
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, model_checkpoint = load_model(args.model, device)
    classifier, classifier_checkpoint = load_classifier(args.classifier, device)
    if (
        model_checkpoint["config"]["seed"] != 0
        or model_checkpoint["config"]["kernels"] != 4
    ):
        raise ValueError("Expected the unchanged K4 seed0 checkpoint")
    if (
        classifier_checkpoint["config"]["seed"] != 0
        or classifier_checkpoint["config"]["input_size"] != 512
    ):
        raise ValueError("Expected the unchanged seed0 512-area classifier checkpoint")
    simulator = SpectralSimulator(device=device)
    thresholds = read_thresholds(
        args.window_policy, args.model, args.classifier, args.alpha
    )
    records, masks, _ = load_public(args.public_data)
    queries = [
        (index, spec)
        for index, record in enumerate(records)
        for spec in record["all_specs"]
    ]
    if (
        len(records),
        len({record["source_group"] for record in records}),
        len(queries),
    ) != (20, 9, 80):
        raise ValueError(
            "Expected the predeclared20 layouts/9 source groups/80 specifications"
        )
    rows = []
    started_all = time.perf_counter()
    for index, record in enumerate(records):
        rows.extend(
            offline_layout(
                masks[index], record, model, classifier, simulator, device, thresholds
            )
        )
        print(
            json.dumps(
                {
                    "event": "offline_public_window",
                    "completed_layouts": index + 1,
                    "total_layouts": len(records),
                }
            ),
            flush=True,
        )
    if len({row["check_id"] for row in rows}) != len(queries):
        raise ValueError("Public query IDs are not unique")

    def operation(method, index):
        layout_index, spec = queries[index]
        threshold = (
            None if method == "all_spectral24" else thresholds[method][spec["kind"]]
        )
        query = WindowQuery(
            masks[layout_index], spec, model, simulator, device, SCORE_SIZE
        )
        if method == "classifier":
            score = online_classifier_score(
                masks[layout_index], spec, classifier, device, query.counts
            )
            passed = score > threshold
            reference = None if passed else query.reference_window_defect()
            return passed, reference, query.counts, score
        return (*query.execute(method, threshold), None)

    times = {method: [] for method in METHODS}
    work = {method: [] for method in METHODS}
    prior_query_work = {}
    classifier_differences = []
    with torch.inference_mode():
        for method in METHODS:
            for _ in range(2):
                operation(method, 0)
        for repeat in range(args.repeats):
            offset = repeat % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                synchronize(device)
                start = time.perf_counter()
                values = [operation(method, index) for index in range(len(queries))]
                synchronize(device)
                seconds = time.perf_counter() - start
                for index, (passed, reference_defect, counts, score) in enumerate(
                    values
                ):
                    row = rows[index]
                    if passed != row[f"{method}_passed"]:
                        raise AssertionError(
                            f"Online strict route mismatch: {method}/{row['check_id']}"
                        )
                    if (
                        reference_defect is not None
                        and reference_defect != row["window_defect"]
                    ):
                        raise AssertionError(
                            f"Full reference union mismatch: {method}/{row['check_id']}"
                        )
                    if counts["reference_calls"] != int(not passed):
                        raise AssertionError(
                            "Reference call count differs from actual fallback"
                        )
                    if (
                        counts["input_ffts"] > 1
                        or counts["coherent_modes_focus"] > 24
                        or counts["coherent_modes_defocus"] > 24
                    ):
                        raise AssertionError(
                            "Shared optical work was recomputed within a query"
                        )
                    key = (method, index)
                    if key in prior_query_work and prior_query_work[key] != counts:
                        raise AssertionError(
                            f"Query work counts changed: {method}/{row['check_id']}"
                        )
                    prior_query_work[key] = counts.copy()
                counts = {
                    key: sum(value[2][key] for value in values) for key in COUNTERS
                }
                if method == "classifier":
                    classifier_differences.append(
                        max(
                            abs(value[3] - row["classifier_score"])
                            for value, row in zip(values, rows)
                        )
                    )
                times[method].append(seconds)
                work[method].append(counts)
                print(
                    json.dumps(
                        {
                            "event": "public_window_timing",
                            "repeat": repeat + 1,
                            "method": method,
                            "seconds": seconds,
                            **outcome_counts(rows, method),
                            "work_counts": counts,
                        }
                    ),
                    flush=True,
                )
    baseline = float(np.median(times["all_spectral24"]))
    methods = {}
    for method in METHODS:
        median = float(np.median(times[method]))
        methods[method] = {
            **correlated_summary(rows, method),
            "all_seconds": times[method],
            "median_seconds": median,
            "speedup_vs_optimized_window_reference": baseline / median,
            "actual_work_by_repeat": work[method],
        }
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "n_layouts": len(records),
        "n_source_groups": 9,
        "n_correlated_specification_events": len(rows),
        "endpoints": list(ENDPOINTS),
        "score_size": SCORE_SIZE,
        "bisection_steps": BISECTION_STEPS,
        "threshold_rule": "score > frozen kind threshold; no public recalibration",
        "thresholds": {
            method: {
                kind: value if np.isfinite(value) else None
                for kind, value in values.items()
            }
            for method, values in thresholds.items()
        },
        "methods": methods,
        "guarantee_transferred": False,
        "classifier_max_abs_score_difference_vs_offline_per_repeat": classifier_differences,
        "checks": {
            "strict_route_matches_offline_scores": True,
            "fallback_union_matches_full_endpoint_labels": True,
            "work_counts_stable_per_query_every_repeat": True,
        },
        "sources": [
            {
                key: record[key]
                for key in ("id", "source", "source_group", "cell")
                if key in record
            }
            for record in records
        ],
        "timing_scope": "Preloaded full uint8 source mask to decision; online preprocessing, H2D, shared input FFT, actual modes/CNN or classifier, CPU geometry and actual full-reference fallback. No across-query reuse. Excludes file/model loading, training/calibration and offline score/label construction.",
        "notes": [
            "True public GDS inputs; all wafer/window labels are full24-mode SOCS simulated labels.",
            "Four endpoint labels are generated by grid*dose^2 then2048 reconstruction, the frozen evaluation path.",
            "The two optical kernels are discrete supplied focus/defocus conditions, not continuous defocus coverage.",
            "Specifications/layout variants share sources; source macro rates exclude groups lacking each conditional denominator.",
            "OOD counts are descriptive; no IID confidence interval or calibrated risk guarantee is asserted.",
            "The fixed existing list and original order are used for every method and repeat.",
            "Optimized fallback reuses modes/grids within each query and checks finite-precision union-label equality against all four full-resolution endpoint labels.",
        ],
        "total_seconds": time.perf_counter() - started_all,
    }
    args.output.mkdir(parents=True)
    with (args.output / "checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report(args.output / "RESULTS.html", result)
    print(
        json.dumps(
            {
                "event": "external_window_complete",
                "output": str(args.output),
                "n_window_defects": sum(row["window_defect"] for row in rows),
                "main": methods["aerial_margin"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
