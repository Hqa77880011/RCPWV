import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import METHODS
from metrics import aggregate_stratified, json_finite
from calibrate_window import ENDPOINTS


SEEDS = (0, 1, 2)
KINDS = ("bridge", "break")
WINDOW_METHODS = [*METHODS, "nominal_aerial_only", "classifier"]
METRICS = (
    "pass_rate",
    "defect_miss_rate",
    "healthy_acceptance_rate",
    "defects_among_passed",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def statistics(values):
    finite = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "n_runs": len(finite),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_rows(root, seed):
    geometry = read_csv(root / f"k4_seed{seed}/per_sample.csv")
    classifier = {
        (row["split"], row["id"]): float(row["classifier"])
        for row in read_csv(root / f"classifier_seed{seed}/per_sample.csv")
        if row["profile"] == "focus"
    }
    grouped = {}
    for row in geometry:
        grouped.setdefault((row["split"], row["id"]), {})[row["profile"]] = row
    rows = []
    for key, profiles in grouped.items():
        if any(name not in profiles for name in (*ENDPOINTS, "focus")):
            raise ValueError(f"Missing window profile: {key}")
        row = {
            "split": key[0],
            "id": key[1],
            "kind": profiles["focus"]["kind"],
            "defect": any(
                profiles[name]["defect"].lower() == "true" for name in ENDPOINTS
            ),
        }
        row.update(
            {
                method: min(float(profiles[name][method]) for name in ENDPOINTS)
                for method in METHODS
            }
        )
        row["nominal_aerial_only"] = float(profiles["focus"]["aerial_margin"])
        row["classifier"] = classifier[key]
        rows.append(row)
    return rows


def decisions(rows, method, policy):
    thresholds = {
        kind: np.inf if value is None else float(value)
        for kind, value in policy["thresholds"].items()
    }
    return np.asarray([row[method] > thresholds[row["kind"]] for row in rows])


def paired_comparisons(rows_by_seed, policies, replicates):
    ordered = [row["id"] for row in rows_by_seed[0]]
    kinds = np.array([row["kind"] for row in rows_by_seed[0]])
    defects = np.array([row["defect"] for row in rows_by_seed[0]])
    aligned = []
    for rows in rows_by_seed:
        indexed = {row["id"]: row for row in rows}
        if set(indexed) != set(ordered):
            raise ValueError("Seed comparisons must use the same independent sources")
        current = [indexed[identifier] for identifier in ordered]
        if any(
            row["kind"] != kind or row["defect"] != defect
            for row, kind, defect in zip(current, kinds, defects)
        ):
            raise ValueError(
                "Kinds and window labels must agree across optimizer seeds"
            )
        aligned.append(current)
    means = {
        method: np.mean(
            [
                decisions(rows, method, policies[seed][method]["0.05"])
                for seed, rows in enumerate(aligned)
            ],
            axis=0,
        )
        for method in WINDOW_METHODS
    }
    rng = np.random.default_rng(20261010)
    sampled = np.concatenate(
        [
            rng.choice(indices, size=(replicates, len(indices)), replace=True)
            for kind in KINDS
            if len(indices := np.flatnonzero(kinds == kind))
        ],
        axis=1,
    )
    result = {}
    for baseline in WINDOW_METHODS[1:]:
        difference = means["aerial_margin"] - means[baseline]
        result[baseline] = {}
        for name, values in (
            ("total_acceptance_difference", difference),
            (
                "correct_healthy_acceptance_per_all_sources_difference",
                difference * ~defects,
            ),
        ):
            result[baseline][name] = {
                "mean": float(values.mean()),
                "paired_95_ci": np.quantile(
                    values[sampled].mean(1), [0.025, 0.975]
                ).tolist(),
            }
    return {
        "n_independent_sources": len(ordered),
        "resamples": replicates,
        "seed": 20261010,
        "thresholds_frozen": True,
        "comparisons": result,
    }


def policy_record(
    distribution, seed, method, alpha, rule, summary, kind="all", threshold=None
):
    return {
        "distribution": distribution,
        "seed": seed,
        "method": method,
        "alpha": alpha,
        "rule": rule,
        "kind": kind,
        "threshold": threshold,
        **{
            key: summary[key]
            for key in (
                "n",
                "n_defects",
                "n_healthy",
                "n_passed",
                "n_referred",
                "n_missed_defects",
                "n_healthy_accepted",
                *METRICS,
            )
        },
        **{f"{metric}_ci": json.dumps(summary[f"{metric}_ci"]) for metric in METRICS},
    }


def collect_window(root, distribution, replicates):
    all_rows, policies, table = [], [], []
    for seed in SEEDS:
        rows = [row for row in window_rows(root, seed) if row["split"] == "test"]
        policy = read_json(root / f"window_seed{seed}.json")[
            "primary_by_kind_calibrated"
        ]
        all_rows.append(rows)
        policies.append(policy)
        for method in WINDOW_METHODS:
            passed = np.asarray(
                [
                    row[method] > -0.5 if method == "classifier" else row[method] >= 0
                    for row in rows
                ]
            )
            defects = np.asarray([row["defect"] for row in rows])
            kinds = np.asarray([row["kind"] for row in rows])
            summary = aggregate_stratified(passed, defects, kinds)
            table.append(
                policy_record(distribution, seed, method, None, "uncalibrated", summary)
            )
            for alpha, entry in policy[method].items():
                table.append(
                    policy_record(
                        distribution,
                        seed,
                        method,
                        float(alpha),
                        "calibrated",
                        entry["test"],
                    )
                )
                for kind in KINDS:
                    table.append(
                        policy_record(
                            distribution,
                            seed,
                            method,
                            float(alpha),
                            "calibrated",
                            entry["by_kind"][kind],
                            kind,
                            entry["thresholds"][kind],
                        )
                    )
    aggregate = []
    for method in WINDOW_METHODS:
        for alpha in (None, 0.01, 0.05, 0.1):
            selected = [
                row
                for row in table
                if row["method"] == method
                and row["alpha"] == alpha
                and row["kind"] == "all"
            ]
            aggregate.append(
                {
                    "distribution": distribution,
                    "method": method,
                    "alpha": alpha,
                    "rule": "uncalibrated" if alpha is None else "calibrated",
                    "metrics": {
                        metric: statistics([row[metric] for row in selected])
                        for metric in METRICS
                    },
                }
            )
    return table, aggregate, paired_comparisons(all_rows, policies, replicates)


def plot_policies(aggregate, output):
    selected = [row for row in aggregate if row["alpha"] in (None, 0.05)]
    names = WINDOW_METHODS
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("healthy_acceptance_rate", "defect_miss_rate")):
        for offset, alpha, label, color in (
            (-0.18, None, "Before calibration", "#74838b"),
            (0.18, 0.05, "5% calibration", "#438982"),
        ):
            stats = [
                next(
                    row["metrics"][metric]
                    for row in selected
                    if row["method"] == name and row["alpha"] == alpha
                )
                for name in names
            ]
            heights = [
                np.nan if item["mean"] is None else item["mean"] for item in stats
            ]
            errors = [
                0 if item["sample_std"] is None else item["sample_std"]
                for item in stats
            ]
            axis.bar(
                np.arange(len(names)) + offset,
                heights,
                width=0.36,
                yerr=errors,
                label=label,
                color=color,
            )
        axis.set_xticks(
            np.arange(len(names)), names, rotation=60, ha="right", fontsize=8
        )
        axis.set_ylabel(metric)
        axis.set_ylim(0, 1.05)
        if metric == "defect_miss_rate":
            axis.axhline(0.05, linestyle="--", color="#426c88", linewidth=1)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output.with_suffix("." + suffix), dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate saved RCPWV fidelity, acceptance, timing and resource metrics."
    )
    parser.add_argument(
        "--work", type=Path, default=Path(__file__).resolve().parent / "work"
    )
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("resamples must be positive")
    output = args.work / "report"
    output.mkdir(parents=True, exist_ok=True)
    fidelity, nominal, resources, classifiers, timings = [], [], [], [], []
    ordinary = args.work / "results/ordinary"
    jobs = (
        [f"k4_seed{seed}" for seed in SEEDS]
        + [f"k0_seed{seed}" for seed in SEEDS]
        + ["k4_no_boundary_seed0"]
    )
    for name in jobs:
        result = read_json(ordinary / name / "results.json")
        for profile, values in result["profiles"].items():
            fidelity.append({"run": name, "profile": profile, **values["mean_iou"]})
            for rule in (
                "primary_by_kind_calibrated",
                "nominal_calibration_transferred",
            ):
                for method, budgets in values[rule].items():
                    for alpha, entry in budgets.items():
                        nominal.append(
                            policy_record(
                                profile,
                                result["config"].get("seed", name),
                                method,
                                float(alpha),
                                rule,
                                entry["test"],
                            )
                        )
    for name in [*jobs, *(f"classifier_seed{seed}" for seed in SEEDS)]:
        summary = read_json(args.work / "runs" / name / "summary.json")
        resources.append(
            {
                "run": name,
                "parameters": summary["config"]["parameters"],
                "best_epoch": summary["best_epoch"],
                "training_seconds": summary["total_seconds"],
                "peak_gpu_allocated_mib": summary["peak_gpu_allocated_mib"],
            }
        )
    fidelity_stats = []
    for profile in ("focus", "defocus"):
        for label, prefix, full_key, central_key in (
            ("K4+CNN", "k4_seed", "iou", "critical_iou"),
            ("K0 CNN", "k0_seed", "iou", "critical_iou"),
            ("K4 no boundary weighting", "k4_no_boundary", "iou", "critical_iou"),
            ("K4 physics", "k4_seed0", "truncated4_iou", "truncated4_critical_iou"),
            ("K1 physics", "k4_seed0", "truncated1_iou", "truncated1_critical_iou"),
        ):
            selected = [
                row
                for row in fidelity
                if row["profile"] == profile and row["run"].startswith(prefix)
            ]
            full = statistics([row[full_key] for row in selected])
            central = statistics([row[central_key] for row in selected])
            fidelity_stats.append(
                {
                    "predictor": label,
                    "profile": profile,
                    "n_runs": full["n_runs"],
                    "full_iou_mean": full["mean"],
                    "full_iou_sample_std": full["sample_std"],
                    "central_iou_mean": central["mean"],
                    "central_iou_sample_std": central["sample_std"],
                }
            )
    for seed in SEEDS:
        result = read_json(ordinary / f"classifier_seed{seed}/results.json")
        for alpha, entry in result["primary_by_kind_calibrated"]["classifier"].items():
            nominal.append(
                policy_record(
                    "focus",
                    seed,
                    "classifier",
                    float(alpha),
                    "primary_by_kind_calibrated",
                    entry["test"],
                )
            )
        for split in ("cal", "test"):
            values = result["split_summary"][split]["classification"]
            classifiers.append(
                {
                    "seed": seed,
                    "split": split,
                    "kind": "all",
                    **{
                        key: values[key]
                        for key in ("n", "n_defects", "bce", "accuracy", "auroc")
                    },
                }
            )
            for kind in KINDS:
                classifiers.append(
                    {
                        "seed": seed,
                        "split": split,
                        "kind": kind,
                        **{
                            key: values["by_kind"][kind][key]
                            for key in ("n", "n_defects", "bce", "accuracy", "auroc")
                        },
                    }
                )
    aggregate, window, paired = {}, [], {}
    for distribution in ("ordinary", "critical"):
        table, stats, bootstrap = collect_window(
            args.work / "results" / distribution, distribution, args.resamples
        )
        window.extend(table)
        aggregate[distribution] = stats
        paired[distribution] = bootstrap
        plot_policies(stats, output / f"{distribution}_window")
    public = read_json(args.work / "results/public/summary.json")
    for scope, path in (
        ("nominal", ordinary / "nominal_timing.json"),
        ("nominal_classifier", ordinary / "classifier_timing.json"),
        ("ordinary_window", ordinary / "window_timing.json"),
        ("critical_window", args.work / "results/critical/window_timing.json"),
        ("public_window", args.work / "results/public/summary.json"),
    ):
        for method, values in read_json(path)["methods"].items():
            timings.append(
                {
                    "scope": scope,
                    "method": method,
                    "median_seconds": values["median_seconds"],
                    "speedup": values.get(
                        "speedup_vs_optimized_window_reference",
                        values.get("speedup_vs_optimized_full_reference"),
                    ),
                    "all_seconds": json.dumps(values["all_seconds"]),
                    "counts_and_work": json.dumps(values),
                }
            )
    for filename, rows in (
        ("fidelity.csv", fidelity),
        ("fidelity_aggregate.csv", fidelity_stats),
        ("nominal_and_shift.csv", nominal),
        ("window_policies.csv", window),
        ("training_resources.csv", resources),
        ("classifier_metrics.csv", classifiers),
        ("timing.csv", timings),
    ):
        write_csv(output / filename, rows)
    result = {
        "window_metrics": aggregate,
        "paired_source_bootstrap": paired,
        "public": public,
        "fidelity": fidelity,
        "fidelity_aggregate": fidelity_stats,
        "resources": resources,
        "classifier_metrics": classifiers,
        "timings": timings,
        "total_training_seconds": sum(row["training_seconds"] for row in resources),
    }
    (output / "summary.json").write_text(
        json.dumps(json_finite(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>RCPWV results</title>',
        "<style>body{font-family:Arial;max-width:1200px;margin:40px auto;padding:20px;color:#25333d}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccd7dc;text-align:left}img{width:100%}</style>",
        "<h1>RCPWV results</h1><p>Rates are conditional on their stated denominators. Optimizer seeds share the same test sources. Paired intervals keep fitted models and calibration thresholds fixed. Public rates are descriptive with source-group summaries.</p>",
    ]
    for distribution, rows in aggregate.items():
        body.append(
            f'<h2>{distribution.capitalize()} window policies</h2><img src="{distribution}_window.png"><table><tr><th>Method</th><th>Budget</th><th>Healthy acceptance</th><th>Defect false acceptance</th><th>Total acceptance</th></tr>'
        )
        for row in rows:

            def format_metric(key):
                item = row["metrics"][key]
                if item["mean"] is None:
                    return "No denominator"
                return f"{item['mean']:.4f}" + (
                    f" ± {item['sample_std']:.4f}"
                    if item["sample_std"] is not None
                    else ""
                )

            body.append(
                f"<tr><td>{html.escape(row['method'])}</td><td>{row['alpha'] if row['alpha'] is not None else 'Uncalibrated'}</td><td>{format_metric('healthy_acceptance_rate')}</td><td>{format_metric('defect_miss_rate')}</td><td>{format_metric('pass_rate')}</td></tr>"
            )
        body.append("</table>")
    body.append(
        "<h2>Complete runtime</h2><table><tr><th>Scope</th><th>Method</th><th>Median seconds</th><th>Speedup</th></tr>"
    )
    for row in timings:
        body.append(
            f"<tr><td>{row['scope']}</td><td>{row['method']}</td><td>{row['median_seconds']:.4f}</td><td>{row['speedup']:.3f}</td></tr>"
        )
    body.append(
        "</table><p>Full reconstruction metrics, classifier AUROC/BCE, per-kind counts and confidence intervals, source groups, timing repeats and work counters are available in the adjacent CSV files and summary.json.</p></html>"
    )
    (output / "index.html").write_text("".join(body), encoding="utf-8")
    print(json.dumps({"report": str(output / "index.html")}), flush=True)


if __name__ == "__main__":
    main()
