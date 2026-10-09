import argparse
import csv
import json
from pathlib import Path

from metrics import json_finite, summarize_policies
from evaluate import METHODS


ENDPOINTS = ("focus_low", "focus_high", "defocus_low", "defocus_high")


def as_bool(value):
    return value.lower() == "true"


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a single layout-level event covering a declared dose window."
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with (args.evaluation / "per_sample.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        raw = list(csv.DictReader(stream))
    grouped = {}
    for row in raw:
        grouped.setdefault((row["split"], row["id"]), {})[row["profile"]] = row
    classifiers = {}
    if args.classifier:
        with args.classifier.open(newline="", encoding="utf-8") as stream:
            classifiers = {
                (row["split"], row["id"]): float(row["classifier"])
                for row in csv.DictReader(stream)
                if (row.get("profile") or row.get("condition")) == "focus"
            }
    rows = []
    methods = [*METHODS, "nominal_aerial_only"] + (
        ["classifier"] if classifiers else []
    )
    for key, conditions in grouped.items():
        if any(profile not in conditions for profile in (*ENDPOINTS, "focus")):
            raise ValueError(f"Missing predeclared endpoint: {key}")
        reference = conditions["focus"]
        row = {
            "split": key[0],
            "id": key[1],
            "kind": reference["kind"],
            "defect": any(
                as_bool(conditions[profile]["defect"]) for profile in ENDPOINTS
            ),
        }
        for method in METHODS:
            row[method] = min(
                float(conditions[profile][method]) for profile in ENDPOINTS
            )
        row["nominal_aerial_only"] = float(reference["aerial_margin"])
        if classifiers:
            row["classifier"] = classifiers[key]
        rows.append(row)
    policies, _ = summarize_policies(
        [row for row in rows if row["split"] == "cal"],
        [row for row in rows if row["split"] == "test"],
        methods,
        [0.01, 0.05, 0.1],
    )
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seed": args.seed,
        "endpoints": list(ENDPOINTS),
        "amplitude_dose_interval": [0.98, 1.02],
        "primary_by_kind_calibrated": policies,
        "n_cal_sources": sum(row["split"] == "cal" for row in rows),
        "n_test_sources": sum(row["split"] == "test" for row in rows),
        "n_cal_window_defects": sum(
            row["split"] == "cal" and row["defect"] for row in rows
        ),
        "n_test_window_defects": sum(
            row["split"] == "test" and row["defect"] for row in rows
        ),
        "notes": [
            "Score=min(endpoint margins); one union-defect event per full source layout.",
            "Window-defect calibration is recomputed; nominal endpoint thresholds are not reused.",
            "Nominal-only and direct classifier are recalibrated against the same window-defect labels.",
            "Within each supplied focus/defocus kernel, dose-positive foreground is monotone.",
            "Low-dose terminal existence and high-dose separation characterize bridge compliance over the interval.",
            "Low-dose connectivity characterizes break compliance over the interval.",
            "This covers the stated dose interval for two supplied kernels, not a continuous defocus range.",
            "Class-conditional marginal guarantee is for P(pass|window defect,kind), not an individual fixed calibration realization.",
            f"This window result uses optimizer seed{args.seed}; seeds share source layouts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_finite(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_window_defects": result["n_test_window_defects"],
                "main": result["primary_by_kind_calibrated"]["aerial_margin"]["0.05"][
                    "test"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
