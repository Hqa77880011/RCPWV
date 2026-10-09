import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from benchmark_window import COUNTERS, WindowQuery, online_classifier_score
from classifier import load_classifier
from geometry import evaluate_spec
from model import load_model
from spectral_simulator import SpectralSimulator


def read_mask(path):
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read binary image: {path}")
    if mask.shape != (2048, 2048):
        raise ValueError("Input mask must have shape 2048x2048 at 1 nm/pixel")
    if np.all((mask == 0) | (mask == 255)):
        mask = mask / 255
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Input must be binary, using 0/1 or 0/255")
    return mask.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Verify one mask and terminal specification with a frozen window policy."
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Binary 2048x2048 .npy or image at 1 nm/pixel.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON with kind, terminals and connectivity.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="window_seed0.json from calibrate_window.py.",
    )
    parser.add_argument(
        "--method",
        choices=(
            "aerial_margin",
            "nominal_aerial_only",
            "truncated4_margin",
            "spectral24_margin",
            "input_uniform_margin",
            "classifier",
            "all_spectral24",
        ),
        default="nominal_aerial_only",
    )
    parser.add_argument("--alpha", type=float, choices=(0.01, 0.05, 0.1), default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    torch.set_num_threads(4)
    mask = read_mask(args.mask)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    evaluate_spec(mask, spec)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    evaluation = json.loads(
        (Path(policy["config"]["evaluation"]) / "results.json").read_text(
            encoding="utf-8"
        )
    )
    if Path(evaluation["config"]["model"]).resolve() != args.model.resolve():
        raise ValueError(
            "Checkpoint differs from the model used to calibrate the policy"
        )
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device)
    if checkpoint["config"]["kernels"] != 4:
        raise ValueError("Online routing requires the calibrated K4 model")
    simulator = SpectralSimulator(device)
    query = WindowQuery(
        mask, spec, model, simulator, device, evaluation["config"]["score_size"]
    )
    value = (
        None
        if args.method == "all_spectral24"
        else policy["primary_by_kind_calibrated"][args.method][str(args.alpha)][
            "thresholds"
        ][spec["kind"]]
    )
    threshold = float("inf") if value is None else float(value)
    with torch.inference_mode():
        if args.method == "classifier":
            if args.classifier is None:
                parser.error("--classifier is required for classifier routing")
            frozen_classifier = json.loads(
                Path(policy["config"]["classifier"])
                .with_name("results.json")
                .read_text(encoding="utf-8")
            )
            if (
                Path(frozen_classifier["config"]["classifier"]).resolve()
                != args.classifier.resolve()
            ):
                raise ValueError(
                    "Classifier differs from the checkpoint used for calibration"
                )
            classifier, _ = load_classifier(args.classifier, device)
            score = online_classifier_score(
                mask, spec, classifier, device, query.counts
            )
            passed = score > threshold
            defect = None if passed else query.reference_window_defect()
        else:
            passed, defect, _ = query.execute(args.method, threshold)
    result = {
        "mask": str(args.mask),
        "spec": spec,
        "method": args.method,
        "alpha": args.alpha,
        "decision": "accept" if passed or not defect else "reject",
        "automatically_accepted": bool(passed),
        "reference_window_defect": defect,
        "threshold": value,
        "work_counts": {key: query.counts[key] for key in COUNTERS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
