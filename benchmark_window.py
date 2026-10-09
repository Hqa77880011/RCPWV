import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_classifier import online_area_occupancy
from classifier import build_classifier_input, load_classifier
from evaluate import project_spec
from geometry import evaluate_spec, margin_pass, signed_distance
from model import load_model, mask_lowpass_grid
from spectral_simulator import SpectralSimulator, embed_band, synthesize_intensity
from calibrate_window import ENDPOINTS


METHODS = (
    "all_spectral24",
    "aerial_margin",
    "nominal_aerial_only",
    "truncated4_margin",
    "spectral24_margin",
    "input_uniform_margin",
    "classifier",
)
DOSES = (0.98, 1.02)
ENDPOINT_CONFIG = ((0, 0.98), (0, 1.02), (1, 0.98), (1, 1.02))
COUNTERS = (
    "h2d_transfers",
    "input_ffts",
    "mask_lowpass_iffts",
    "coherent_ifft_batches",
    "coherent_modes_focus",
    "coherent_modes_defocus",
    "cnn_calls",
    "coarse_reconstructions",
    "full_reconstructions",
    "coarse_margin_checks",
    "terminal_presence_checks",
    "full_connectivity_checks",
    "reference_calls",
    "skipped_highdose_break_checks",
    "classifier_preprocessing_calls",
    "classifier_h2d_transfers",
    "classifier_calls",
)


class WindowQuery:
    def __init__(self, mask, spec, model, simulator, device, score_size):
        self.mask = mask
        self.spec = spec
        self.projected = project_spec(spec, score_size)
        self.model, self.simulator = model, simulator
        self.device, self.score_size = device, score_size
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.band = None
        self.mask_low = None
        self.modes = {}
        self.grids = {}
        self.full = {}

    def get_band(self):
        if self.band is None:
            tensor = torch.from_numpy(self.mask[None]).to(
                self.device, dtype=torch.float32
            )
            self.counts["h2d_transfers"] += 1
            self.band = self.simulator.mask_band(tensor)
            self.counts["input_ffts"] += 1
        return self.band

    def get_modes(self, condition, count):
        previous = self.modes.get(condition)
        start = 0 if previous is None else previous.shape[1]
        if start < count:
            name = "defocus" if condition else "focus"
            kernel = getattr(self.simulator, f"{name}_kernel")[start:count]
            scales = getattr(self.simulator, f"{name}_scales")[start:count]
            field = torch.fft.ifft2(
                embed_band(self.get_band()[:, None] * kernel[None], 128), norm="forward"
            )
            additional = field.abs().square() * scales[None, :, None, None]
            self.modes[condition] = (
                additional
                if previous is None
                else torch.cat((previous, additional), dim=1)
            )
            self.counts["coherent_ifft_batches"] += 1
            self.counts[f"coherent_modes_{name}"] += count - start

        return self.modes[condition][:, :count]

    def get_grid(self, condition, method):
        key = (condition, method)
        if key not in self.grids:
            count = 24 if method == "spectral24_margin" else 4
            modes = self.get_modes(condition, count)
            if method == "aerial_margin":
                if self.mask_low is None:
                    self.mask_low = mask_lowpass_grid(self.get_band())[:, None]
                    self.counts["mask_lowpass_iffts"] += 1
                features = torch.cat((modes, self.mask_low), dim=1)
                with torch.autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ):
                    grid = self.model.forward_features(features, condition)
                self.grids[key] = grid.float()
                self.counts["cnn_calls"] += 1
            else:
                self.grids[key] = modes.sum(dim=1)
        return self.grids[key]

    def reference_window_defect(self):

        self.counts["reference_calls"] += 1
        for condition in (0, 1):
            if condition not in self.full:
                grid = self.get_grid(condition, "spectral24_margin")
                self.full[condition] = synthesize_intensity(grid)
                self.counts["full_reconstructions"] += 1

            if self.spec["kind"] == "bridge":
                low_threshold = self.simulator.threshold / DOSES[0] ** 2
                presence = (
                    torch.stack(
                        [
                            torch.any(
                                self.full[condition][0, y0:y1, x0:x1] > low_threshold
                            )
                            for y0, x0, y1, x1 in self.spec["terminals"]
                        ]
                    )
                    .cpu()
                    .numpy()
                )
                self.counts["terminal_presence_checks"] += 1
                if not presence.all():
                    return True
                dose = DOSES[1]
            else:
                dose = DOSES[0]
            actual = (
                (self.full[condition][0] > self.simulator.threshold / dose**2)
                .cpu()
                .numpy()
            )
            self.counts["full_connectivity_checks"] += 1
            if evaluate_spec(actual, self.spec)["defect"]:
                return True
            if self.spec["kind"] == "break":
                self.counts["skipped_highdose_break_checks"] += 1
        return False

    def route(self, method, threshold):

        if threshold == float("inf"):
            return False
        if method == "input_uniform_margin":
            stride = 2048 // self.score_size
            coarse = self.mask[::stride, ::stride]
            self.counts["coarse_margin_checks"] += 1
            if not evaluate_spec(coarse, self.projected)["compliant"]:
                return -1.0 > threshold
            field = (
                signed_distance(np.pad(coarse, 1))[1:-1, 1:-1]
                if coarse.all()
                else signed_distance(coarse)
            )

            return margin_pass(field, 1.0, self.projected, threshold)
        endpoints = ((0, 1.0),) if method == "nominal_aerial_only" else ENDPOINT_CONFIG
        grid_method = "aerial_margin" if method == "nominal_aerial_only" else method
        for condition, dose in endpoints:
            grid = self.get_grid(condition, grid_method)

            coarse = (
                synthesize_intensity(grid * dose**2, self.score_size)[0].cpu().numpy()
            )
            self.counts["coarse_reconstructions"] += 1
            self.counts["coarse_margin_checks"] += 1
            if not margin_pass(coarse - 0.225, 1.0, self.projected, threshold):
                return False
        return True

    def execute(self, method, threshold=None):

        if method != "all_spectral24" and self.route(method, threshold):
            return True, None, self.counts
        return False, self.reference_window_defect(), self.counts


def online_classifier_score(mask, spec, classifier, device, counts):

    area = online_area_occupancy(mask)
    counts["classifier_preprocessing_calls"] += 1
    area_tensor = torch.from_numpy(area[None, None]).to(device)
    counts["classifier_h2d_transfers"] += 1
    classifier_spec = {**spec, "source_shape": [2048, 2048]}
    inputs = build_classifier_input(area_tensor, [classifier_spec])
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logit = classifier(inputs)
    counts["classifier_calls"] += 1
    return float(-torch.sigmoid(logit.float().cpu())[0])


def read_frozen_window(evaluation, policy_path, records, alpha):

    evaluation_result = json.loads(
        (evaluation / "results.json").read_text(encoding="utf-8")
    )
    window = json.loads(policy_path.read_text(encoding="utf-8"))
    if tuple(window["endpoints"]) != ENDPOINTS:
        raise ValueError("Window endpoints differ from the fixed benchmark definition")
    classifier_path = window["config"].get("classifier")
    if not classifier_path or "classifier" not in window["primary_by_kind_calibrated"]:
        raise ValueError(
            "Window benchmark requires the frozen classifier CSV and its window-kind policy"
        )
    classifier_rows = {}
    with Path(classifier_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row["split"] != "test"
                or (row.get("profile") or row.get("condition")) != "focus"
            ):
                continue
            if row["id"] in classifier_rows:
                raise ValueError(
                    f"Duplicate frozen nominal classifier score: {row['id']}"
                )
            classifier_rows[row["id"]] = row
    grouped = {}
    with (evaluation / "per_sample.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test" and row["profile"] in (*ENDPOINTS, "focus"):
                grouped.setdefault(row["id"], {})[row["profile"]] = row
    expected = []
    thresholds = {}
    for method in METHODS[1:]:
        policy = window["primary_by_kind_calibrated"][method][str(alpha)]
        thresholds[method] = {
            kind: float("inf") if value is None else float(value)
            for kind, value in policy["thresholds"].items()
        }
    for record in records:
        endpoints = grouped[record["id"]]
        if any(profile not in endpoints for profile in (*ENDPOINTS, "focus")):
            raise ValueError(f"Missing saved endpoint for {record['id']}")
        kind = record["spec"]["kind"]
        if any(row["kind"] != kind for row in endpoints.values()):
            raise ValueError("Saved kind differs from current fixed specification")
        scores = {
            method: min(float(endpoints[profile][method]) for profile in ENDPOINTS)
            for method in METHODS[1:]
            if method not in {"classifier", "nominal_aerial_only"}
        }
        scores["nominal_aerial_only"] = float(endpoints["focus"]["aerial_margin"])
        classifier_row = classifier_rows[record["id"]]
        if classifier_row["kind"] != kind:
            raise ValueError(
                "Frozen classifier kind differs from current fixed specification"
            )
        scores["classifier"] = float(classifier_row["classifier"])
        expected.append(
            {
                "id": record["id"],
                "window_defect": any(
                    endpoints[profile]["defect"].lower() == "true"
                    for profile in ENDPOINTS
                ),
                "passes": {
                    method: score > thresholds[method][kind]
                    for method, score in scores.items()
                },
                "scores": scores,
            }
        )
    return evaluation_result, window, thresholds, expected


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(
        description="Time frozen layout-level process-window routing with shared optical work."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--window-policy", type=Path, required=True)
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
            "Window timing already exists; retain the saved measurement"
        )
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device)
    if checkpoint["config"]["kernels"] != 4 or checkpoint["config"]["seed"] != 0:
        raise ValueError("This predeclared benchmark requires the K4 seed0 model")
    classifier, classifier_checkpoint = load_classifier(args.classifier, device)
    if classifier_checkpoint["config"]["input_size"] != 512:
        raise ValueError(
            "Window classifier timing requires the frozen 512-occupancy classifier"
        )
    simulator = SpectralSimulator(device=device)
    all_records = json.loads((args.data / "test.json").read_text(encoding="utf-8"))
    if len(all_records) < args.samples:
        raise ValueError("Requested more sources than available in the fixed test set")
    records = all_records[: args.samples]
    with np.load(args.data / "test.npz", allow_pickle=False) as archive:
        masks = np.unpackbits(archive["masks_packed"][: args.samples], axis=-1)
    result, window, thresholds, expected = read_frozen_window(
        args.evaluation, args.window_policy, records, args.alpha
    )
    if Path(window["config"]["evaluation"]).resolve() != args.evaluation.resolve():
        raise ValueError(
            "Window policy was constructed from a different endpoint evaluation"
        )
    if Path(result["config"]["model"]).resolve() != args.model.resolve():
        raise ValueError(
            "Benchmark checkpoint differs from frozen evaluation checkpoint"
        )
    classifier_result_path = Path(window["config"]["classifier"]).with_name(
        "results.json"
    )
    classifier_result = json.loads(classifier_result_path.read_text(encoding="utf-8"))
    classifier_config = classifier_result["config"]
    if Path(classifier_config["classifier"]).resolve() != args.classifier.resolve():
        raise ValueError(
            "Window classifier checkpoint differs from frozen nominal evaluation"
        )
    if (
        classifier_config["condition"] != "focus"
        or classifier_config["batch_size"] != 1
    ):
        raise ValueError(
            "Window classifier timing requires frozen nominal focus scores at batch_size=1"
        )
    score_size = result["config"]["score_size"]
    if masks.shape != (args.samples, 2048, 2048):
        raise ValueError("Expected preloaded full 2048 uint8 masks")
    times = {method: [] for method in METHODS}
    work = {method: [] for method in METHODS}
    outcomes = {}
    classifier_score_differences = []

    def operation(method, index):
        spec = records[index]["spec"]
        threshold = (
            None if method == "all_spectral24" else thresholds[method][spec["kind"]]
        )
        query = WindowQuery(masks[index], spec, model, simulator, device, score_size)
        if method == "classifier":
            score = online_classifier_score(
                masks[index], spec, classifier, device, query.counts
            )
            if score > threshold:
                return True, None, query.counts, score
            return False, query.reference_window_defect(), query.counts, score
        return (*query.execute(method, threshold), None)

    with torch.inference_mode():
        for method in METHODS:
            for _ in range(2):
                operation(method, 0)
        for repeat in range(args.repeats):
            offset = repeat % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                synchronize(device)
                started = time.perf_counter()
                values = [operation(method, index) for index in range(args.samples)]
                synchronize(device)
                elapsed = time.perf_counter() - started
                counts = {
                    key: sum(value[2][key] for value in values) for key in COUNTERS
                }
                for index, (passed, reference_defect, _, _) in enumerate(values):
                    if (
                        method != "all_spectral24"
                        and passed != expected[index]["passes"][method]
                    ):
                        raise AssertionError(
                            f"Online strict window route mismatch: {method}/{records[index]['id']}"
                        )
                    if (
                        reference_defect is not None
                        and reference_defect != expected[index]["window_defect"]
                    ):
                        raise AssertionError(
                            f"Reference union label mismatch: {method}/{records[index]['id']}"
                        )
                passed = [value[0] for value in values]
                summary = {
                    "n_samples": args.samples,
                    "n_window_defects": sum(item["window_defect"] for item in expected),
                    "n_passed": sum(passed),
                    "n_referred": args.samples - sum(passed),
                    "n_defects_missed": sum(
                        accept and item["window_defect"]
                        for accept, item in zip(passed, expected)
                    ),
                }
                if method in outcomes and outcomes[method] != summary:
                    raise AssertionError("Outcomes changed across timing repeats")
                if work[method] and work[method][0] != counts:
                    raise AssertionError(
                        "Actual work counts changed across timing repeats"
                    )
                if method == "classifier":
                    classifier_score_differences.append(
                        max(
                            abs(value[3] - frozen["scores"]["classifier"])
                            for value, frozen in zip(values, expected)
                        )
                    )
                outcomes[method] = summary
                times[method].append(elapsed)
                work[method].append(counts)
                print(
                    json.dumps(
                        {
                            "repeat": repeat + 1,
                            "method": method,
                            "seconds": elapsed,
                            **summary,
                            "work_counts": counts,
                        }
                    ),
                    flush=True,
                )
    baseline = float(np.median(times["all_spectral24"]))
    for method in METHODS:
        median = float(np.median(times[method]))
        outcomes[method].update(
            {
                "all_seconds": times[method],
                "median_seconds": median,
                "speedup_vs_optimized_window_reference": baseline / median,
                "actual_work_by_repeat": work[method],
            }
        )
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "endpoints": list(ENDPOINTS),
        "score_size": score_size,
        "methods": outcomes,
        "checks": {
            "strict_frozen_route_match": True,
            "reference_union_labels_match": True,
            "work_counts_stable_every_repeat": True,
        },
        "classifier_max_abs_score_difference_vs_frozen_per_repeat": classifier_score_differences,
        "scope": "Preloaded uint8 full masks to window decision; includes online classifier area/terminal preprocessing and classifier H2D/inference/CPU float32 sigmoid when applicable, optical-mask H2D, one shared optical input FFT, required optical modes, residual CNN, coarse CPU geometry, actual reference fallback and full-resolution connectivity. Excludes file loading, model loading and offline training/calibration.",
        "notes": [
            "Fixed first sources and original window-kind calibration thresholds; no policy retuning.",
            "At most one input FFT per query; each mode/condition is computed at most once.",
            "Dose endpoints reuse each supplied kernel's nominal intensity; no fourfold optical simulation.",
            "Bridge reference checks low-dose terminal presence and high-dose separation; break checks only low-dose connectivity by foreground monotonicity.",
            "Reference exits on first failed endpoint; identical reference semantics are used by every fallback.",
            "Each required condition is reconstructed once at full resolution; dose changes only its threshold.",
            "K4 fallback retains individual modes, appends remaining modes in original order and sums all 24 together.",
            "spectral24 routing fallback reuses its already evaluated full-mode grid.",
            "Routing evaluates the frozen four endpoint margin decisions with early rejection; input morphology is endpoint-invariant.",
            "nominal_aerial_only computes one focus/nominal-dose aerial margin and applies its frozen window-kind threshold; rejected queries use the same four-endpoint union reference.",
            "Finite precision route and union-label equality are verified on every query and repeat, not assumed bit-exact.",
            "reference_calls counts source queries requiring exact verification; coherent_modes_* counts actual optical mode evaluations.",
            "Classifier uses its nominal -sigmoid(logit) ranking score with the separately frozen window-kind calibrated threshold; rejection uses the identical optimized window reference.",
            "Classifier inputs and outputs are recomputed online at batch_size=1; no cached occupancy or logits are timed.",
            "classifier_h2d_transfers counts occupancy-to-classifier transfers; h2d_transfers counts original-mask-to-optics transfers. A rejected classifier query requires both, while an accepted query performs no optics.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "window_timing_complete",
                "output": str(args.output),
                "methods": outcomes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
