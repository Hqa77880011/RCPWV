import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geometry import evaluate_spec
from prepare_data import (
    CONDITIONS,
    SIZE,
    make_full_pattern,
    save_split,
    simulate_features,
)
from simulator import Simulator
from spectral_simulator import SpectralSimulator, synthesize_intensity


BOUNDS = {"bridge": (8, 160), "break": (8, 140)}
DOSES = (("low", 0.98), ("high", 1.02))


@torch.inference_mode()
def window_truth_from_grid(grid, spec):

    endpoints = {}
    for index, condition in enumerate(CONDITIONS):
        for name, dose in DOSES:
            intensity = synthesize_intensity(grid[index : index + 1] * dose**2, SIZE)
            mask = (intensity[0] > 0.225).cpu().numpy()
            endpoints[f"{condition}_{name}"] = evaluate_spec(mask, spec)
    return {
        "defect": any(truth["defect"] for truth in endpoints.values()),
        "endpoints": endpoints,
    }


def find_label_transition(probe, lower, upper):

    first_lower, first_upper = lower, upper
    lower_truth, upper_truth = probe(lower), probe(upper)
    initial = {str(lower): lower_truth, str(upper): upper_truth}
    bracket_found = lower_truth["defect"] != upper_truth["defect"]
    steps = []
    if bracket_found:
        while upper - lower > 1:
            middle = (lower + upper) // 2
            truth = probe(middle)
            steps.append({"critical": middle, "window_defect": truth["defect"]})
            if truth["defect"] == lower_truth["defect"]:
                lower, lower_truth = middle, truth
            else:
                upper, upper_truth = middle, truth
            if lower_truth["defect"] == upper_truth["defect"]:
                raise AssertionError(
                    "Bisection lost its unequal-label endpoint bracket"
                )
    return {
        "initial_bounds": [first_lower, first_upper],
        "initial_endpoint_truth": initial,
        "bracket_found": bool(bracket_found),
        "search_steps": steps,
        "retained_bracket": [lower, upper],
        "retained_endpoint_truth": {str(lower): lower_truth, str(upper): upper_truth},
        "base_critical": (lower + upper) // 2,
    }


def make_critical_source(
    source_seed, jitter_seed, parent_seed, kind, simulator, device
):

    lower, upper = BOUNDS[kind]

    def rebuild(critical):

        return make_full_pattern(
            np.random.default_rng(source_seed), kind, source_seed, critical=critical
        )

    @torch.inference_mode()
    def probe(critical):
        mask, metadata = rebuild(critical)
        tensor = torch.as_tensor(mask[None], device=device, dtype=torch.float32)
        _, grids = simulate_features(simulator, tensor)
        return window_truth_from_grid(grids[0], metadata["spec"])

    search = find_label_transition(probe, lower, upper)
    jitter = int(np.random.default_rng(jitter_seed).integers(-6, 7))
    final_critical = int(np.clip(search["base_critical"] + jitter, lower, upper))
    mask, metadata = rebuild(final_critical)
    metadata.update(
        {
            "source_group": f"critical_seed_{source_seed}",
            "source": "simulator_mined_critical_full_context",
            "parent_seed": int(parent_seed),
            "jitter_seed": int(jitter_seed),
            "critical_parameter": "gap" if kind == "bridge" else "width",
            "critical_search": search,
            "critical_jitter": jitter,
            "final_critical": final_critical,
        }
    )
    return mask, metadata


def append_final_window_labels(output, split, device):

    record_path = output / f"{split}.json"
    records = json.loads(record_path.read_text(encoding="utf-8"))
    with np.load(output / f"{split}.npz", allow_pickle=False) as archive:
        grids = archive["aerial_grid"]
    for record, grid in zip(records, grids):
        truth = window_truth_from_grid(
            torch.as_tensor(grid, device=device), record["spec"]
        )
        record["truth_window"] = {"defect": truth["defect"]}
        for profile, endpoint in truth["endpoints"].items():
            record[f"truth_{profile}"] = endpoint
    record_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {
        "n_bracket_found": sum(
            record["critical_search"]["bracket_found"] for record in records
        ),
        "n_no_bracket_retained": sum(
            not record["critical_search"]["bracket_found"] for record in records
        ),
        "window_defects": sum(record["truth_window"]["defect"] for record in records),
        "by_kind_window": {
            kind: {
                "n": sum(record["spec"]["kind"] == kind for record in records),
                "defects": sum(
                    record["spec"]["kind"] == kind and record["truth_window"]["defect"]
                    for record in records
                ),
            }
            for kind in BOUNDS
        },
        "endpoint_defects": {
            profile: sum(record[f"truth_{profile}"]["defect"] for record in records)
            for profile in ("focus_low", "focus_high", "defocus_low", "defocus_high")
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create independent simulator-mined critical-window calibration/test sources."
    )
    parser.add_argument("--output", type=Path, default=Path("data/critical"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--cal", type=int, default=256)
    parser.add_argument("--test", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    counts = {"cal": args.cal, "test": args.test}
    if any(count < 2 or count % 2 for count in counts.values()):
        parser.error(
            "cal/test counts must be positive even numbers for equal kind quotas"
        )
    if any(
        (args.output / f"{split}.{suffix}").exists()
        for split in counts
        for suffix in ("npz", "json")
    ):
        raise FileExistsError(
            "Critical split already exists; do not overwrite established data"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    simulator = SpectralSimulator(device=args.device)
    original = Simulator(device=args.device)
    sequence = np.random.SeedSequence(args.seed)
    used_source_seeds = set()
    started = time.perf_counter()
    summary = {
        "seed": args.seed,
        "input_size": SIZE,
        "pixel_size_nm": 1.0,
        "distribution": "simulator-mined critical-window stress distribution; not natural industrial prevalence",
        "simulation": "public full 24-mode SOCS at2048 pixels; periodic domain",
        "condition_order": list(CONDITIONS),
        "dose_amplitudes": [0.98, 1.02],
        "intensity_multipliers": [0.98**2, 1.02**2],
        "resist_threshold": 0.225,
        "critical_bounds": BOUNDS,
        "jitter_uniform_integer_bounds_inclusive": [-6, 6],
        "truth_guided_parameter_search": True,
        "no_final_label_selection_or_filtering": True,
        "same_label_endpoint_source_retained": True,
        "critical_search_rule": "unequal-label endpoint bisection, no global monotonicity assumption; floor midpoint plus independent jitter",
        "independence_unit": "source layout; probe parameters and process endpoints are correlated within source",
        "split_source_groups_disjoint": True,
        "train_models": False,
        "splits": {},
    }
    for split, count in counts.items():
        source_seeds, jitter_seeds = [], []
        for child in sequence.spawn(count):
            source_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            jitter_seed = int(child.spawn(1)[0].generate_state(1, dtype=np.uint64)[0])
            if source_seed in used_source_seeds:
                raise RuntimeError("Duplicate critical source seed across splits")
            used_source_seeds.add(source_seed)
            source_seeds.append(source_seed)
            jitter_seeds.append(jitter_seed)
        sources = (
            make_critical_source(
                source_seed,
                jitter_seed,
                args.seed,
                "bridge" if index % 2 == 0 else "break",
                simulator,
                args.device,
            )
            for index, (source_seed, jitter_seed) in enumerate(
                zip(source_seeds, jitter_seeds)
            )
        )
        summary["splits"][split] = save_split(
            args.output, split, sources, simulator, original, args.device, count=count
        )
        summary["splits"][split].update(
            append_final_window_labels(args.output, split, args.device)
        )
    summary["n_unique_source_seeds"] = len(used_source_seeds)
    summary["total_seconds"] = time.perf_counter() - started
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "critical_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
