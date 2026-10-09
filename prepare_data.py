import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from patterns import select_specs
from geometry import evaluate_spec
from patterns import make_pattern
from simulator import Simulator
from spectral_simulator import SpectralSimulator, embed_band, synthesize_intensity


ROOT = Path(__file__).resolve().parent
SIZE = 2048
CONDITIONS = ("focus", "defocus")
DEVELOPMENT_FAMILIES = ("AND", "OR", "BUF", "INV", "NAND", "NOR")
HOLDOUT_FAMILIES = ("AOI", "OAI", "MUX", "XOR", "XNOR", "DFF", "SDFF", "CLKGATE")


def draw_rectangle(mask, rng, bounds, widths=(24, 121), lengths=(70, 421)):

    lower, upper = bounds
    x, y = rng.integers(lower, upper, size=2)
    width, length = int(rng.integers(*widths)), int(rng.integers(*lengths))
    if rng.integers(2):
        width, length = length, width
    x1, y1 = min(int(x + width), upper), min(int(y + length), upper)
    mask[int(y) : y1, int(x) : x1] = 1
    return [int(x), int(y), x1, y1]


def make_full_pattern(rng, kind, source_seed, critical=None):

    center, local_spec, parameters, _ = make_pattern(rng, kind, critical=critical)
    dense_center = bool(rng.random() < 0.65)
    accepted_rectangles = []
    if dense_center:
        for _ in range(int(rng.integers(6, 16))):
            candidate = center.copy()
            rectangle = draw_rectangle(candidate, rng, (16, 496), (20, 101), (50, 191))

            if not evaluate_spec(candidate, local_spec)["defect"]:
                center = candidate
                accepted_rectangles.append(rectangle)

    full = np.zeros((SIZE, SIZE), dtype=np.uint8)
    density_level = int(rng.integers(3))
    background_rectangles = []
    for _ in range(
        int(rng.integers(55 + 45 * density_level, 105 + 45 * density_level))
    ):
        background_rectangles.append(draw_rectangle(full, rng, (0, SIZE)))

    for _ in range(int(rng.integers(4, 11))):
        position = int(rng.integers(0, SIZE - 90))
        width = int(rng.integers(35, 91))
        if rng.integers(2):
            full[position : position + width, :] = 1
        else:
            full[:, position : position + width] = 1
    full[704:1344, 704:1344] = 0
    full[768:1280, 768:1280] = center
    spec = {
        **local_spec,
        "terminals": [
            [int(value + 768) for value in terminal]
            for terminal in local_spec["terminals"]
        ],
    }
    if evaluate_spec(full, spec)["defect"]:
        raise RuntimeError("Generated full input violates its specification")
    metadata = {
        "source_group": f"procedural_seed_{source_seed}",
        "source_seed": int(source_seed),
        "source": "new_procedural_full_context",
        "central_parameters": parameters,
        "dense_center": dense_center,
        "accepted_center_rectangles": accepted_rectangles,
        "background_density_level": density_level,
        "background_rectangle_count": len(background_rectangles),
        "spec": spec,
    }
    return full, metadata


def logical_family(cell_name):

    stem = cell_name.split("_X")[0]
    if stem.startswith("SDFF"):
        return "SDFF"
    if stem.startswith("DFF"):
        return "DFF"
    return re.sub(r"\d+", "", stem)


def read_gds(path):

    import gdstk

    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        return gdstk.read_gds(path.name)
    finally:
        os.chdir(previous)


def rasterize_polygons(polygons, unit_m, center_nm=None):

    polygons_nm = [np.asarray(polygon.points) * (unit_m * 1e9) for polygon in polygons]
    if not polygons_nm:
        raise ValueError("Requested cell contains no M1 polygons")
    points = np.concatenate(polygons_nm)
    if center_nm is None:
        center_nm = (points.min(axis=0) + points.max(axis=0)) / 2
    origin = np.asarray(center_nm) - SIZE / 2
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for polygon in polygons_nm:
        coordinates = np.rint(polygon - origin).astype(np.int32)
        cv2.fillPoly(mask, [coordinates], color=1)
    return mask, origin.tolist()


def public_sources():

    library = read_gds(ROOT / "vendor/nangate45/NangateOpenCellLibrary.gds")
    sources = {"public_development": [], "public": []}
    for split, families in (
        ("public_development", DEVELOPMENT_FAMILIES),
        ("public", HOLDOUT_FAMILIES),
    ):
        for family in families:
            candidates = sorted(
                (cell for cell in library.cells if logical_family(cell.name) == family),
                key=lambda cell: cell.name,
            )

            for cell in candidates[:2]:
                polygons = cell.get_polygons(layer=11, datatype=0)
                mask, origin = rasterize_polygons(polygons, library.unit)
                specs = select_specs(mask)
                if not specs:
                    raise ValueError(
                        f"Cannot define an input specification for {cell.name}"
                    )
                preferred_kind = "bridge" if len(sources[split]) % 2 == 0 else "break"
                primary = next(
                    (spec for spec in specs if spec["kind"] == preferred_kind), specs[0]
                )
                sources[split].append(
                    (
                        mask,
                        {
                            "source": "OpenROAD_Nangate45_M1",
                            "cell": cell.name,
                            "source_group": f"nangate45_family_{family}",
                            "origin_nm": origin,
                            "spec": primary,
                            "all_specs": specs,
                            "layer": [11, 0],
                            "context": "native-scale 2048nm M1 window; reference periodic boundary",
                        },
                    )
                )
    gcd = read_gds(ROOT / "vendor/openilt/gcd_45nm.gds")
    polygons = gcd.top_level()[0].get_polygons(layer=11, datatype=0)
    points = np.concatenate([p.points for p in polygons]) * (gcd.unit * 1e9)
    low, high = points.min(axis=0), points.max(axis=0)
    for index, fraction in enumerate(((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))):
        center = low + (high - low) * np.asarray(fraction)
        mask, origin = rasterize_polygons(polygons, gcd.unit, center_nm=center)
        specs = select_specs(mask)
        if not specs:
            raise ValueError("GCD window lacks a definable input specification")
        preferred_kind = "bridge" if index % 2 == 0 else "break"
        primary = next(
            (spec for spec in specs if spec["kind"] == preferred_kind), specs[0]
        )
        sources["public"].append(
            (
                mask,
                {
                    "source": "OpenILT_gcd45nm_M1",
                    "cell": "TOP",
                    "source_group": "gcd45nm_TOP",
                    "window": index,
                    "origin_nm": origin,
                    "spec": primary,
                    "all_specs": specs,
                    "layer": [11, 0],
                    "context": "native-scale 2048nm M1 window; reference periodic boundary",
                },
            )
        )
    return sources


@torch.no_grad()
def simulate_features(simulator, mask):

    band = simulator.mask_band(mask)
    lowpass_mask = torch.fft.ifft2(embed_band(band, 128), norm="forward").real
    features, aerial = [], []
    for name in CONDITIONS:
        kernels = getattr(simulator, f"{name}_kernel")
        scales = getattr(simulator, f"{name}_scales")
        fields = torch.fft.ifft2(
            embed_band(band[:, None] * kernels[None], 128), norm="forward"
        )
        modes = fields.abs().square() * scales[None, :, None, None]
        features.append(torch.cat((modes[:, :4], lowpass_mask[:, None]), dim=1))
        aerial.append(modes.sum(dim=1))
    return torch.stack(features, dim=1), torch.stack(aerial, dim=1)


def save_split(output, split, sources, simulator, original, device, count=None):

    count = len(sources) if count is None else count
    features = np.empty((count, 2, 5, 128, 128), dtype=np.float32)
    aerial_grid = np.empty((count, 2, 128, 128), dtype=np.float32)
    packed = np.empty((count, SIZE, SIZE // 8), dtype=np.uint8)
    records, reference_checks = [], []
    started = time.perf_counter()
    for index, (mask, metadata) in enumerate(sources):
        tensor = torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            feature_tensor, grid = simulate_features(simulator, tensor)
            printed_intensity = synthesize_intensity(grid[0], SIZE)
        features[index] = feature_tensor[0].cpu().numpy()
        aerial_grid[index] = grid[0].cpu().numpy()
        packed[index] = np.packbits(mask, axis=-1, bitorder="big")
        record = {
            "id": f"{split}_{index:05d}",
            "group": metadata["source_group"],
            **metadata,
        }
        for condition, name in enumerate(CONDITIONS):
            truth = (printed_intensity[condition] > 0.225).cpu().numpy()
            record[f"truth_{name}"] = evaluate_spec(truth, metadata["spec"])
            if "all_specs" in metadata:
                record[f"all_truth_{name}"] = [
                    evaluate_spec(truth, spec) for spec in metadata["all_specs"]
                ]
        records.append(record)
        if index == 0:
            for condition, name in enumerate(CONDITIONS):
                reference = original.aerial(tensor, defocus=name == "defocus")[0]
                error = float((reference - printed_intensity[condition]).abs().max())
                disagreement = int(
                    (
                        (reference > 0.225) != (printed_intensity[condition] > 0.225)
                    ).sum()
                )
                if error > 2e-5:
                    raise ArithmeticError(f"Spectral reference mismatch: {error}")
                reference_checks.append(
                    {
                        "id": record["id"],
                        "condition": name,
                        "max_abs_aerial_error": error,
                        "binary_disagreement_pixels": disagreement,
                    }
                )
        if (index + 1) % 64 == 0 or index + 1 == count:
            print(
                json.dumps(
                    {
                        "split": split,
                        "completed": index + 1,
                        "total": count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    np.savez_compressed(
        output / f"{split}.npz",
        features=features,
        aerial_grid=aerial_grid,
        masks_packed=packed,
    )
    (output / f"{split}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    summary = {
        "n_layouts": count,
        "n_source_groups": len({record["source_group"] for record in records}),
        "n_conditions_per_layout": 2,
        "condition_order": list(CONDITIONS),
        "reference_checks": reference_checks,
        "seconds": time.perf_counter() - started,
        "by_condition": {
            name: {
                "defects": sum(record[f"truth_{name}"]["defect"] for record in records),
                "by_kind": {
                    kind: {
                        "n": sum(record["spec"]["kind"] == kind for record in records),
                        "defects": sum(
                            record["spec"]["kind"] == kind
                            and record[f"truth_{name}"]["defect"]
                            for record in records
                        ),
                    }
                    for kind in ("bridge", "break")
                },
            }
            for name in CONDITIONS
        },
    }
    print(
        json.dumps({"event": "split_complete", "split": split, **summary}), flush=True
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare full-context SOCS data for spectral residual learning."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--train", type=int, default=1536)
    parser.add_argument("--val", type=int, default=192)
    parser.add_argument("--cal", type=int, default=512)
    parser.add_argument("--test", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--public-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    counts = (
        {}
        if args.public_only
        else {"train": args.train, "val": args.val, "cal": args.cal, "test": args.test}
    )
    if any(count < 2 or count % 2 for count in counts.values()):
        parser.error(
            "All split sizes must be positive even numbers for equal kind quotas"
        )
    split_names = list(counts)
    if args.public or args.public_only:
        split_names += ["public_development", "public"]
    args.output.mkdir(parents=True, exist_ok=True)
    if any(
        (args.output / f"{split}.npz").exists()
        or (args.output / f"{split}.json").exists()
        for split in split_names
    ):
        raise FileExistsError(
            "A requested split already exists; do not overwrite established data"
        )
    simulator = SpectralSimulator(device=args.device)
    original = Simulator(device=args.device)
    summary = {
        "seed": args.seed,
        "input_size": SIZE,
        "pixel_size_nm": 1.0,
        "simulation": "public 24-mode SOCS, exact band-limited implementation; periodic 2048nm domain",
        "condition_order": list(CONDITIONS),
        "dose_by_condition": [1.0, 1.0],
        "resist_threshold": 0.225,
        "no_truth_based_filtering": True,
        "feature_channels": [
            "weighted_mode_0",
            "weighted_mode_1",
            "weighted_mode_2",
            "weighted_mode_3",
            "mask_bandlimited_real",
        ],
        "aerial_grid_reconstruction": "spectral_simulator.synthesize_intensity(grid,2048)",
        "packed_mask_reconstruction": "np.unpackbits(masks_packed,axis=-1,count=2048,bitorder='big')",
        "independence_unit": "procedural source layout, not optical condition or pixel",
        "public_independence_unit": "logical family, or entire GCD source circuit",
        "split_source_groups_disjoint": True,
        "splits": {},
    }
    used_seeds = set()
    sequence = np.random.SeedSequence(args.seed)
    for split, count in counts.items():
        children = sequence.spawn(count)
        source_seeds = []
        for index, child in enumerate(children):
            source_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            if source_seed in used_seeds:
                raise RuntimeError("Duplicate source seed across splits")
            used_seeds.add(source_seed)
            source_seeds.append(source_seed)

        sources = (
            make_full_pattern(
                np.random.default_rng(source_seed),
                "bridge" if index % 2 == 0 else "break",
                source_seed,
            )
            for index, source_seed in enumerate(source_seeds)
        )
        summary["splits"][split] = save_split(
            args.output, split, sources, simulator, original, args.device, count=count
        )
    if args.public or args.public_only:
        for split, sources in public_sources().items():
            summary["splits"][split] = save_split(
                args.output, split, sources, simulator, original, args.device
            )
    name = "public_summary.json" if args.public_only else "dataset_summary.json"
    (args.output / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
