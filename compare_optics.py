import argparse
import json
from pathlib import Path

import numpy as np
import torch

from public_data import load_public
from simulator import Simulator
from spectral_simulator import SpectralSimulator


def main():
    parser = argparse.ArgumentParser(
        description="Compare full-grid and compact-grid SOCS numerical precision on public masks."
    )
    parser.add_argument(
        "--public-data",
        type=Path,
        default=Path(__file__).resolve().parent / "work/data/ordinary/public",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--doses", type=float, nargs="+", default=[0.98, 1.0, 1.02])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or any(
        not np.isfinite(dose) or dose <= 0 for dose in args.doses
    ):
        parser.error("samples and finite amplitude multipliers must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    records, masks, _ = load_public(args.public_data)
    if args.samples > len(records):
        parser.error("samples exceeds the public source count")
    original = Simulator(args.device)
    compact = SpectralSimulator(args.device)
    rows = []
    with torch.inference_mode():
        for record, mask in zip(records[: args.samples], masks[: args.samples]):
            tensor = torch.from_numpy(mask).to(args.device, dtype=torch.float32)
            for condition in (0, 1):
                for dose in args.doses:
                    reference = original.aerial(
                        tensor, dose=dose, defocus=bool(condition)
                    )
                    prediction = compact.aerial(
                        tensor, dose=dose, defocus=bool(condition)
                    )
                    rows.append(
                        {
                            "id": record["id"],
                            "condition": condition,
                            "dose": dose,
                            "max_abs_intensity_difference": float(
                                (reference - prediction).abs().max()
                            ),
                            "thresholded_pixel_disagreements": int(
                                ((reference > 0.225) != (prediction > 0.225)).sum()
                            ),
                        }
                    )
    result = {
        "n_combinations": len(rows),
        "max_abs_intensity_difference": max(
            row["max_abs_intensity_difference"] for row in rows
        ),
        "total_thresholded_pixel_disagreements": sum(
            row["thresholded_pixel_disagreements"] for row in rows
        ),
        "combinations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
