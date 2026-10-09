import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

from geometry import evaluate_spec
from model import load_model
from spectral_simulator import synthesize_intensity


SIZE = 2048
CROP_START, CROP_STOP = 768, 1280
THRESHOLD = 0.225
BLUE = "#426C88"
TEAL = "#438982"
GRAY = "#74838B"


def select_indices(records):

    selected, missing = [], []
    for kind in ("bridge", "break"):
        for defect in (False, True):
            index = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record["spec"]["kind"] == kind
                    and bool(record["truth_focus"]["defect"]) == defect
                ),
                None,
            )
            if index is None:
                missing.append({"kind": kind, "defect": defect})
            else:
                selected.append(index)
    if not selected:
        raise ValueError("No bridge/break records with truth_focus were found")
    return selected, missing


def crop_iou(prediction, reference):

    crop = np.s_[CROP_START:CROP_STOP, CROP_START:CROP_STOP]
    prediction, reference = prediction[crop], reference[crop]
    union = np.count_nonzero(prediction | reference)
    return float(np.count_nonzero(prediction & reference) / union) if union else 1.0


def load_examples(data, checkpoint_path):

    records = json.loads((data / "test.json").read_text(encoding="utf-8"))
    selected, missing = select_indices(records)
    model, checkpoint = load_model(checkpoint_path, "cpu")
    if checkpoint["config"]["kernels"] != 4:
        raise ValueError("This K4 comparison figure requires a kernels=4 checkpoint")
    model = model.float().eval()
    with np.load(data / "test.npz", allow_pickle=False) as archive:
        features = archive["features"][selected, 0].astype(np.float32)
        references = archive["aerial_grid"][selected, 0].astype(np.float32)
        packed = archive["masks_packed"][selected]
    examples = []
    with torch.inference_mode():
        for offset, index in enumerate(selected):
            feature = torch.from_numpy(features[offset : offset + 1])
            corrected_grid = model.forward_features(feature, 0).float()
            if corrected_grid.ndim == 4:
                corrected_grid = corrected_grid[:, 0]
            grids = torch.stack(
                (
                    torch.from_numpy(references[offset]),
                    feature[0, :4].sum(0),
                    corrected_grid[0],
                )
            )
            images = (synthesize_intensity(grids, SIZE) > THRESHOLD).numpy()
            reference, truncated, corrected = images
            mask = np.unpackbits(packed[offset], axis=-1, count=SIZE, bitorder="big")
            record = records[index]
            examples.append(
                {
                    "record": record,
                    "input": mask.astype(bool),
                    "reference": reference,
                    "truncated4": truncated,
                    "corrected": corrected,
                    "cpu_reference_defect": evaluate_spec(reference, record["spec"])[
                        "defect"
                    ],
                    "truncated4_crop_iou": crop_iou(truncated, reference),
                    "corrected_crop_iou": crop_iou(corrected, reference),
                }
            )
    return examples, missing, checkpoint


def _outline(axis, binary, color, linestyle="solid", linewidth=0.9):
    crop = binary[CROP_START:CROP_STOP, CROP_START:CROP_STOP]
    if crop.any() and not crop.all():
        coordinates = np.arange(CROP_START, CROP_STOP)
        axis.contour(
            coordinates,
            coordinates,
            crop.astype(float),
            levels=[0.5],
            colors=[color],
            linestyles=[linestyle],
            linewidths=[linewidth],
        )


def _terminals(axis, spec):
    for index, (y0, x0, y1, x1) in enumerate(spec["terminals"], start=1):
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=GRAY,
                linewidth=1.0,
                linestyle="dotted",
                zorder=8,
            )
        )
        if CROP_START <= x0 < CROP_STOP and CROP_START <= y0 < CROP_STOP:
            axis.text(
                x1 + 3,
                y0 - 3,
                f"T{index}",
                color=GRAY,
                fontsize=10.5,
                ha="left",
                va="bottom",
                zorder=9,
            )


def render_figure(examples, missing, output, checkpoint_name):

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    figure = plt.figure(figsize=(11.2, 2.45 * (len(examples) + 1) + 0.8))
    grid = figure.add_gridspec(
        len(examples) + 1,
        4,
        height_ratios=[0.95] + [1.0] * len(examples),
        hspace=0.60,
        wspace=0.12,
    )
    mask_cmap = ListedColormap(["#FFFFFF", "#64747E"])
    reference_cmap = ListedColormap(["#FFFFFF", "#D3DCE1"])
    prediction_cmap = ListedColormap(["#FFFFFF", "#DCE9E6"])

    context = figure.add_subplot(grid[0, 0])
    context.imshow(
        examples[0]["input"],
        cmap=mask_cmap,
        vmin=0,
        vmax=1,
        origin="upper",
        extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5),
    )
    context.add_patch(
        Rectangle(
            (CROP_START, CROP_START),
            512,
            512,
            fill=False,
            edgecolor=BLUE,
            linewidth=1.3,
        )
    )
    context.set_title("Full 2048 nm input context")
    context.set_xticks([0, 1024, 2048])
    context.set_yticks([0, 1024, 2048])
    context.set_xlabel("x (nm)")
    context.set_ylabel("y (nm)")

    description = figure.add_subplot(grid[0, 1:])
    description.axis("off")
    description.text(
        0,
        0.99,
        "Nominal focus; amplitude d = 1.0. First example per fixed stratum.\n"
        "Full 2048 nm context; lower panels: central 512 × 512 nm.\n"
        "Terminal ROIs retain full-image coordinates.\n"
        "Reference: 24-mode SOCS; resist threshold = 0.225.",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    description.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.2, label="24-mode reference contour"),
            Line2D([0], [0], color=TEAL, lw=1.2, ls="--", label="Predicted contour"),
            Line2D([0], [0], color=GRAY, lw=1.0, ls=":", label="Fixed terminal ROI"),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.015, -0.07),
        frameon=False,
        ncol=2,
        fontsize=11,
    )

    crop = np.s_[CROP_START:CROP_STOP, CROP_START:CROP_STOP]
    extent = (CROP_START - 0.5, CROP_STOP - 0.5, CROP_STOP - 0.5, CROP_START - 0.5)
    for row_index, example in enumerate(examples, start=1):
        record = example["record"]
        defect = bool(record["truth_focus"]["defect"])
        title = (
            f"Input: {record['id']} · {record['spec']['kind']}\n"
            f"Reference defect = {defect}"
        )
        titles = [
            title,
            "Reference: 24 modes",
            f"K4 only\nCPU crop IoU = {example['truncated4_crop_iou']:.4f}",
            f"K4 + CNN\nCPU crop IoU = {example['corrected_crop_iou']:.4f}",
        ]
        for column, name in enumerate(
            ("input", "reference", "truncated4", "corrected")
        ):
            axis = figure.add_subplot(grid[row_index, column])
            cmap = (
                mask_cmap
                if name == "input"
                else (reference_cmap if name == "reference" else prediction_cmap)
            )
            axis.imshow(
                example[name][crop],
                cmap=cmap,
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
                interpolation="nearest",
            )
            if name != "input":
                _outline(axis, example["reference"], BLUE)
            if name in {"truncated4", "corrected"}:
                _outline(axis, example[name], TEAL, "dashed")
            _terminals(axis, record["spec"])
            axis.set_title(titles[column], pad=8)
            axis.set_xlim(CROP_START - 0.5, CROP_STOP - 0.5)
            axis.set_ylim(CROP_STOP - 0.5, CROP_START - 0.5)
            axis.set_aspect("equal")
            axis.set_xticks([768, 1024, 1280] if row_index == len(examples) else [])
            axis.set_yticks([768, 1024, 1280] if column == 0 else [])
            if row_index == len(examples):
                axis.set_xlabel("x (nm)")
            if column == 0:
                axis.set_ylabel("y (nm)")
            for spine in axis.spines.values():
                spine.set_color("#C6D0D5")
                spine.set_linewidth(0.6)

    caption = (
        "Qualitative reconstruction uses CPU FP32, including model.forward_features and "
        "full 2048-grid spectral synthesis. Displayed IoU is measured only in each 512 nm crop; "
        "it is a visualization diagnostic, not the GPU AMP main quantitative metric. "
        "Sample selection is fixed before model inference and does not rank prediction quality."
    )
    if missing:
        absent = "; ".join(
            f"{item['kind']}, defect={item['defect']}" for item in missing
        )
        caption += f" Missing test strata, omitted: {absent}."
    mismatched = [
        example["record"]["id"]
        for example in examples
        if example["cpu_reference_defect"]
        != bool(example["record"]["truth_focus"]["defect"])
    ]
    if mismatched:
        caption += (
            " CPU reference connectivity differs from stored labels for: "
            + ", ".join(mismatched)
            + "."
        )
    wrapped_caption = "CPU FP32 crop IoUs are visualization diagnostics; quantitative results use GPU AMP."
    figure.text(
        0.055,
        0.016,
        wrapped_caption,
        va="bottom",
        ha="left",
        fontsize=11,
        linespacing=1.35,
    )
    footer_fraction = (
        0.35 + 0.16 * len(wrapped_caption.splitlines())
    ) / figure.get_figheight()
    figure.subplots_adjust(
        left=0.055, right=0.99, top=0.97, bottom=max(0.095, footer_fraction)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(
        output.with_suffix(".svg"),
        metadata={
            "Title": "Fixed lithography verification examples",
            "Description": caption + " Checkpoint: " + checkpoint_name,
        },
    )
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    return caption


def main():
    parser = argparse.ArgumentParser(
        description="Plot fixed test examples with CPU FP32 reconstruction."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    examples, missing, checkpoint = load_examples(args.data, args.model)
    caption = render_figure(examples, missing, args.output, str(args.model))
    print(
        json.dumps(
            {
                "outputs": [
                    str(args.output.with_suffix(suffix))
                    for suffix in (".png", ".svg", ".pdf")
                ],
                "selection": "first per (kind, stored truth_focus.defect)",
                "precision": "CPU FP32; qualitative only",
                "best_epoch": checkpoint.get("best_epoch"),
                "missing_strata": missing,
                "examples": [
                    {
                        "id": example["record"]["id"],
                        "kind": example["record"]["spec"]["kind"],
                        "stored_reference_defect": example["record"]["truth_focus"][
                            "defect"
                        ],
                        "cpu_reference_defect": example["cpu_reference_defect"],
                        "truncated4_cpu_crop_iou": example["truncated4_crop_iou"],
                        "corrected_cpu_crop_iou": example["corrected_crop_iou"],
                    }
                    for example in examples
                ],
                "caption": caption,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
