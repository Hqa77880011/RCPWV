import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = ("download", "prepare", "train", "evaluate", "critical", "public", "report")
PROFILES = (
    "focus",
    "defocus",
    "focus_low",
    "focus_high",
    "defocus_low",
    "defocus_high",
)


def invoke(script, *arguments):
    command = [sys.executable, "-B", str(ROOT / script), *map(str, arguments)]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def fresh_run(script, marker, *arguments):
    if marker.exists():
        raise FileExistsError(
            f"Output already exists: {marker}. Choose a new --work directory or run the next stage."
        )
    invoke(script, *arguments)


def train_models(args, config, data, runs):
    jobs = [(f"k4_seed{seed}", 4, 9.0, seed) for seed in config["seeds"]]
    jobs += [(f"k0_seed{seed}", 0, 9.0, seed) for seed in config["seeds"]]
    jobs.append(("k4_no_boundary_seed0", 4, 0.0, 0))
    for name, kernels, weight, seed in jobs:
        output = runs / name
        fresh_run(
            "train.py",
            output / "best.pt",
            "--data",
            data,
            "--output",
            output,
            "--epochs",
            config["epochs"],
            "--batch-size",
            config["surrogate_batch"],
            "--kernels",
            kernels,
            "--width",
            16,
            "--seed",
            seed,
            "--boundary-weight",
            weight,
            "--device",
            args.device,
        )
    for seed in config["seeds"]:
        output = runs / f"classifier_seed{seed}"
        fresh_run(
            "classifier.py",
            output / "best.pt",
            "--data",
            data,
            "--output",
            output,
            "--epochs",
            config["epochs"],
            "--batch-size",
            config["classifier_batch"],
            "--seed",
            seed,
            "--device",
            args.device,
        )


def evaluate_models(args, config, data, runs, results, critical=False):
    jobs = [f"k4_seed{seed}" for seed in config["seeds"]]
    if not critical:
        jobs += [f"k0_seed{seed}" for seed in config["seeds"]]
        jobs.append("k4_no_boundary_seed0")
    for name in jobs:
        output = results / name
        fresh_run(
            "evaluate.py",
            output / "results.json",
            "--data",
            data,
            "--model",
            runs / name / "best.pt",
            "--output",
            output,
            "--profiles",
            *PROFILES,
            "--device",
            args.device,
        )
    for seed in config["seeds"]:
        output = results / f"classifier_seed{seed}"
        fresh_run(
            "evaluate_classifier.py",
            output / "results.json",
            "--data",
            data,
            "--classifier",
            runs / f"classifier_seed{seed}/best.pt",
            "--output",
            output,
            "--device",
            args.device,
            "--batch-size",
            1,
        )
        policy = results / f"window_seed{seed}.json"
        fresh_run(
            "calibrate_window.py",
            policy,
            "--evaluation",
            results / f"k4_seed{seed}",
            "--classifier",
            output / "per_sample.csv",
            "--output",
            policy,
            "--seed",
            seed,
        )
    if not critical:
        for script, output, flag, checkpoint, evaluation in (
            (
                "benchmark_nominal.py",
                "nominal_timing.json",
                "--model",
                runs / "k4_seed0/best.pt",
                results / "k4_seed0",
            ),
            (
                "benchmark_classifier.py",
                "classifier_timing.json",
                "--classifier",
                runs / "classifier_seed0/best.pt",
                results / "classifier_seed0",
            ),
        ):
            fresh_run(
                script,
                results / output,
                "--data",
                data,
                flag,
                checkpoint,
                "--evaluation",
                evaluation,
                "--output",
                results / output,
                "--samples",
                config["timing_samples"],
                "--repeats",
                config["timing_repeats"],
                "--device",
                args.device,
            )
    fresh_run(
        "benchmark_window.py",
        results / "window_timing.json",
        "--data",
        data,
        "--model",
        runs / "k4_seed0/best.pt",
        "--classifier",
        runs / "classifier_seed0/best.pt",
        "--evaluation",
        results / "k4_seed0",
        "--window-policy",
        results / "window_seed0.json",
        "--output",
        results / "window_timing.json",
        "--samples",
        config["timing_samples"],
        "--repeats",
        config["timing_repeats"],
        "--device",
        args.device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the RCPWV data, training, evaluation and reporting stages."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.toml")
    parser.add_argument(
        "--work",
        type=Path,
        default=ROOT / "work",
        help="Root for data, checkpoints, scores and figures.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    required = (
        "seeds",
        "epochs",
        "surrogate_batch",
        "classifier_batch",
        "train",
        "val",
        "cal",
        "test",
        "critical_cal",
        "critical_test",
        "timing_samples",
        "timing_repeats",
    )
    if any(key not in config for key in required):
        parser.error("Configuration must provide the fields in configs/paper.toml")
    if config["seeds"] != [0, 1, 2]:
        parser.error(
            "The aggregate protocol uses optimizer seeds [0, 1, 2]; use train.py for individual runs"
        )
    if config["timing_repeats"] != 3:
        parser.error("The aggregate and public protocol uses three timing repeats")
    work = args.work.resolve()
    data, runs = work / "data/ordinary", work / "runs"
    ordinary, critical = work / "results/ordinary", work / "results/critical"
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "download":
            invoke("download_sources.py")
        elif stage == "prepare":
            fresh_run(
                "prepare_data.py",
                data / "dataset_summary.json",
                "--output",
                data,
                "--train",
                config["train"],
                "--val",
                config["val"],
                "--cal",
                config["cal"],
                "--test",
                config["test"],
                "--public",
                "--device",
                args.device,
            )
        elif stage == "train":
            train_models(args, config, data, runs)
        elif stage == "evaluate":
            evaluate_models(args, config, data, runs, ordinary)
        elif stage == "critical":
            critical_data = work / "data/critical"
            fresh_run(
                "prepare_critical.py",
                critical_data / "dataset_summary.json",
                "--output",
                critical_data,
                "--cal",
                config["critical_cal"],
                "--test",
                config["critical_test"],
                "--device",
                args.device,
            )
            evaluate_models(args, config, critical_data, runs, critical, critical=True)
        elif stage == "public":
            fresh_run(
                "evaluate_public.py",
                work / "results/public/summary.json",
                "--public-data",
                data / "public",
                "--model",
                runs / "k4_seed0/best.pt",
                "--classifier",
                runs / "classifier_seed0/best.pt",
                "--window-policy",
                critical / "window_seed0.json",
                "--output",
                work / "results/public",
                "--device",
                args.device,
            )
        elif stage == "report":
            invoke("summarize.py", "--work", work)
            invoke(
                "plot_examples.py",
                "--data",
                data,
                "--model",
                runs / "k4_seed0/best.pt",
                "--output",
                work / "figures/qualitative",
            )
    print(json.dumps({"completed_stages": list(stages), "work": str(work)}), flush=True)


if __name__ == "__main__":
    main()
