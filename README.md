# RCPWV

Implementation of **Risk-Calibrated Process-Window Verification of Lithography Surrogates: Image Fidelity, Connectivity, and End-to-End Cost**.

RCPWV combines a compact optical residual CNN, fixed-terminal connectivity scores, defect-stratified calibration, and full-resolution reference fallback. It compares image fidelity, acceptance quality, and complete verification cost on ordinary procedural layouts, critical-window stress layouts, and public metal-1 layout windows.

## Installation

Use Python 3.11 or newer. An NVIDIA GPU is recommended for the paper-sized workload; every computational entry point also accepts `--device cpu`. The feature cache for the default training split alone occupies about 1.4 GiB before loader copies, so allow several GiB of host memory and disk space.

```bash
git clone https://github.com/Hqa77880011/RCPWV.git
cd RCPWV
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in PowerShell. Then install PyTorch and the project:

```bash
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
```

For a CUDA installation, choose the wheel matching your driver using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). A CPU-only environment can use:

```bash
python -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Run commands from the repository root. The implementation uses PyTorch, NumPy, SciPy, Matplotlib, OpenCV and gdstk. CUDA enables automatic mixed precision; CPU uses FP32.

## Data sources

Download the optical kernels, their configuration, and the two public GDS sources:

```bash
python download_sources.py
```

The downloader pins the source revisions used to define the data preparation protocol:

| Source | Files | Revision |
| --- | --- | --- |
| [LithoBench](https://github.com/shelljane/lithobench) | Focus/defocus kernels and scales, `lithosimple.txt`, license | `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6` |
| [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) | Nangate45 GDS, layer map, license | `e54722b45e683cd3f2a7506cb50596df8e0dc5cd` |
| [OpenILT](https://github.com/OpenOPC/OpenILT) | `gcd_45nm.gds`, license | `dabb97c6ca3dfd159362e48273c436444c77353b` |

Files are downloaded into `vendor/`, with the corresponding source licenses. `--optics-only` downloads just the optical resources for procedural experiments. Downloaded files are reused. Keep the pinned files together; replacing one kernel changes the reference model.

The procedural datasets are generated locally. The public check uses new native-scale windows extracted from the GDS sources: two drive-strength variants from each of eight held-out Nangate45 families, plus four predetermined GCD windows. The generated windows form a separate evaluation cohort from LithoBench's published dataset splits. Labels throughout this project are SOCS simulations.

## Run the workflow

The stages below follow the data and model dependencies. `configs/paper.toml` sets 1,536 training, 192 validation, 512 calibration and 512 test sources, optimizer seeds 0/1/2, 30 training epochs, and 256/256 critical calibration/test sources. Each procedural split has equal bridge and break quotas.

```bash
python run_all.py --stage prepare --device cuda
python run_all.py --stage train --device cuda
python run_all.py --stage evaluate --device cuda
python run_all.py --stage critical --device cuda
python run_all.py --stage public --device cuda
python run_all.py --stage report
```

Alternatively, `python run_all.py --stage all --device cuda` performs download and every stage in order. These commands launch the full workload. Use `--work PATH` to choose a fresh output root, `--config PATH` to select a TOML configuration, and `--device cpu` for CPU execution. The aggregate workflow keeps three optimizer seeds and three timing repeats; individual scripts support smaller or custom runs.

Data and experiment commands refuse to overwrite their outputs. After completing a stage, invoke the next stage. To repeat the workflow with different settings, use a new `--work` directory. For an interrupted stage, use the individual commands below to run the remaining jobs into fresh output directories.

### 1. Prepare data

```bash
python prepare_data.py --output work/data/ordinary --train 1536 --val 192 --cal 512 --test 512 --seed 20261009 --public --device cuda
```

`--train`, `--val`, `--cal` and `--test` are even numbers of source layouts. Both supplied optical conditions stay with their source in the same split. `--public` additionally prepares development families and the held-out public windows. `--public-only` prepares those public cohorts in a fresh directory without procedural generation.

Each split has a `.npz` archive and a `.json` record list:

| Field | Shape | Meaning |
| --- | --- | --- |
| `features` | `N,2,5,128,128` | Four weighted coherent-mode intensities and the real band-limited mask |
| `aerial_grid` | `N,2,128,128` | Nominal-amplitude 24-mode reference intensity |
| `masks_packed` | `N,2048,256` | Full binary input packed in NumPy big-bit order |

The condition order is focus, defocus. Records include source groups, fixed terminal specifications, generation parameters, and full-resolution connectivity labels. `dataset_summary.json` records counts and reference consistency checks. Public records also store all input-selected specifications.

### 2. Train the surrogate and learned baselines

```bash
python train.py --data work/data/ordinary --output work/runs/k4_seed0 --kernels 4 --seed 0 --epochs 30 --batch-size 32 --device cuda
python train.py --data work/data/ordinary --output work/runs/k0_seed0 --kernels 0 --seed 0 --epochs 30 --batch-size 32 --device cuda
python train.py --data work/data/ordinary --output work/runs/k4_no_boundary_seed0 --kernels 4 --boundary-weight 0 --seed 0 --epochs 30 --batch-size 32 --device cuda
python classifier.py --data work/data/ordinary --output work/runs/classifier_seed0 --seed 0 --epochs 30 --batch-size 8 --device cuda
```

Use seeds 1 and 2 for the main surrogate, K0 ablation and classifier. `run_all.py --stage train` runs all ten jobs, including the single-seed loss-weighting ablation.

The surrogate has four circularly padded dilated convolution layers of width 16 and a 1×1 output layer. Its nonnegative residual corrects the four-mode sum. `--kernels 0` removes optical intensity channels while retaining the mask, condition flag, bandwidth projection and amplitude-square law. `--boundary-weight 0` removes threshold emphasis from the training loss; validation retains weight 9. Other useful parameters are `--lr`, `--boundary-width`, `--width`, `--threads` and `--batch-size`; `--help` lists defaults.

Both trainers use AdamW and cosine annealing, read only train/validation splits, and select checkpoints by validation loss. Outputs are `best.pt`, `epochs.jsonl` and `summary.json`, including the selected epoch, parameter count, training time and peak allocated GPU memory.

### 3. Evaluate ordinary layouts and calibrate the window event

```bash
python evaluate.py --data work/data/ordinary --model work/runs/k4_seed0/best.pt --output work/results/ordinary/k4_seed0 --profiles focus defocus focus_low focus_high defocus_low defocus_high --device cuda
python evaluate_classifier.py --data work/data/ordinary --classifier work/runs/classifier_seed0/best.pt --output work/results/ordinary/classifier_seed0 --batch-size 1 --device cuda
python calibrate_window.py --evaluation work/results/ordinary/k4_seed0 --classifier work/results/ordinary/classifier_seed0/per_sample.csv --output work/results/ordinary/window_seed0.json --seed 0
```

Repeat for the other optimizer seeds and evaluate the K0 and no-boundary models. `run_all.py --stage evaluate` runs these comparisons and the nominal/window timing commands.

`evaluate.py` writes per-source image metrics, full-resolution defect labels and coarse scores to `per_sample.csv`. `results.json` contains per-profile calibration and nominal-threshold transfer comparisons. `--profiles` selects optical conditions/amplitudes. `--score-size 256` is the paper's routing grid; supported alternatives preserve the full physical domain and change the geometry resolution. Calibration, online execution and timing must use the same grid and numerical precision path. `--batch-size 1` matches the deployment timing convention.

`calibrate_window.py` takes the union of the four endpoint labels as one event per source. It uses the endpoint-minimum score for window policies and the nominal-focus score for `nominal_aerial_only`. Both are calibrated against that same window-defect event. Each risk budget, 0.01/0.05/0.10, gets separate bridge and break thresholds. JSON `null` thresholds represent positive infinity, which refers every source of that kind.

The available comparisons are:

| Score/model | Purpose |
| --- | --- |
| `aerial_margin` | K4+CNN with task-specific intensity margin; minimum over endpoints in a window policy |
| `nominal_aerial_only` | One nominal CNN score calibrated against window labels |
| `truncated1_margin`, `truncated4_margin` | Physical mode truncation without learning |
| `spectral24_margin` | All optical modes with coarse routing geometry |
| `uniform_margin` | Euclidean signed-distance margin of the CNN prediction |
| `input_uniform_margin` | Input-mask EDT without optical prediction |
| `global_confidence` | Mean absolute intensity offset when the coarse specification is compliant |
| `classifier` | Negative defect probability from the terminal-conditioned classifier |
| K0 / no-boundary models | Optical-feature and loss-weighting ablations |

### 4. Prepare and evaluate critical windows

```bash
python prepare_critical.py --output work/data/critical --cal 256 --test 256 --seed 20261010 --device cuda
python evaluate.py --data work/data/critical --model work/runs/k4_seed0/best.pt --output work/results/critical/k4_seed0 --profiles focus defocus focus_low focus_high defocus_low defocus_high --device cuda
python evaluate_classifier.py --data work/data/critical --classifier work/runs/classifier_seed0/best.pt --output work/results/critical/classifier_seed0 --batch-size 1 --device cuda
python calibrate_window.py --evaluation work/results/critical/k4_seed0 --classifier work/results/critical/classifier_seed0/per_sample.csv --output work/results/critical/window_seed0.json --seed 0
```

Repeat evaluation and calibration for seeds 1/2 without retraining. The critical generator searches for a window-label transition by varying bridge gap or break neck width, then applies independent integer jitter in `[-6,6]`. Sources without a transition bracket remain in the split. This defines a simulator-mined stress distribution, with separate source seeds for calibration and test.

### 5. Measure complete verification cost

```bash
python benchmark_nominal.py --data work/data/ordinary --model work/runs/k4_seed0/best.pt --evaluation work/results/ordinary/k4_seed0 --output work/results/ordinary/nominal_timing.json --samples 128 --repeats 3 --device cuda
python benchmark_classifier.py --data work/data/ordinary --classifier work/runs/classifier_seed0/best.pt --evaluation work/results/ordinary/classifier_seed0 --output work/results/ordinary/classifier_timing.json --samples 128 --repeats 3 --device cuda
python benchmark_window.py --data work/data/critical --model work/runs/k4_seed0/best.pt --classifier work/runs/classifier_seed0/best.pt --evaluation work/results/critical/k4_seed0 --window-policy work/results/critical/window_seed0.json --output work/results/critical/window_timing.json --samples 128 --repeats 3 --device cuda
```

For ordinary-window timing, use the ordinary data/evaluation/policy paths and write `work/results/ordinary/window_timing.json`. `--samples` selects the predetermined first test sources; `--repeats` sets complete timing repetitions and `--alpha` selects a saved policy.

Timings start from preloaded full binary masks and include preprocessing, transfers, optical features, CNN/classifier computation, CPU geometry and actual full-resolution fallback. They exclude file I/O, model loading, training and offline calibration. Each query shares its input FFT, condition-specific modes and nominal intensities; fallback extends computed modes to 24. The scripts check strict online/offline decisions and reference labels while recording all repeat times and work counters. Speedup is the median full-reference time divided by the median complete policy time on the same subset.

### 6. Evaluate the frozen public policy

```bash
python evaluate_public.py --public-data work/data/ordinary/public --model work/runs/k4_seed0/best.pt --classifier work/runs/classifier_seed0/best.pt --window-policy work/results/critical/window_seed0.json --output work/results/public --device cuda
```

This evaluates the 20 public windows and 80 fixed specifications using critical seed-0 thresholds at 5%, without recalibration. It writes `checks.csv`, `summary.json` and `RESULTS.html`. The summary contains event counts, source-group macro rates, all three timing repeats and actual work counters. Correlated layout variants and specifications retain their source groups.

### 7. Summarize and visualize

```bash
python summarize.py --work work --resamples 2000
python plot_examples.py --data work/data/ordinary --model work/runs/k4_seed0/best.pt --output work/figures/qualitative
```

The summary reads the complete ordinary, critical and public outputs from the preceding stages. It produces `work/report/index.html`, `summary.json`, window comparison plots and CSV tables for fidelity (including optimizer-seed means and sample standard deviations), nominal/process-shift policies, window budgets, classifier metrics, training resources and runtime. Paired bootstrap comparisons average optimizer-seed decisions per source before resampling within kind. Models and calibration thresholds remain frozen. The qualitative figure uses the first source in each kind/reference-defect stratum and saves PNG, SVG and PDF.

To report the numerical difference between full-grid and compact-grid SOCS, use:

```bash
python compare_optics.py --public-data work/data/ordinary/public --samples 12 --doses 0.98 1.0 1.02 --output work/report/optical_precision.json --device cuda
```

This compares 72 layout/condition/amplitude combinations and records maximum absolute intensity differences and thresholded-pixel disagreements. It does not train or select a model.

## Verify a new layout

Supply a binary `2048×2048` `.npy` or image at 1 nm/pixel, and a JSON specification. Terminals use half-open `[y0,x0,y1,x1]` rectangles:

```json
{
  "kind": "bridge",
  "connectivity": 4,
  "terminals": [[900, 800, 909, 809], [900, 1200, 909, 1209]]
}
```

`bridge` requires the two terminals to remain separated; `break` requires them to connect. Either missing terminal is a defect. Paths may use the full canvas.

```bash
python verify.py --mask mask.npy --spec spec.json --model work/runs/k4_seed0/best.pt --policy work/results/critical/window_seed0.json --method nominal_aerial_only --alpha 0.05 --output decision.json --device cuda
```

Use `--method aerial_margin` for endpoint CNN routing, `spectral24_margin` for coarse full-mode routing, or `all_spectral24` for full reference verification. Classifier routing additionally requires `--classifier work/runs/classifier_seed0/best.pt`. The output contains the decision, automatic-acceptance flag, fallback label when computed, threshold and work counters. A `null` reference label means the query was automatically accepted and no reference label was computed. Retain the policy's referenced evaluation files and calibrated checkpoints; the command checks their association.

## Metrics and implementation conventions

The optical model uses a periodic 2048 nm square at 1 nm/pixel, 24 coherent modes, threshold `0.225`, and two supplied categorical focus/defocus settings. Their physical defocus magnitude is unspecified. Optical field frequencies `[-17,17]²` and intensity frequencies `[-34,34]²` are evaluated on a 128 grid, then synthesized at the requested physical sample centers. The physical domain stays fixed when the grid changes. Positive amplitude multiplies intensity by its square. The window is `{focus,defocus} × [0.98,1.02]`; endpoint foreground nesting covers this amplitude interval within each supplied kernel.

The 256-grid routing score uses the inner set `I−τ > q`, outer set `I−τ ≥ −q`, sample-center terminal mapping with ceiling endpoints, and ten deterministic bisection steps. Failed zero-margin checks receive `-1`. Acceptance is strictly `score > threshold`. Calibration uses only defective sources of the matching kind, with order-statistic rank `ceil((n+1)(1−alpha))`. Insufficient rank support produces an infinite threshold. An uncalibrated geometric rule uses `score ≥ 0`; the uncalibrated classifier uses defect probability below `0.5`.

| Metric | Denominator / interpretation |
| --- | --- |
| Full / central IoU | Foreground union on the full 2048 grid / fixed central `[768,1280)²` region |
| Defect false acceptance | Automatically accepted defective sources / all defective sources |
| Healthy acceptance | Automatically accepted healthy sources / all healthy sources |
| Total acceptance | All automatically accepted sources / all sources |
| Defects among accepted | Automatically accepted defective sources / all accepted sources |
| AUROC, BCE, accuracy | Nominal classifier discrimination, also reported within kind |
| Complete runtime / speedup | Same queried subset, actual fallback, median of three repeats |
| Resources / work | Parameter count, training time, peak GPU allocation, FFT/mode/geometry counters |

Empty denominators are `null`, rather than zero. Procedural intervals combine within-kind exact Clopper–Pearson bounds with Bonferroni adjustment and observed denominator weights. Optimizer-seed means and sample standard deviations describe optimization variability on shared sources. Public rates are descriptive under source shift and are also summarized by source group; calibration's exchangeability bound does not transfer automatically to those sources.

The published configuration follows the paper's specified architecture, losses, data roles, calibration event, baselines, ablations and metrics. Procedural clutter generation, integer rasterization, source ordering and terminal selection are explicit deterministic conventions in `patterns.py` and `prepare_data.py`. Pixel connectivity uses four neighbors and does not wrap across the image edge, while optical synthesis is periodic. Checkpoints use native PyTorch state dictionaries. Runtime comparisons require matching the device, batch size, routing grid and AMP/FP32 path used for calibration. The qualitative CPU FP32 crop IoUs are separate from quantitative CUDA AMP metrics.

CI checks dependency installation, Python syntax and command-line imports. Experiment execution is left to the commands above; the repository does not ship measured results, generated data or trained checkpoints.

## License

The implementation is released under the MIT license. Downloaded optical and GDS resources retain their source licenses in `vendor/`.
