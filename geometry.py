from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from scipy.stats import beta


def _parse_spec(spec: dict, shape: tuple[int, int]):
    kind = spec["kind"]
    if kind not in ("bridge", "break"):
        raise ValueError("spec.kind must be 'bridge' or 'break'")
    terminals = spec["terminals"]
    if len(terminals) != 2:
        raise ValueError("Exactly two fixed terminal ROIs are required")
    rois = []
    for terminal in terminals:
        if len(terminal) != 4 or any(int(x) != x for x in terminal):
            raise ValueError("Terminal ROIs must be four integer coordinates")
        y0, x0, y1, x1 = map(int, terminal)
        if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
            raise ValueError(f"Terminal ROI {terminal} is outside shape {shape}")
        rois.append((slice(y0, y1), slice(x0, x1)))
    connectivity = spec.get("connectivity", 4)
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    return kind, rois, structure


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must contain only 0 and 1; threshold it explicitly")
    return mask.astype(bool, copy=False)


def _connected(mask: np.ndarray, rois: list, structure: np.ndarray) -> bool:
    labels, _ = ndimage.label(mask, structure=structure)
    first = np.unique(labels[rois[0]])
    second = np.unique(labels[rois[1]])
    return bool(np.intersect1d(first[first != 0], second[second != 0]).size)


def evaluate_spec(mask: np.ndarray, spec: dict) -> dict:

    mask = _binary_mask(mask)
    kind, rois, structure = _parse_spec(spec, mask.shape)
    present = [bool(np.any(mask[roi])) for roi in rois]
    connected = _connected(mask, rois, structure) if all(present) else False
    compliant = all(present) and (connected if kind == "break" else not connected)
    return {
        "kind": kind,
        "terminal_present": present,
        "connected": connected,
        "compliant": bool(compliant),
        "defect": not compliant,
    }


def signed_distance(mask: np.ndarray) -> np.ndarray:

    mask = _binary_mask(mask)
    if not np.any(mask) or np.all(mask):
        raise ValueError("Signed distance requires foreground and background")
    return (
        ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    ).astype(np.float32)


def _prepare_fields(mu: np.ndarray, scale: np.ndarray | float):
    mu = np.asarray(mu, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if mu.ndim != 2 or scale.shape not in ((), mu.shape):
        raise ValueError("mu must be [H,W]; scale must be positive scalar or [H,W]")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(scale)):
        raise ValueError("mu and scale must be finite")
    if np.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    ratio = mu / scale
    if not np.all(np.isfinite(ratio)):
        raise ValueError("mu / scale must be finite")
    return ratio


def _satisfies(ratio, kind, rois, structure, q):
    inner = ratio > q
    if not all(np.any(inner[roi]) for roi in rois):
        return False
    if kind == "break":
        return _connected(inner, rois, structure)
    outer = ratio >= -q
    return not _connected(outer, rois, structure)


def check_margin(
    mu: np.ndarray, scale: np.ndarray | float, spec: dict, q: float
) -> bool:

    if not np.isfinite(q) or q < 0:
        raise ValueError("q must be finite and nonnegative")
    ratio = _prepare_fields(mu, scale)
    return _satisfies(ratio, *_parse_spec(spec, ratio.shape), q)


def geometric_margin(
    mu: np.ndarray,
    scale: np.ndarray | float,
    spec: dict,
    max_iter: int = 10,
) -> float:

    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    ratio = _prepare_fields(mu, scale)
    kind, rois, structure = _parse_spec(spec, ratio.shape)
    if not _satisfies(ratio, kind, rois, structure, 0.0):
        return -1.0
    low = 0.0
    high = min(float(np.max(ratio[roi])) for roi in rois)
    for _ in range(max_iter):
        middle = (low + high) / 2.0
        if _satisfies(ratio, kind, rois, structure, middle):
            low = middle
        else:
            high = middle
    return low


def margin_pass(
    mu: np.ndarray,
    scale: np.ndarray | float,
    spec: dict,
    threshold: float,
    max_iter: int = 10,
) -> bool:

    if np.isnan(threshold):
        raise ValueError("threshold must not be NaN")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if threshold == float("inf"):
        return False
    ratio = _prepare_fields(mu, scale)
    kind, rois, structure = _parse_spec(spec, ratio.shape)
    if not _satisfies(ratio, kind, rois, structure, 0.0):
        return -1.0 > threshold
    low = 0.0
    high = min(float(np.max(ratio[roi])) for roi in rois)
    for _ in range(max_iter):
        if low > threshold:
            return True
        if high <= threshold:
            return False
        middle = (low + high) / 2.0
        if _satisfies(ratio, kind, rois, structure, middle):
            low = middle
        else:
            high = middle
    return low > threshold


def uniform_margin(pred_mask: np.ndarray, spec: dict, max_iter: int = 10) -> float:

    mask = _binary_mask(pred_mask)
    if not evaluate_spec(mask, spec)["compliant"]:
        return -1.0
    if np.all(mask):
        padded = np.pad(mask, 1)
        field = signed_distance(padded)[1:-1, 1:-1]
    else:
        field = signed_distance(mask)
    return geometric_margin(field, 1.0, spec, max_iter=max_iter)


def calibrate_defect_threshold(
    scores: Sequence[float], defects: Sequence[bool], alpha: float = 0.05
) -> float:

    scores = np.asarray(scores, dtype=np.float64)
    defects = np.asarray(defects, dtype=bool)
    if scores.ndim != 1 or scores.shape != defects.shape:
        raise ValueError("scores and defects must be equal-length 1D arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Calibration scores must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    defect_scores = scores[defects]
    rank = math.ceil((len(defect_scores) + 1) * (1.0 - alpha))
    if rank > len(defect_scores):
        return float("inf")
    return float(np.partition(defect_scores, rank - 1)[rank - 1])


def exact_binomial_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> list[float] | None:

    if not 0 <= successes <= trials or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    if trials == 0:
        return None
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else beta.ppf(tail, successes, trials - successes + 1)
    high = (
        1.0
        if successes == trials
        else beta.ppf(1 - tail, successes + 1, trials - successes)
    )
    return [float(low), float(high)]


def summarize_selective(
    scores: Sequence[float],
    defects: Sequence[bool],
    threshold: float,
    confidence: float = 0.95,
) -> dict:

    scores = np.asarray(scores, dtype=np.float64)
    defects = np.asarray(defects, dtype=bool)
    if scores.ndim != 1 or scores.shape != defects.shape:
        raise ValueError("scores and defects must be equal-length 1D arrays")
    if not np.all(np.isfinite(scores)) or np.isnan(threshold):
        raise ValueError("Scores must be finite and threshold must not be NaN")
    passed = scores > threshold
    total = len(scores)
    n_defects = int(defects.sum())
    n_passed = int(passed.sum())
    n_missed = int((passed & defects).sum())
    n_healthy = total - n_defects
    n_healthy_accepted = n_passed - n_missed

    def rate(num, den):
        return float(num / den) if den else None

    return {
        "n": total,
        "n_defects": n_defects,
        "n_passed": n_passed,
        "n_referred": total - n_passed,
        "n_healthy": n_healthy,
        "n_healthy_accepted": n_healthy_accepted,
        "healthy_acceptance_rate": rate(n_healthy_accepted, n_healthy),
        "healthy_acceptance_rate_ci": exact_binomial_interval(
            n_healthy_accepted, n_healthy, confidence
        ),
        "n_missed_defects": n_missed,
        "threshold": float(threshold),
        "pass_rate": rate(n_passed, total),
        "pass_rate_ci": exact_binomial_interval(n_passed, total, confidence),
        "defect_miss_rate": rate(n_missed, n_defects),
        "defect_miss_rate_ci": exact_binomial_interval(n_missed, n_defects, confidence),
        "defects_among_passed": rate(n_missed, n_passed),
        "defects_among_passed_ci": exact_binomial_interval(
            n_missed, n_passed, confidence
        ),
        "confidence": confidence,
    }
