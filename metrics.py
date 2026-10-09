import numpy as np

from geometry import (
    calibrate_defect_threshold,
    exact_binomial_interval,
    summarize_selective,
)

KINDS = ("bridge", "break")


def aggregate_stratified(passed, defects, kinds, confidence=0.95):

    passed = np.asarray(passed, dtype=bool)
    defects = np.asarray(defects, dtype=bool)
    kinds = np.asarray(kinds)
    summary = summarize_selective(passed.astype(float), defects, 0.5, confidence)
    summary.pop("threshold")
    active_kinds = [kind for kind in KINDS if np.any(kinds == kind)]
    adjusted_confidence = 1 - (1 - confidence) / max(len(active_kinds), 1)
    for metric, numerator, denominator in [
        ("pass_rate", passed, np.ones(len(passed), dtype=bool)),
        ("defect_miss_rate", passed & defects, defects),
        ("healthy_acceptance_rate", passed & ~defects, ~defects),
        ("defects_among_passed", passed & defects, passed),
    ]:
        weighted = np.zeros(2)
        total = int(denominator.sum())
        for kind in active_kinds:
            subset = kinds == kind
            n = int((denominator & subset).sum())
            if n:
                k = int((numerator & subset).sum())
                interval = exact_binomial_interval(k, n, adjusted_confidence)
                weighted += n * np.asarray(interval)
        summary[f"{metric}_ci"] = (weighted / total).tolist() if total else None
    summary["interval_method"] = "kind-stratified Bonferroni-weighted Clopper-Pearson"
    return summary


def summarize_policies(cal_rows, test_rows, methods, alphas):

    cal_defects = np.array([row["defect"] for row in cal_rows], dtype=bool)
    test_defects = np.array([row["defect"] for row in test_rows], dtype=bool)
    cal_kinds = np.array([row["kind"] for row in cal_rows])
    test_kinds = np.array([row["kind"] for row in test_rows])
    primary, pooled = {}, {}
    for method in methods:
        primary[method], pooled[method] = {}, {}
        cal_scores = np.array([row[method] for row in cal_rows])
        test_scores = np.array([row[method] for row in test_rows])
        for alpha in alphas:
            thresholds, by_kind, calibration_defects = {}, {}, {}
            passed = np.zeros(len(test_rows), dtype=bool)
            for kind in KINDS:
                cal_subset, test_subset = cal_kinds == kind, test_kinds == kind
                threshold = calibrate_defect_threshold(
                    cal_scores[cal_subset], cal_defects[cal_subset], alpha
                )
                thresholds[kind] = threshold
                calibration_defects[kind] = int(cal_defects[cal_subset].sum())
                passed[test_subset] = test_scores[test_subset] > threshold
                by_kind[kind] = summarize_selective(
                    test_scores[test_subset], test_defects[test_subset], threshold
                )
            primary[method][str(alpha)] = {
                "thresholds": thresholds,
                "threshold_finite": {
                    kind: bool(np.isfinite(value)) for kind, value in thresholds.items()
                },
                "calibration_defects": calibration_defects,
                "test": aggregate_stratified(passed, test_defects, test_kinds),
                "by_kind": by_kind,
            }
            threshold = calibrate_defect_threshold(cal_scores, cal_defects, alpha)
            pooled[method][str(alpha)] = {
                "threshold": threshold,
                "threshold_finite": bool(np.isfinite(threshold)),
                "calibration_defects": int(cal_defects.sum()),
                "test": aggregate_stratified(
                    test_scores > threshold, test_defects, test_kinds
                ),
                "by_kind": {
                    kind: summarize_selective(
                        test_scores[test_kinds == kind],
                        test_defects[test_kinds == kind],
                        threshold,
                    )
                    for kind in KINDS
                },
            }
    return primary, pooled


def json_finite(value):

    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
