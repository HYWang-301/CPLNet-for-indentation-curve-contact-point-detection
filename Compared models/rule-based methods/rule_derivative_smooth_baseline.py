# -*- coding: utf-8 -*-
"""
AFM contact point baseline: Derivative + smooth

This script implements the FC_analysis paper's rule-driven contact point method
"Derivative + smooth" and evaluates it against manually labelled ContactIndex
values in the original, un-resampled index space.

Paper description:
    Smooth force curve -> derivative -> smooth derivative -> scan backward from
    the maximum-load point and select the first zero value on the smoothed
    derivative curve.

Parameter notes from the paper:
    - The smoothing interval is user-defined. The GUI example and the peeled-RBC
      example use 5 points for "derivative + smooth".
    - FC_analysis uses a zero-line fit/subtraction before contact point analysis;
      reported examples use ranges such as 0-60%, 0-70%, 10-80%, 10-85%.
      The Figure 2/3 example illustrates 10-70% for zero-line fitting.

Your txt format:
    first half  = deflection
    second half = Z_displacement
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# 1) Paths: edit these before running on your computer
# =============================================================================
DATA_DIR = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\组织+细胞"
LABELS_EXCEL = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\组织+细胞\contact_points_labels.xlsx"
SAVE_DIR = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\rule_baselines_fc_analysis\derivative_smooth"

# Excel column names used by your V14 dataset code
FILENAME_COL = "FileName"
CONTACT_COL = "ContactIndex"


# =============================================================================
# 2) Method parameters: paper-style user-settable parameters
# =============================================================================
# Moving-average smoothing interval. Paper examples for derivative+smooth use 5.
SMOOTHING_POINTS_FORCE = 5
SMOOTHING_POINTS_DERIVATIVE = 5

# Zero-line subtraction before CP detection.
# Figure 2/3 illustrate 10-70%; examples also use 0-60%, 0-70%, 10-80%, 10-85%.
USE_ZEROLINE_SUBTRACTION = True
ZEROLINE_FIT_START_PCT = 10.0
ZEROLINE_FIT_END_PCT = 70.0

# The paper says scan from the point of maximum load. For approach-only curves
# this is often the final point. Use "end" for that literal interpretation, or
# "argmax" to use the maximum of the processed force signal.
MAX_LOAD_MODE = "end"  # "end" or "argmax"

# Force sign handling. "auto" flips the force signal if the final-load region is
# below the zero-line region, so the contact part rises upward.
FORCE_SIGN_MODE = "auto"  # "auto", "positive", or "negative"
SIGN_ESTIMATE_HEAD_PCT = 10.0
SIGN_ESTIMATE_TAIL_PCT = 10.0

# Numerical zero threshold for the smoothed derivative.
# If DERIVATIVE_ABS_ZERO_THRESHOLD is None, threshold = factor * robust noise
# estimated from the zero-line derivative segment.
DERIVATIVE_ABS_ZERO_THRESHOLD: Optional[float] = None
DERIVATIVE_ZERO_NOISE_FACTOR = 3.0
MIN_DERIVATIVE_ZERO_THRESHOLD = 1e-12
CONSECUTIVE_ZERO_POINTS = 3

# Optional guard against detecting very early/late artifacts.
MIN_SEARCH_INDEX_PCT = 0.0
MAX_SEARCH_INDEX_PCT = 100.0

# Save per-curve predictions in addition to summary metrics.
SAVE_PREDICTIONS_CSV = True


# =============================================================================
# 3) Utilities
# =============================================================================
def load_labels(excel_path: str) -> Dict[str, int]:
    df = pd.read_excel(excel_path)
    if FILENAME_COL not in df.columns or CONTACT_COL not in df.columns:
        raise ValueError(
            f"Excel must contain columns {FILENAME_COL!r} and {CONTACT_COL!r}; "
            f"got columns: {list(df.columns)}"
        )
    return {str(row[FILENAME_COL]): int(row[CONTACT_COL]) for _, row in df.iterrows()}


def load_curve_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError:
                continue

    data = np.asarray(values, dtype=np.float64)
    if data.size < 4:
        raise ValueError(f"too few numeric values: {data.size}")
    mid = data.size // 2
    deflection = data[:mid]
    z_displacement = data[mid:]
    n = min(deflection.size, z_displacement.size)
    if n < 4:
        raise ValueError(f"too few paired points: {n}")
    return deflection[:n], z_displacement[:n]


def moving_average_reflect(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, round(window)))
    if window <= 1 or y.size <= 2:
        return y.astype(np.float64, copy=True)
    if window > y.size:
        window = y.size
    # Prefer odd windows to avoid half-sample phase shifts.
    if window % 2 == 0:
        window += 1
        if window > y.size:
            window -= 2
    if window <= 1:
        return y.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(y.astype(np.float64), pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def pct_to_slice(n: int, start_pct: float, end_pct: float) -> slice:
    lo = int(np.floor(np.clip(start_pct, 0, 100) / 100.0 * (n - 1)))
    hi = int(np.ceil(np.clip(end_pct, 0, 100) / 100.0 * (n - 1))) + 1
    lo, hi = max(0, min(lo, n - 1)), max(1, min(hi, n))
    if hi <= lo + 1:
        hi = min(n, lo + 2)
    return slice(lo, hi)


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(x))


def linear_baseline_subtract(y: np.ndarray) -> np.ndarray:
    if not USE_ZEROLINE_SUBTRACTION:
        return y.astype(np.float64, copy=True)
    n = y.size
    fit_sl = pct_to_slice(n, ZEROLINE_FIT_START_PCT, ZEROLINE_FIT_END_PCT)
    x_fit = np.arange(n, dtype=np.float64)[fit_sl]
    y_fit = y[fit_sl]
    if x_fit.size < 2:
        return y.astype(np.float64, copy=True)
    a, b = np.polyfit(x_fit, y_fit, 1)
    baseline = a * np.arange(n, dtype=np.float64) + b
    return y - baseline


def orient_force(y: np.ndarray) -> np.ndarray:
    mode = FORCE_SIGN_MODE.lower().strip()
    if mode == "positive":
        return y
    if mode == "negative":
        return -y
    if mode != "auto":
        raise ValueError("FORCE_SIGN_MODE must be 'auto', 'positive', or 'negative'")

    n = y.size
    head_sl = pct_to_slice(n, 0.0, SIGN_ESTIMATE_HEAD_PCT)
    tail_sl = pct_to_slice(n, 100.0 - SIGN_ESTIMATE_TAIL_PCT, 100.0)
    head = float(np.median(y[head_sl]))
    tail = float(np.median(y[tail_sl]))
    return y if tail >= head else -y


def max_load_index(y: np.ndarray) -> int:
    mode = MAX_LOAD_MODE.lower().strip()
    if mode == "end":
        return int(y.size - 1)
    if mode == "argmax":
        return int(np.nanargmax(y))
    raise ValueError("MAX_LOAD_MODE must be 'end' or 'argmax'")


# =============================================================================
# 4) Derivative + smooth CP detector
# =============================================================================
def detect_contact_derivative_smooth(deflection: np.ndarray) -> int:
    y = np.asarray(deflection, dtype=np.float64)
    if y.size < 4:
        return 0

    # Preliminary smoothing, as requested and as in the paper routine.
    y_sm = moving_average_reflect(y, SMOOTHING_POINTS_FORCE)
    y_proc = orient_force(linear_baseline_subtract(y_sm))

    # Derivative and smoothing of derivative.
    dy = np.gradient(y_proc)
    dy_sm = moving_average_reflect(dy, SMOOTHING_POINTS_DERIVATIVE)

    n = y.size
    start_idx = max_load_index(y_proc)
    lo = int(np.floor(np.clip(MIN_SEARCH_INDEX_PCT, 0, 100) / 100.0 * (n - 1)))
    hi = int(np.ceil(np.clip(MAX_SEARCH_INDEX_PCT, 0, 100) / 100.0 * (n - 1)))
    start_idx = int(np.clip(start_idx, lo + 1, hi))

    # Estimate numerical zero from the zero-line derivative segment.
    if DERIVATIVE_ABS_ZERO_THRESHOLD is not None:
        thr = float(DERIVATIVE_ABS_ZERO_THRESHOLD)
    else:
        noise_sl = pct_to_slice(n, ZEROLINE_FIT_START_PCT, ZEROLINE_FIT_END_PCT)
        thr = DERIVATIVE_ZERO_NOISE_FACTOR * robust_std(dy_sm[noise_sl])
        thr = max(float(thr), MIN_DERIVATIVE_ZERO_THRESHOLD)

    # The contact-line derivative should be positive after auto-orientation.
    # Scanning backward from max load, the first run near zero is the CP region.
    run = max(1, int(CONSECUTIVE_ZERO_POINTS))
    for i in range(start_idx, lo + run - 2, -1):
        seg = dy_sm[max(lo, i - run + 1): i + 1]
        if seg.size >= run and np.all(np.abs(seg) <= thr):
            return int(np.clip(i, 0, n - 1))

    # Fallback: closest-to-zero derivative before the maximum-load point.
    search = np.abs(dy_sm[lo:start_idx + 1])
    if search.size == 0 or not np.isfinite(search).any():
        return int(np.clip(start_idx, 0, n - 1))
    return int(lo + np.nanargmin(search))


# =============================================================================
# 5) Evaluation
# =============================================================================
def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {
            "N": 0,
            "MAE_pts": math.nan,
            "MedAE_pts": math.nan,
            "Std_pts": math.nan,
            "Max_Error_pts": math.nan,
            "le_10_pts_percent": math.nan,
            "le_30_pts_percent": math.nan,
            "le_50_pts_percent": math.nan,
        }
    return {
        "N": int(errors.size),
        "MAE_pts": float(np.mean(errors)),
        "MedAE_pts": float(np.median(errors)),
        "Std_pts": float(np.std(errors, ddof=0)),
        "Max_Error_pts": float(np.max(errors)),
        "le_10_pts_percent": float(np.mean(errors <= 10) * 100.0),
        "le_30_pts_percent": float(np.mean(errors <= 30) * 100.0),
        "le_50_pts_percent": float(np.mean(errors <= 50) * 100.0),
    }


def evaluate_all() -> Dict[str, float]:
    data_dir = Path(DATA_DIR)
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(LABELS_EXCEL)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for filename, gt in tqdm(labels.items(), desc="Derivative+smooth"):
        path = data_dir / filename
        if not path.exists():
            failures.append({"filename": filename, "reason": "file not found"})
            continue
        try:
            deflection, _z = load_curve_txt(path)
            pred = detect_contact_derivative_smooth(deflection)
            gt_i = int(gt)
            err = abs(pred - gt_i)
            rows.append({
                "filename": filename,
                "original_length": int(deflection.size),
                "gt_idx": gt_i,
                "pred_idx": int(pred),
                "abs_error_pts": int(err),
            })
        except Exception as e:
            failures.append({"filename": filename, "reason": repr(e)})

    errors = np.asarray([r["abs_error_pts"] for r in rows], dtype=np.float64)
    summary = summarize_errors(errors)
    summary.update({
        "method": "Derivative + smooth",
        "total_labels": int(len(labels)),
        "successful": int(len(rows)),
        "failed": int(len(failures)),
        "params": {
            "SMOOTHING_POINTS_FORCE": SMOOTHING_POINTS_FORCE,
            "SMOOTHING_POINTS_DERIVATIVE": SMOOTHING_POINTS_DERIVATIVE,
            "USE_ZEROLINE_SUBTRACTION": USE_ZEROLINE_SUBTRACTION,
            "ZEROLINE_FIT_START_PCT": ZEROLINE_FIT_START_PCT,
            "ZEROLINE_FIT_END_PCT": ZEROLINE_FIT_END_PCT,
            "MAX_LOAD_MODE": MAX_LOAD_MODE,
            "FORCE_SIGN_MODE": FORCE_SIGN_MODE,
            "DERIVATIVE_ABS_ZERO_THRESHOLD": DERIVATIVE_ABS_ZERO_THRESHOLD,
            "DERIVATIVE_ZERO_NOISE_FACTOR": DERIVATIVE_ZERO_NOISE_FACTOR,
            "CONSECUTIVE_ZERO_POINTS": CONSECUTIVE_ZERO_POINTS,
        },
    })

    if SAVE_PREDICTIONS_CSV:
        pred_csv = save_dir / "derivative_smooth_predictions.csv"
        pd.DataFrame(rows).to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if failures:
        pd.DataFrame(failures).to_csv(save_dir / "derivative_smooth_failures.csv", index=False, encoding="utf-8-sig")

    with (save_dir / "derivative_smooth_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Compact CSV summary for quick comparison.
    flat = {k: v for k, v in summary.items() if k != "params"}
    pd.DataFrame([flat]).to_csv(save_dir / "derivative_smooth_summary.csv", index=False, encoding="utf-8-sig")

    return summary


def print_summary(summary: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print(summary["method"])
    print("=" * 72)
    for key in ["total_labels", "successful", "failed", "MAE_pts", "MedAE_pts", "Std_pts", "Max_Error_pts", "le_10_pts_percent", "le_30_pts_percent", "le_50_pts_percent"]:
        print(f"{key:>22}: {summary.get(key)}")
    print(f"\nSaved to: {SAVE_DIR}")


if __name__ == "__main__":
    print_summary(evaluate_all())
