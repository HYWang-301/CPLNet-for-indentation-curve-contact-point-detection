# -*- coding: utf-8 -*-
"""
AFM contact point baseline: Fit intersection

This script implements the FC_analysis paper's rule-driven contact point method
"Fit intersection" and evaluates it against manually labelled ContactIndex
values in the original, un-resampled index space.

Paper description:
    Fit the zero line and the contact line using user-set percentage intervals;
    the contact point is the point closest to the intersection of the two lines.

Parameter notes from the paper:
    - The zero-line fit interval is user-defined. Figure 2/3 illustrate 10-70%;
      reported examples include 0-60%, 0-70%, 10-80%, and 10-85%.
    - The contact-line fit interval is user-defined. Figure 3 uses 90-100%.
    - Fit intersection is described as particularly suitable for hard samples.

Your txt format:
    first half  = deflection
    second half = Z_displacement
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

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
SAVE_DIR = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\rule_baselines_fc_analysis\fit_intersection"

FILENAME_COL = "FileName"
CONTACT_COL = "ContactIndex"


# =============================================================================
# 2) Method parameters: paper-style user-settable parameters
# =============================================================================
# Smooth before calculation, as requested. The paper's fit-intersection method
# itself uses linear fits; this smoothing only reduces high-frequency noise.
SMOOTHING_POINTS_FORCE = 5

# Paper Fig. 3 illustrates zero-line 10-70% and contact-line 90-100%.
ZEROLINE_FIT_START_PCT = 10.0
ZEROLINE_FIT_END_PCT = 70.0
CONTACTLINE_FIT_START_PCT = 90.0
CONTACTLINE_FIT_END_PCT = 100.0

# Force sign handling. "auto" flips the force signal if the final-load region is
# below the zero-line region, so the contact part rises upward.
FORCE_SIGN_MODE = "auto"  # "auto", "positive", or "negative"
SIGN_ESTIMATE_HEAD_PCT = 10.0
SIGN_ESTIMATE_TAIL_PCT = 10.0

# Fallback if the two fitted lines are nearly parallel or intersection is invalid.
# Options: "lowest" or "argmax".
PARALLEL_SLOPE_EPS = 1e-12
FALLBACK_ON_BAD_INTERSECTION = "lowest"

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


def orient_force(y: np.ndarray) -> np.ndarray:
    mode = FORCE_SIGN_MODE.lower().strip()
    if mode == "positive":
        return y
    if mode == "negative":
        return -y
    if mode != "auto":
        raise ValueError("FORCE_SIGN_MODE must be 'auto', 'positive', or 'negative'")
    n = y.size
    head = float(np.median(y[pct_to_slice(n, 0.0, SIGN_ESTIMATE_HEAD_PCT)]))
    tail = float(np.median(y[pct_to_slice(n, 100.0 - SIGN_ESTIMATE_TAIL_PCT, 100.0)]))
    return y if tail >= head else -y


def fit_line_by_pct(y: np.ndarray, start_pct: float, end_pct: float) -> Tuple[float, float, slice]:
    n = y.size
    sl = pct_to_slice(n, start_pct, end_pct)
    x = np.arange(n, dtype=np.float64)[sl]
    yy = y[sl]
    finite = np.isfinite(x) & np.isfinite(yy)
    x, yy = x[finite], yy[finite]
    if x.size < 2:
        raise ValueError(f"not enough finite points for fit {start_pct}-{end_pct}%")
    a, b = np.polyfit(x, yy, 1)
    return float(a), float(b), sl


def fallback_lowest(y: np.ndarray) -> int:
    # Simple discrete version used only if intersection is invalid.
    if y.size == 0:
        return 0
    return int(np.nanargmin(y))


# =============================================================================
# 4) Fit intersection CP detector
# =============================================================================
def detect_contact_fit_intersection(deflection: np.ndarray) -> int:
    y = np.asarray(deflection, dtype=np.float64)
    if y.size < 4:
        return 0

    # Preliminary smoothing, as requested.
    y_sm = moving_average_reflect(y, SMOOTHING_POINTS_FORCE)
    y_proc = orient_force(y_sm)
    n = y_proc.size

    try:
        a0, b0, _ = fit_line_by_pct(y_proc, ZEROLINE_FIT_START_PCT, ZEROLINE_FIT_END_PCT)
        a1, b1, _ = fit_line_by_pct(y_proc, CONTACTLINE_FIT_START_PCT, CONTACTLINE_FIT_END_PCT)
        denom = a0 - a1
        if abs(denom) < PARALLEL_SLOPE_EPS:
            raise ValueError("zero-line and contact-line fits are nearly parallel")
        x_intersect = (b1 - b0) / denom
        if not np.isfinite(x_intersect):
            raise ValueError("invalid line intersection")
        return int(np.clip(round(x_intersect), 0, n - 1))
    except Exception:
        if FALLBACK_ON_BAD_INTERSECTION.lower() == "argmax":
            return int(np.nanargmax(y_proc))
        return fallback_lowest(y_proc)


# =============================================================================
# 5) Evaluation
# =============================================================================
def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {"N": 0, "MAE_pts": math.nan, "MedAE_pts": math.nan, "Std_pts": math.nan,
                "Max_Error_pts": math.nan, "le_10_pts_percent": math.nan,
                "le_30_pts_percent": math.nan, "le_50_pts_percent": math.nan}
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

    for filename, gt in tqdm(labels.items(), desc="Fit intersection"):
        path = data_dir / filename
        if not path.exists():
            failures.append({"filename": filename, "reason": "file not found"})
            continue
        try:
            deflection, _z = load_curve_txt(path)
            pred = detect_contact_fit_intersection(deflection)
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
        "method": "Fit intersection",
        "total_labels": int(len(labels)),
        "successful": int(len(rows)),
        "failed": int(len(failures)),
        "params": {
            "SMOOTHING_POINTS_FORCE": SMOOTHING_POINTS_FORCE,
            "ZEROLINE_FIT_START_PCT": ZEROLINE_FIT_START_PCT,
            "ZEROLINE_FIT_END_PCT": ZEROLINE_FIT_END_PCT,
            "CONTACTLINE_FIT_START_PCT": CONTACTLINE_FIT_START_PCT,
            "CONTACTLINE_FIT_END_PCT": CONTACTLINE_FIT_END_PCT,
            "FORCE_SIGN_MODE": FORCE_SIGN_MODE,
            "FALLBACK_ON_BAD_INTERSECTION": FALLBACK_ON_BAD_INTERSECTION,
        },
    })

    if SAVE_PREDICTIONS_CSV:
        pd.DataFrame(rows).to_csv(save_dir / "fit_intersection_predictions.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(save_dir / "fit_intersection_failures.csv", index=False, encoding="utf-8-sig")
    with (save_dir / "fit_intersection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    flat = {k: v for k, v in summary.items() if k != "params"}
    pd.DataFrame([flat]).to_csv(save_dir / "fit_intersection_summary.csv", index=False, encoding="utf-8-sig")
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
