# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def load_labels(excel_path: str, filename_col: str = "FileName", contact_col: str = "ContactIndex") -> Dict[str, int]:
    df = pd.read_excel(excel_path)
    if filename_col not in df.columns or contact_col not in df.columns:
        raise ValueError(f"标签表必须包含列: {filename_col!r}, {contact_col!r}; 当前列: {list(df.columns)}")
    return {str(row[filename_col]): int(row[contact_col]) for _, row in df.iterrows()}


def load_curve_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vals: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                pass
    data = np.asarray(vals, dtype=np.float64)
    if data.size < 4:
        raise ValueError(f"too few numeric values: {data.size}")
    mid = data.size // 2
    d, z = data[:mid], data[mid:]
    n = min(d.size, z.size)
    if n < 4:
        raise ValueError(f"too few paired points: {n}")
    return d[:n], z[:n]


def preprocess_deflection(deflection: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    y = np.asarray(deflection, dtype=np.float64).copy()
    if y.size == 0:
        return y
    if bool(p.get("first_point_zero_normalize", True)):
        y = y - y[0]
    y = y * float(p.get("deflection_multiplier", 1.0))
    return y


def moving_average_reflect(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, round(window)))
    if window <= 1 or y.size <= 2:
        return y.astype(np.float64, copy=True)
    window = min(window, y.size)
    if window % 2 == 0:
        window = window + 1 if window + 1 <= y.size else window - 1
    if window <= 1:
        return y.astype(np.float64, copy=True)
    pad = window // 2
    yy = np.pad(y.astype(np.float64), pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(yy, kernel, mode="valid")


def pct_to_slice(n: int, start_pct: float, end_pct: float) -> slice:
    start_pct = float(np.clip(start_pct, 0, 100))
    end_pct = float(np.clip(end_pct, 0, 100))
    lo = int(np.floor(start_pct / 100.0 * (n - 1)))
    hi = int(np.ceil(end_pct / 100.0 * (n - 1))) + 1
    lo = max(0, min(lo, n - 1))
    hi = max(1, min(hi, n))
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
    return float(1.4826 * mad) if mad > 0 else float(np.std(x))


def fit_line_by_slice(y: np.ndarray, sl: slice) -> Tuple[float, float]:
    n = y.size
    x = np.arange(n, dtype=np.float64)[sl]
    yy = y[sl]
    if x.size < 2:
        raise ValueError("fit interval too short")
    a, b = np.polyfit(x, yy, 1)
    return float(a), float(b)


def linear_baseline_subtract(y: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    if not bool(p.get("use_zeroline_subtraction", False)):
        return y.astype(np.float64, copy=True)
    sl = pct_to_slice(y.size, float(p.get("zeroline_fit_start_pct", 0.0)), float(p.get("zeroline_fit_end_pct", 15.0)))
    a, b = fit_line_by_slice(y, sl)
    x = np.arange(y.size, dtype=np.float64)
    return y - (a * x + b)


def orient_force(y: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    mode = str(p.get("force_sign_mode", "keep")).lower().strip()
    if mode in ("keep", "none", "positive"):
        return y
    if mode == "negative":
        return -y
    if mode != "auto":
        raise ValueError("force_sign_mode must be keep/negative/auto")
    n = y.size
    head = y[pct_to_slice(n, 0.0, float(p.get("sign_estimate_head_pct", 10.0)))]
    tail = y[pct_to_slice(n, 100.0 - float(p.get("sign_estimate_tail_pct", 10.0)), 100.0)]
    return y if float(np.median(tail)) >= float(np.median(head)) else -y


def max_load_index(y: np.ndarray, p: Dict[str, object]) -> int:
    mode = str(p.get("max_load_mode", "argmax")).lower().strip()
    if mode == "end":
        return int(y.size - 1)
    if mode == "argmax":
        return int(np.nanargmax(y))
    raise ValueError("max_load_mode must be end/argmax")


def _baseline_slice_for_threshold(n: int, start_idx: int, p: Dict[str, object]) -> slice:
    start_idx = int(np.clip(start_idx, 2, n - 1))
    hi = int(round(start_idx * float(p.get("noise_end_frac_of_max_load", 0.35))))
    hi = int(np.clip(hi, 3, max(3, start_idx - 1)))
    return slice(0, hi)


def _estimate_derivative_threshold(dy_sm: np.ndarray, n: int, start_idx: Optional[int], p: Dict[str, object]) -> float:
    manual = p.get("derivative_abs_zero_threshold", None)
    if manual is not None:
        return float(manual)
    if start_idx is None:
        sl = pct_to_slice(n, float(p.get("zeroline_fit_start_pct", 0.0)), float(p.get("zeroline_fit_end_pct", 15.0)))
    else:
        sl = _baseline_slice_for_threshold(n, start_idx, p)
    thr = float(p.get("derivative_zero_noise_factor", 4.0)) * robust_std(dy_sm[sl])
    return max(thr, float(p.get("min_derivative_zero_threshold", 1e-12)))


def _estimate_contact_slope_sign(dy_sm: np.ndarray, lo: int, hi: int, start_idx: int) -> float:
    a = max(lo, int(0.10 * start_idx))
    b = min(dy_sm.size, start_idx + 1)
    seg = dy_sm[a:b]
    if seg.size and np.isfinite(seg).any():
        val = float(seg[int(np.nanargmax(np.abs(seg)))])
        if abs(val) > 1e-15:
            return 1.0 if val > 0 else -1.0
    return 1.0


def _first_sustained_true(mask: np.ndarray, run: int) -> Optional[int]:
    run = max(1, int(run))
    cnt = 0
    for i, ok in enumerate(mask):
        if bool(ok):
            cnt += 1
            if cnt >= run:
                return i - run + 1
        else:
            cnt = 0
    return None


def prepare_derivative_signal(deflection: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    y = preprocess_deflection(deflection, p)
    y = moving_average_reflect(y, int(p.get("smoothing_points_force", 51)))
    y = linear_baseline_subtract(y, p)
    return orient_force(y, p)


def detect_derivative_smooth(deflection: np.ndarray, p: Dict[str, object]) -> int:
    y = prepare_derivative_signal(deflection, p)
    if y.size < 4:
        return 0
    n = y.size
    dy = moving_average_reflect(np.gradient(y), int(p.get("smoothing_points_derivative", 51)))
    start_idx = max_load_index(y, p)
    lo = int(np.floor(np.clip(float(p.get("min_search_index_pct", 5.0)), 0, 100) / 100.0 * (n - 1)))
    hi = int(np.ceil(np.clip(float(p.get("max_search_index_pct", 100.0)), 0, 100) / 100.0 * (n - 1)))
    start_idx = int(np.clip(start_idx, lo + 1, hi))

    strategy = str(p.get("detection_strategy", "sustained_rise")).lower().strip()
    thr = _estimate_derivative_threshold(dy, n, start_idx, p)
    slope_sign = _estimate_contact_slope_sign(dy, lo, hi, start_idx)
    signed_dy = slope_sign * dy

    if strategy == "paper_zero_backscan":
        run = max(1, int(p.get("consecutive_zero_points", 5)))
        entered = False
        for i in range(start_idx, lo - 1, -1):
            val = signed_dy[i]
            if np.isfinite(val) and val > thr:
                entered = True
                continue
            if entered:
                left = max(lo, i - run + 1)
                seg = signed_dy[left:i + 1]
                if seg.size >= run and np.all(np.isfinite(seg)) and np.all(seg <= thr):
                    return int(np.clip(i + run, 0, n - 1))

    search_lo = int(np.clip(lo, 0, n - 1))
    search_hi = int(np.clip(start_idx, search_lo + 1, min(hi, n - 1)))
    mask = signed_dy[search_lo:search_hi + 1] > thr
    if bool(p.get("use_signal_level_gate", True)):
        base_sl = _baseline_slice_for_threshold(n, start_idx, p)
        base_y = y[base_sl]
        base_med = float(np.median(base_y)) if base_y.size else 0.0
        level = slope_sign * (y[search_lo:search_hi + 1] - base_med)
        level_thr = float(p.get("signal_level_noise_factor", 2.5)) * max(robust_std(base_y), 1e-12)
        mask = mask & (level > level_thr)
    local = _first_sustained_true(mask, int(p.get("consecutive_contact_slope_points", 25)))
    if local is not None:
        return int(search_lo + local)

    seg = signed_dy[search_lo:search_hi + 1]
    if seg.size == 0 or not np.isfinite(seg).any():
        return int(start_idx)
    return int(search_lo + np.nanargmax(seg))


def prepare_lowest_signal(deflection: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    y = preprocess_deflection(deflection, p)
    y = moving_average_reflect(y, int(p.get("smoothing_points_force", 51)))
    y = linear_baseline_subtract(y, p)
    return orient_force(y, p)


def _lowest_search_bounds(n: int, start_idx: int, p: Dict[str, object]) -> Tuple[int, int]:
    lo_pct = int(np.floor(np.clip(float(p.get("min_search_index_pct", 0.0)), 0, 100) / 100.0 * (n - 1)))
    frac = p.get("min_search_frac_of_max_load", None)
    if frac is None or (isinstance(frac, float) and np.isnan(frac)):
        lo_frac = 0
    else:
        lo_frac = int(np.floor(float(frac) * start_idx))
    lo = int(np.clip(max(lo_pct, lo_frac), 0, n - 2))
    hi = int(np.ceil(np.clip(float(p.get("max_search_index_pct", 100.0)), 0, 100) / 100.0 * (n - 1)))
    hi = int(np.clip(min(start_idx, hi), lo + 1, n - 1))
    return lo, hi


def detect_lowest_value_smooth(deflection: np.ndarray, p: Dict[str, object]) -> int:
    y = prepare_lowest_signal(deflection, p)
    if y.size < 4:
        return 0
    n = y.size
    start_idx = max_load_index(y, p)
    lo, hi = _lowest_search_bounds(n, start_idx, p)
    r = max(1, int(p.get("local_min_radius", 12)))
    eps = float(p.get("local_min_eps", 1e-15))
    mode = str(p.get("lowest_detection_mode", "first_local_min_backscan")).lower().strip()

    if mode == "global_min_in_window":
        search = y[lo:hi + 1]
        if search.size == 0 or not np.isfinite(search).any():
            return int(hi)
        return int(lo + np.nanargmin(search))

    for i in range(hi - r, lo + r - 1, -1):
        left = max(lo, i - r)
        right = min(hi + 1, i + r + 1)
        win = y[left:right]
        if win.size < 2 * r + 1:
            continue
        center = y[i]
        if np.isfinite(center) and center <= np.nanmin(win) + eps:
            return int(np.clip(i, 0, n - 1))

    search = y[lo:hi + 1]
    if search.size == 0 or not np.isfinite(search).any():
        return int(hi)
    return int(lo + np.nanargmin(search))


def prepare_fit_signal(deflection: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    y = preprocess_deflection(deflection, p)
    y = moving_average_reflect(y, int(p.get("smoothing_points_force", 51)))
    return orient_force(y, p)


def _contactline_slice_before_max(y: np.ndarray, p: Dict[str, object]) -> slice:
    n = y.size
    max_idx = int(np.nanargmax(y))
    width = max(2, int(round(float(p.get("contactline_back_width_pct", 3.0)) / 100.0 * (n - 1))))
    end_offset = max(0, int(round(float(p.get("contactline_end_offset_pct", 0.0)) / 100.0 * (n - 1))))
    hi = int(np.clip(max_idx - end_offset + 1, 2, n))
    lo = int(np.clip(hi - width, 0, hi - 2))
    return slice(lo, hi)


def _contactline_slice_by_pct(y: np.ndarray, p: Dict[str, object]) -> slice:
    return pct_to_slice(y.size, float(p.get("contactline_fit_start_pct", 90.0)), float(p.get("contactline_fit_end_pct", 100.0)))


def detect_fit_intersection(deflection: np.ndarray, p: Dict[str, object]) -> int:
    y = prepare_fit_signal(deflection, p)
    if y.size < 4:
        return 0
    n = y.size
    try:
        z_sl = pct_to_slice(n, float(p.get("zeroline_fit_start_pct", 0.0)), float(p.get("zeroline_fit_end_pct", 15.0)))
        mode = str(p.get("contactline_mode", "before_max_load")).lower().strip()
        c_sl = _contactline_slice_before_max(y, p) if mode == "before_max_load" else _contactline_slice_by_pct(y, p)
        a0, b0 = fit_line_by_slice(y, z_sl)
        a1, b1 = fit_line_by_slice(y, c_sl)
        denom = a0 - a1
        if abs(denom) < float(p.get("parallel_slope_eps", 1e-12)):
            raise ValueError("parallel fits")
        x_intersect = (b1 - b0) / denom
        if not np.isfinite(x_intersect):
            raise ValueError("invalid intersection")
        max_idx = int(np.nanargmax(y))
        if bool(p.get("clip_intersection_to_approach", True)):
            x_intersect = np.clip(x_intersect, 0, max_idx)
        return int(np.clip(round(x_intersect), 0, n - 1))
    except Exception:
        fallback = str(p.get("fallback_on_bad_intersection", "lowest_before_max")).lower().strip()
        if fallback == "argmax":
            return int(np.nanargmax(y))
        max_idx = int(np.nanargmax(y))
        search = y[:max_idx + 1]
        if search.size == 0 or not np.isfinite(search).any():
            return max_idx
        return int(np.nanargmin(search))


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {
            "N": 0, "MAE_pts": math.nan, "MedAE_pts": math.nan, "Std_pts": math.nan,
            "Max_Error_pts": math.nan, "le_10_pts_percent": math.nan,
            "le_30_pts_percent": math.nan, "le_50_pts_percent": math.nan,
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
