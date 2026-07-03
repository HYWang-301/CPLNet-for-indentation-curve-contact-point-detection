# -*- coding: utf-8 -*-
"""
Threshold percentage baseline for AFM contact-point detection.

流程：
1) deflection 首点归零：y = y - y[0]
2) 使用曲线前 15% 拟合线性 zero-line，并对整条曲线做基线修正
3) 以基线修正后 deflection 最大值的一定比例作为阈值
4) 在最大值之前寻找首次超过阈值的位置作为接触点
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ===================== 路径参数 =====================
DATA_DIR = r"E:\力曲线智能识别训练数据\组织+细胞"
LABELS_EXCEL = r"E:\力曲线智能识别训练数据\组织+细胞\contact_points_labels.xlsx"
SAVE_DIR = r"E:\力曲线智能识别训练数据\对比\规则驱动\threshold_percent"

# ===================== 方法参数 =====================
THRESHOLD_PERCENT_LIST = [1, 2, 3, 4, 5, 6, 7]
FIRST_POINT_ZERO_NORMALIZE = True
ZEROLINE_FIT_START_PCT = 0.0
ZEROLINE_FIT_END_PCT = 10.0
MAX_LOAD_MODE = "argmax"
MIN_SEARCH_INDEX_PCT = 10.0
MAX_FILES_FOR_DEBUG = None

# 如果想排除偶然单点噪声，可改成 3、5 等；严格按阈值首次越过时保持为 1
CONSECUTIVE_ABOVE_POINTS = 7


NUMERIC_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")


def load_curve_txt(path):
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = NUMERIC_RE.findall(line)
            if nums:
                values.extend(float(x) for x in nums)

    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 4:
        raise ValueError("too few numeric values")

    half = arr.size // 2
    deflection = arr[:half].copy()
    z_displacement = arr[half:half + half].copy()

    n = min(deflection.size, z_displacement.size)
    return deflection[:n], z_displacement[:n]


def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    if "FileName" not in df.columns or "ContactIndex" not in df.columns:
        raise ValueError("Excel must contain columns: FileName, ContactIndex")

    df = df[["FileName", "ContactIndex"]].copy()
    df = df.dropna(subset=["FileName", "ContactIndex"])
    df["FileName"] = df["FileName"].astype(str)
    df["ContactIndex"] = df["ContactIndex"].astype(int)
    return df


def resolve_curve_path(data_dir, file_name):
    data_dir = Path(data_dir)
    p = data_dir / file_name
    if p.exists():
        return p

    p = data_dir / f"{file_name}.txt"
    if p.exists():
        return p

    matches = list(data_dir.rglob(file_name))
    if matches:
        return matches[0]

    matches = list(data_dir.rglob(f"{file_name}.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(file_name)


def first_point_normalize(y):
    y = np.asarray(y, dtype=np.float64).copy()
    if FIRST_POINT_ZERO_NORMALIZE and y.size > 0:
        y = y - y[0]
    return y


def subtract_zeroline(y, start_pct, end_pct):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    start = int(round(n * start_pct / 100.0))
    end = int(round(n * end_pct / 100.0))
    start = max(0, min(start, n - 2))
    end = max(start + 2, min(end, n))

    x_fit = np.arange(start, end, dtype=np.float64)
    y_fit = y[start:end]
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)

    x_all = np.arange(n, dtype=np.float64)
    baseline = slope * x_all + intercept
    y_corr = y - baseline
    return y_corr, slope, intercept, start, end


def find_first_sustained_crossing(y, threshold, start_idx, end_idx, consecutive_points):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    start_idx = max(0, min(int(start_idx), n - 1))
    end_idx = max(start_idx, min(int(end_idx), n - 1))
    consecutive_points = max(1, int(consecutive_points))

    if consecutive_points == 1:
        idx = np.where(y[start_idx:end_idx + 1] >= threshold)[0]
        if idx.size == 0:
            return None
        return int(start_idx + idx[0])

    above = y >= threshold
    last_start = end_idx - consecutive_points + 1
    if last_start < start_idx:
        return None

    for i in range(start_idx, last_start + 1):
        if np.all(above[i:i + consecutive_points]):
            return int(i)
    return None


def detect_threshold_percent(deflection, threshold_percent):
    y0 = first_point_normalize(deflection)
    y_corr, slope, intercept, z_start, z_end = subtract_zeroline(
        y0,
        ZEROLINE_FIT_START_PCT,
        ZEROLINE_FIT_END_PCT,
    )

    if MAX_LOAD_MODE != "argmax":
        raise ValueError(f"Unsupported MAX_LOAD_MODE: {MAX_LOAD_MODE}")

    max_idx = int(np.argmax(y_corr))
    max_value = float(y_corr[max_idx])
    threshold_value = float(max_value * threshold_percent / 100.0)

    min_search_idx = int(round(len(y_corr) * MIN_SEARCH_INDEX_PCT / 100.0))
    pred_idx = find_first_sustained_crossing(
        y_corr,
        threshold_value,
        min_search_idx,
        max_idx,
        CONSECUTIVE_ABOVE_POINTS,
    )

    if pred_idx is None:
        pred_idx = max_idx

    return {
        "pred_idx": int(pred_idx),
        "max_idx": int(max_idx),
        "max_value": max_value,
        "threshold_value": threshold_value,
        "zeroline_slope": float(slope),
        "zeroline_intercept": float(intercept),
        "zeroline_start_idx": int(z_start),
        "zeroline_end_idx": int(z_end),
    }


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "n_success": int(errors.size),
        "MAE_pts": float(np.mean(errors)) if errors.size else math.nan,
        "MedAE_pts": float(np.median(errors)) if errors.size else math.nan,
        "Std_pts": float(np.std(errors)) if errors.size else math.nan,
        "Max_Error_pts": float(np.max(errors)) if errors.size else math.nan,
        "le_10_pts_percent": float(np.mean(errors <= 10) * 100.0) if errors.size else math.nan,
        "le_30_pts_percent": float(np.mean(errors <= 30) * 100.0) if errors.size else math.nan,
        "le_50_pts_percent": float(np.mean(errors <= 50) * 100.0) if errors.size else math.nan,
    }


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(LABELS_EXCEL)
    if MAX_FILES_FOR_DEBUG is not None:
        labels = labels.iloc[:MAX_FILES_FOR_DEBUG].copy()

    prediction_rows = []
    failure_rows = []
    summary_rows = []

    for threshold_percent in THRESHOLD_PERCENT_LIST:
        method_name = f"threshold_{threshold_percent:02d}pct"
        errors = []

        for row in tqdm(labels.itertuples(index=False), total=len(labels), desc=method_name):
            file_name = row.FileName
            gt_idx = int(row.ContactIndex)

            try:
                path = resolve_curve_path(DATA_DIR, file_name)
                deflection, _ = load_curve_txt(path)
                result = detect_threshold_percent(deflection, threshold_percent)
                pred_idx = int(result["pred_idx"])
                err = abs(pred_idx - gt_idx)
                errors.append(err)

                prediction_rows.append({
                    "method": "Threshold percentage",
                    "param_id": method_name,
                    "threshold_percent": threshold_percent,
                    "FileName": file_name,
                    "ContactIndex": gt_idx,
                    "PredIndex": pred_idx,
                    "AbsError_pts": err,
                    **result,
                })

            except Exception as e:
                failure_rows.append({
                    "method": "Threshold percentage",
                    "param_id": method_name,
                    "threshold_percent": threshold_percent,
                    "FileName": file_name,
                    "ContactIndex": gt_idx,
                    "error": repr(e),
                })

        s = summarize_errors(errors)
        summary_rows.append({
            "method": "Threshold percentage",
            "param_id": method_name,
            "threshold_percent": threshold_percent,
            **s,
            "n_fail": int(len(labels) - len(errors)),
        })
        print(
            f"[RESULT] {method_name}: "
            f"MAE={s['MAE_pts']:.3f}, MedAE={s['MedAE_pts']:.3f}, "
            f"<=50={s['le_50_pts_percent']:.3f}%"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["MAE_pts", "MedAE_pts"])
    pred_df = pd.DataFrame(prediction_rows)
    fail_df = pd.DataFrame(failure_rows)

    summary_df.to_csv(save_dir / "threshold_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(save_dir / "threshold_predictions_all.csv", index=False, encoding="utf-8-sig")
    fail_df.to_csv(save_dir / "threshold_failures.csv", index=False, encoding="utf-8-sig")

    config = {
        "DATA_DIR": DATA_DIR,
        "LABELS_EXCEL": LABELS_EXCEL,
        "SAVE_DIR": SAVE_DIR,
        "THRESHOLD_PERCENT_LIST": THRESHOLD_PERCENT_LIST,
        "FIRST_POINT_ZERO_NORMALIZE": FIRST_POINT_ZERO_NORMALIZE,
        "ZEROLINE_FIT_START_PCT": ZEROLINE_FIT_START_PCT,
        "ZEROLINE_FIT_END_PCT": ZEROLINE_FIT_END_PCT,
        "MAX_LOAD_MODE": MAX_LOAD_MODE,
        "MIN_SEARCH_INDEX_PCT": MIN_SEARCH_INDEX_PCT,
        "CONSECUTIVE_ABOVE_POINTS": CONSECUTIVE_ABOVE_POINTS,
        "MAX_FILES_FOR_DEBUG": MAX_FILES_FOR_DEBUG,
    }
    with open(save_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Threshold percentage baseline finished.")
    print(f"Results saved to: {save_dir}")
    print("Best parameter by MAE:")
    print(summary_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
