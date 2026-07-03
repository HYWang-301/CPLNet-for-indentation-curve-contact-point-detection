# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from afm_rule_common import (
    detect_derivative_smooth,
    detect_fit_intersection,
    detect_lowest_value_smooth,
    load_curve_txt,
    load_labels,
    summarize_errors,
    tqdm,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

DATA_DIR = r"E:\力曲线智能识别训练数据\组织+细胞"
LABELS_EXCEL = r"E:\力曲线智能识别训练数据\组织+细胞\contact_points_labels.xlsx"
SAVE_DIR = r"E:\力曲线智能识别训练数据\对比\规则驱动\param_grid"

FILENAME_COL = "FileName"
CONTACT_COL = "ContactIndex"

# None 表示全量。调参数时建议先设为 200 或 1000。
MAX_FILES_FOR_DEBUG = None

# focused 是默认推荐组合；broad 会加入更多参数水平，计算量更大。
SEARCH_PRESET = "focused"
METHODS_TO_RUN = ["derivative_smooth", "lowest_value_smooth", "fit_intersection"]

# 平滑窗口固定，不纳入交叉搜索。
FIXED_SMOOTH_FORCE = 51
FIXED_SMOOTH_DERIVATIVE = 51

COMMON_BASE = {
    "first_point_zero_normalize": True,
    "deflection_multiplier": 1.0,
    "force_sign_mode": "keep",
}

DERIVATIVE_BASE = {
    **COMMON_BASE,
    "smoothing_points_force": FIXED_SMOOTH_FORCE,
    "smoothing_points_derivative": FIXED_SMOOTH_DERIVATIVE,
    "max_load_mode": "argmax",
    "detection_strategy": "sustained_rise",
    "derivative_abs_zero_threshold": None,
    "min_derivative_zero_threshold": 1e-12,
    "use_zeroline_subtraction": False,
    "noise_end_frac_of_max_load": 0.35,
    "use_signal_level_gate": True,
    "signal_level_noise_factor": 2.5,
    "min_search_index_pct": 5.0,
    "max_search_index_pct": 100.0,
}

LOWEST_BASE = {
    **COMMON_BASE,
    "smoothing_points_force": FIXED_SMOOTH_FORCE,
    "max_load_mode": "argmax",
    "max_search_index_pct": 100.0,
    "lowest_detection_mode": "first_local_min_backscan",
    "local_min_eps": 1e-15,
    "min_search_index_pct": 5.0,
}

FIT_BASE = {
    **COMMON_BASE,
    "smoothing_points_force": FIXED_SMOOTH_FORCE,
    "contactline_mode": "before_max_load",
    "contactline_end_offset_pct": 0.0,
    "clip_intersection_to_approach": True,
    "parallel_slope_eps": 1e-12,
    "fallback_on_bad_intersection": "lowest_before_max",
}


def _fmt_frac(v) -> str:
    return "full" if v is None else f"q{int(round(float(v) * 100)):02d}"


def _zeroline_values(label: str) -> Dict[str, object]:
    if label == "none":
        return {"zeroline_mode": "none", "use_zeroline_subtraction": False, "zeroline_fit_start_pct": 0.0, "zeroline_fit_end_pct": 15.0}
    if label == "z0_15":
        return {"zeroline_mode": "z0_15", "use_zeroline_subtraction": True, "zeroline_fit_start_pct": 0.0, "zeroline_fit_end_pct": 15.0}
    raise ValueError(label)


def make_derivative_param_sets(preset: str) -> List[Dict[str, object]]:
    # Derivative + smooth：采用 4×4 交叉设计，围绕当前最优区域细化。
    if preset == "broad":
        noise_factors = [3.0, 3.5, 4.0, 4.5, 5.0]
        run_points = [15, 20, 25, 30, 40]
    else:
        noise_factors = [3.0, 3.5, 4.0, 4.5]
        run_points = [15, 20, 25, 30]
    rows = []
    for nf, run in itertools.product(noise_factors, run_points):
        rows.append({
            "id": f"D_nf{nf:g}_run{run}",
            "derivative_zero_noise_factor": nf,
            "consecutive_contact_slope_points": run,
        })
    return rows


def make_lowest_param_sets(preset: str) -> List[Dict[str, object]]:
    if preset == "broad":
        radii = [10, 12, 14, 16, 18, 20]
        fracs = [None, 0.30, 0.50, 0.60]
    else:
        radii = [12, 14, 16, 18]
        fracs = [None, 0.50, 0.60]
    zeroline_modes = ["none", "z0_15"]
    rows = []
    for zmode, r, q in itertools.product(zeroline_modes, radii, fracs):
        row = {
            "id": f"L_{zmode}_r{r}_{_fmt_frac(q)}",
            "local_min_radius": r,
            "min_search_frac_of_max_load": q,
        }
        row.update(_zeroline_values(zmode))
        rows.append(row)
    return rows


def make_fit_param_sets(preset: str) -> List[Dict[str, object]]:
    # Fit intersection：zero-line 只取曲线起点后的前段；contact-line 取最大 deflection 往前的短窗口。
    zero_windows = [(0.0, 10.0), (0.0, 15.0), (0.0, 20.0), (0.0, 25.0)]
    contact_widths = [1.0, 3.0, 5.0, 7.0]
    rows = []
    for (zs, ze), cw in itertools.product(zero_windows, contact_widths):
        rows.append({
            "id": f"F_z{int(zs)}_{int(ze)}_cw{cw:g}",
            "zeroline_fit_start_pct": zs,
            "zeroline_fit_end_pct": ze,
            "zero_line_window": f"{zs:g}-{ze:g}%",
            "contactline_back_width_pct": cw,
            "contact_line_window": f"max-{cw:g}% to max",
        })
    return rows


PARAM_DESIGN_ROWS = [
    {"method": "Derivative + smooth", "factor": "derivative_zero_noise_factor", "levels_focused": "3.0, 3.5, 4.0, 4.5", "levels_broad": "3.0, 3.5, 4.0, 4.5, 5.0", "fixed_or_note": "导数阈值 = 基线导数噪声 × 该因子"},
    {"method": "Derivative + smooth", "factor": "consecutive_contact_slope_points", "levels_focused": "15, 20, 25, 30", "levels_broad": "15, 20, 25, 30, 40", "fixed_or_note": "连续超过导数阈值的点数"},
    {"method": "Lowest value + smooth", "factor": "local_min_radius", "levels_focused": "12, 14, 16, 18", "levels_broad": "10, 12, 14, 16, 18, 20", "fixed_or_note": "局部最低点判定半径"},
    {"method": "Lowest value + smooth", "factor": "min_search_frac_of_max_load", "levels_focused": "None, 0.50, 0.60", "levels_broad": "None, 0.30, 0.50, 0.60", "fixed_or_note": "从最大载荷点的指定比例位置开始搜索"},
    {"method": "Lowest value + smooth", "factor": "zeroline_mode", "levels_focused": "none, z0_15", "levels_broad": "none, z0_15", "fixed_or_note": "z0_15 表示用曲线 0–15% 拟合并扣除 zero-line"},
    {"method": "Fit intersection", "factor": "zero_line_window", "levels_focused": "0–10%, 0–15%, 0–20%, 0–25%", "levels_broad": "同 focused", "fixed_or_note": "zero-line 从曲线起点往后取，不超过前 25%"},
    {"method": "Fit intersection", "factor": "contactline_back_width_pct", "levels_focused": "1%, 3%, 5%, 7%", "levels_broad": "同 focused", "fixed_or_note": "从 deflection 最大值往前取指定宽度拟合 contact-line"},
]


def method_specs() -> Dict[str, Tuple[str, object, Dict[str, object], List[Dict[str, object]]]]:
    return {
        "derivative_smooth": ("Derivative + smooth", detect_derivative_smooth, DERIVATIVE_BASE, make_derivative_param_sets(SEARCH_PRESET)),
        "lowest_value_smooth": ("Lowest value + smooth", detect_lowest_value_smooth, LOWEST_BASE, make_lowest_param_sets(SEARCH_PRESET)),
        "fit_intersection": ("Fit intersection", detect_fit_intersection, FIT_BASE, make_fit_param_sets(SEARCH_PRESET)),
    }


def evaluate_combo(method_name: str, detect_fn, params: Dict[str, object], labels: Dict[str, int], data_dir: Path) -> Dict[str, object]:
    errors: List[int] = []
    failed = 0
    for filename, gt in tqdm(labels.items(), desc=str(params["combo_id"]), leave=False):
        try:
            deflection, _ = load_curve_txt(data_dir / filename)
            pred = int(detect_fn(deflection, params))
            errors.append(abs(pred - int(gt)))
        except Exception:
            failed += 1
    summary = summarize_errors(np.asarray(errors, dtype=np.float64))
    summary.update({
        "method": method_name,
        "combo_id": str(params["combo_id"]),
        "total_labels": int(len(labels)),
        "successful": int(len(errors)),
        "failed": int(failed),
    })
    return summary


def paper_row(summary: Dict[str, object], params: Dict[str, object]) -> Dict[str, object]:
    method = str(summary["method"])
    row = {
        "Method": method,
        "Parameter set": summary["combo_id"],
        "N": summary["N"],
        "MAE (pts)": summary["MAE_pts"],
        "MedAE (pts)": summary["MedAE_pts"],
        "Std (pts)": summary["Std_pts"],
        "Max Error (pts)": summary["Max_Error_pts"],
        "≤10 pts (%)": summary["le_10_pts_percent"],
        "≤30 pts (%)": summary["le_30_pts_percent"],
        "≤50 pts (%)": summary["le_50_pts_percent"],
    }
    if method == "Derivative + smooth":
        row.update({
            "Factor A": "derivative_zero_noise_factor",
            "A value": params.get("derivative_zero_noise_factor"),
            "Factor B": "consecutive_contact_slope_points",
            "B value": params.get("consecutive_contact_slope_points"),
            "Factor C": "-",
            "C value": "-",
        })
    elif method == "Lowest value + smooth":
        row.update({
            "Factor A": "local_min_radius",
            "A value": params.get("local_min_radius"),
            "Factor B": "min_search_frac_of_max_load",
            "B value": "full" if params.get("min_search_frac_of_max_load") is None else params.get("min_search_frac_of_max_load"),
            "Factor C": "zeroline_mode",
            "C value": params.get("zeroline_mode", "none"),
        })
    else:
        row.update({
            "Factor A": "zero_line_window",
            "A value": params.get("zero_line_window"),
            "Factor B": "contactline_back_width_pct",
            "B value": params.get("contactline_back_width_pct"),
            "Factor C": "-",
            "C value": "-",
        })
    return row


def safe_label(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "full"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def save_heatmap_matrix(df: pd.DataFrame, method: str, row_key: str, col_key: str, metric: str, out_dir: Path, suffix: str = "") -> None:
    if df.empty or row_key not in df.columns or col_key not in df.columns:
        return
    d = df.copy()
    d[row_key] = d[row_key].map(safe_label)
    d[col_key] = d[col_key].map(safe_label)
    pivot = d.pivot_table(index=row_key, columns=col_key, values=metric, aggfunc="mean")
    name = f"heatmap_{method}{suffix}_{metric}"
    pivot.to_csv(out_dir / f"{name}.csv", encoding="utf-8-sig")
    if not HAS_MPL:
        return
    fig_w = max(6, 0.9 * len(pivot.columns) + 2)
    fig_h = max(4, 0.55 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(pivot.values.astype(float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    ax.set_title(f"{method}{suffix}: {metric}")
    fig.colorbar(im, ax=ax, shrink=0.85)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def save_all_heatmaps(raw_df: pd.DataFrame, save_dir: Path) -> None:
    out_dir = save_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 为论文统计中的全部核心指标生成热图矩阵。
    metrics = [
        "MAE_pts",
        "MedAE_pts",
        "Std_pts",
        "Max_Error_pts",
        "le_10_pts_percent",
        "le_30_pts_percent",
        "le_50_pts_percent",
    ]

    ddf = raw_df[raw_df["method"] == "Derivative + smooth"].copy()
    for metric in metrics:
        save_heatmap_matrix(ddf, "derivative_smooth", "derivative_zero_noise_factor", "consecutive_contact_slope_points", metric, out_dir)

    ldf = raw_df[raw_df["method"] == "Lowest value + smooth"].copy()
    for zmode in sorted(ldf["zeroline_mode"].dropna().unique()):
        sub = ldf[ldf["zeroline_mode"] == zmode]
        for metric in metrics:
            save_heatmap_matrix(sub, "lowest_value_smooth", "local_min_radius", "min_search_frac_of_max_load", metric, out_dir, suffix=f"_{zmode}")

    fdf = raw_df[raw_df["method"] == "Fit intersection"].copy()
    for metric in metrics:
        save_heatmap_matrix(fdf, "fit_intersection", "zero_line_window", "contactline_back_width_pct", metric, out_dir)


def main() -> None:
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(DATA_DIR)
    labels = load_labels(LABELS_EXCEL, FILENAME_COL, CONTACT_COL)
    if MAX_FILES_FOR_DEBUG is not None:
        labels = dict(list(labels.items())[:int(MAX_FILES_FOR_DEBUG)])

    specs = method_specs()
    all_raw: List[Dict[str, object]] = []
    all_paper: List[Dict[str, object]] = []
    t0 = time.time()

    pd.DataFrame(PARAM_DESIGN_ROWS).to_csv(save_dir / "table_parameter_design.csv", index=False, encoding="utf-8-sig")

    for method_key in METHODS_TO_RUN:
        method_name, detect_fn, base, param_sets = specs[method_key]
        method_raw: List[Dict[str, object]] = []
        method_paper: List[Dict[str, object]] = []
        for overrides in param_sets:
            combo_id = str(overrides["id"])
            params = dict(base)
            params.update({k: v for k, v in overrides.items() if k != "id"})
            params["combo_id"] = combo_id
            print(f"\n[RUN] {method_name} | {combo_id}")
            summary = evaluate_combo(method_name, detect_fn, params, labels, data_dir)
            raw_row = {**summary, **params}
            paper = paper_row(summary, params)
            method_raw.append(raw_row)
            method_paper.append(paper)
            all_raw.append(raw_row)
            all_paper.append(paper)
            print(f"  MAE={summary['MAE_pts']:.3f}, MedAE={summary['MedAE_pts']:.3f}, <=50={summary['le_50_pts_percent']:.3f}%")

        raw_df = pd.DataFrame(method_raw).sort_values("MAE_pts", ascending=True)
        paper_df = pd.DataFrame(method_paper).sort_values("MAE (pts)", ascending=True)
        raw_df.to_csv(save_dir / f"grid_summary_{method_key}_raw.csv", index=False, encoding="utf-8-sig")
        paper_df.to_csv(save_dir / f"table_results_{method_key}.csv", index=False, encoding="utf-8-sig")

    raw_all = pd.DataFrame(all_raw).sort_values(["method", "MAE_pts"], ascending=[True, True])
    paper_all = pd.DataFrame(all_paper).sort_values(["Method", "MAE (pts)"], ascending=[True, True])
    raw_all.to_csv(save_dir / "grid_summary_all_raw.csv", index=False, encoding="utf-8-sig")
    paper_all.to_csv(save_dir / "table_results_all.csv", index=False, encoding="utf-8-sig")

    best = paper_all.sort_values("MAE (pts)").groupby("Method", as_index=False).first()
    best.to_csv(save_dir / "table_best_by_method.csv", index=False, encoding="utf-8-sig")

    save_all_heatmaps(raw_all, save_dir)

    manifest = {
        "search_preset": SEARCH_PRESET,
        "methods_to_run": METHODS_TO_RUN,
        "max_files_for_debug": MAX_FILES_FOR_DEBUG,
        "n_labels": len(labels),
        "n_combinations": len(all_raw),
        "fixed_smoothing_points_force": FIXED_SMOOTH_FORCE,
        "fixed_smoothing_points_derivative": FIXED_SMOOTH_DERIVATIVE,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    with (save_dir / "grid_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(all_raw)} 组参数完成，用时 {(time.time() - t0) / 60:.1f} min")
    print(f"结果保存到: {save_dir}")


if __name__ == "__main__":
    main()
