# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from afm_rule_common import (
    detect_lowest_value_smooth,
    evaluate_method,
    prepare_lowest_signal,
    print_summary,
)

DATA_DIR = r"E:\力曲线智能识别训练数据\组织+细胞"
LABELS_EXCEL = r"E:\力曲线智能识别训练数据\组织+细胞\contact_points_labels.xlsx"
SAVE_DIR = r"E:\力曲线智能识别训练数据\对比\规则驱动\lowest_value_smooth"

FILENAME_COL = "FileName"
CONTACT_COL = "ContactIndex"

# 平滑窗口固定；文章中该方法需要先平滑，本文数据点数较长，建议用 51。
SMOOTHING_POINTS_FORCE = 51

# deflection 预处理：首点归零，不反向。
FIRST_POINT_ZERO_NORMALIZE = True
DEFLECTION_MULTIPLIER = 1.0
FORCE_SIGN_MODE = "keep"

# 最高载荷点：你的曲线最高载荷在中间，因此用 argmax，不用 end。
MAX_LOAD_MODE = "argmax"

# Lowest value + smooth 核心参数。
LOWEST_DETECTION_MODE = "first_local_min_backscan"  # 可选：first_local_min_backscan / global_min_in_window
LOCAL_MIN_RADIUS = 12
MIN_SEARCH_INDEX_PCT = 2.0
MAX_SEARCH_INDEX_PCT = 100.0
MIN_SEARCH_FRAC_OF_MAX_LOAD = 0.50  # 限制搜索左边界，避免前段长基线噪声被选中；可设 None 关闭。

# zero-line 扣除：Lowest 方法本身不需要 contact-line；该项只用于校正前段基线漂移。
USE_ZEROLINE_SUBTRACTION = False
ZEROLINE_FIT_START_PCT = 0.0
ZEROLINE_FIT_END_PCT = 60.0

NUM_VISUALIZE = 30
VISUALIZE_SELECTION = "largest_errors"
RANDOM_SEED = 42


def build_params() -> Dict[str, object]:
    return {
        "first_point_zero_normalize": FIRST_POINT_ZERO_NORMALIZE,
        "deflection_multiplier": DEFLECTION_MULTIPLIER,
        "force_sign_mode": FORCE_SIGN_MODE,
        "smoothing_points_force": SMOOTHING_POINTS_FORCE,
        "max_load_mode": MAX_LOAD_MODE,
        "lowest_detection_mode": LOWEST_DETECTION_MODE,
        "local_min_radius": LOCAL_MIN_RADIUS,
        "min_search_index_pct": MIN_SEARCH_INDEX_PCT,
        "max_search_index_pct": MAX_SEARCH_INDEX_PCT,
        "min_search_frac_of_max_load": MIN_SEARCH_FRAC_OF_MAX_LOAD,
        "use_zeroline_subtraction": USE_ZEROLINE_SUBTRACTION,
        "zeroline_fit_start_pct": ZEROLINE_FIT_START_PCT,
        "zeroline_fit_end_pct": ZEROLINE_FIT_END_PCT,
    }


def main() -> None:
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    params = build_params()
    summary = evaluate_method(
        method_name="Lowest value + smooth",
        detect_fn=detect_lowest_value_smooth,
        prepare_signal_fn=prepare_lowest_signal,
        params=params,
        data_dir=DATA_DIR,
        labels_excel=LABELS_EXCEL,
        save_dir=SAVE_DIR,
        filename_col=FILENAME_COL,
        contact_col=CONTACT_COL,
        visualize_n=NUM_VISUALIZE,
        visualize_selection=VISUALIZE_SELECTION,
        visualize_random_seed=RANDOM_SEED,
        progress_desc="Lowest value + smooth",
    )
    with (save_dir / "lowest_value_smooth_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print_summary("Lowest value + smooth", summary, SAVE_DIR)


if __name__ == "__main__":
    main()
