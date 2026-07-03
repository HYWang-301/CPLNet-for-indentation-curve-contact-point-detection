import os
import sys
import csv
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

DATA_DIR     = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\组织+细胞"
LABELS_EXCEL = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\组织+细胞\contact_points_labels改.xlsx"
SAVE_DIR     = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\AAA20260527\v11 ablation_v15\双通道且不要T及log概率转换基础上且只有2项Loss（KL和Ord）2"

CONFIG = {
    'epochs': 200,
    'num_epochs':       200,
    'batch_size':       32,
    'learning_rate':    0.0002,
    'weight_decay':     0.001,
    'patience':         30,
    'seed':             42,

    'target_length':     8192,
    'num_workers':       4,
    'drop_path':         0.15,
    'dropout':           0.18,
    'boundary_window':   64,
    'softargmax_window': 100,
    'use_transformer':   True,
    'use_boundary':      True,
    'use_multiscale':    True,

    'use_amp':           True,
    'fast_mode':         False,

    'lr_warmup_epochs':  5,

    'ema_decay':         0.999,
    'grad_accum_steps':  1,

    'use_tta':           True,

    'group_aware_split':      True,
    'use_stratified_sampler': True,
    'stratified_bins':        20,
    'cutout_prob':            0.6,
    'resample_mode':          'poly',

    'sharpness_radius':   80,
    'sharpness_weight':   0.5,
    'ordinal_n_neg':      20,
    'ordinal_margin':     4.0,
    'ordinal_neg_thresh': 0.1,
    'heatmap_weight':     1.0,
    'ordinal_weight':     0.15,
    'regression_weight':  0.0,

    'sigma_max':             80,
    'sigma_min':             10,
    'sigma_warmup':           2,
    'sigma_schedule_epochs': 70,
    'heatmap_type':          'laplacian',

    'num_visualize': 0,
}

ORIG_THRESHOLDS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]


def print_orig_cumulative(errors_orig, save_dir=None):
    header = "原始空间累计精度 (Original Space — lower threshold is stricter)"
    sep    = "=" * 58
    print(f"\n{sep}")
    print(header)
    print(f"{'Threshold':<16} {'# Samples':>10} {'Cumulative %':>14}")
    print("-" * 58)

    rows = []
    n    = len(errors_orig)
    for t in ORIG_THRESHOLDS:
        cnt = int((errors_orig <= t).sum())
        pct = cnt / n * 100.0 if n > 0 else 0.0
        print(f"  <= {t:4d} pts      {cnt:>10d}    {pct:>12.2f}%")
        rows.append({
            'threshold_pts':    t,
            'n_within':         cnt,
            'total_samples':    n,
            'cumulative_pct':   round(pct, 4),
        })
    print(sep)
    print(f"  Total samples evaluated: {n}")
    print(sep)

    if save_dir:
        csv_path = os.path.join(save_dir, 'orig_cumulative_accuracy.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(
                f, fieldnames=['threshold_pts', 'n_within',
                               'total_samples', 'cumulative_pct'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Saved] {csv_path}")
        _plot_cumulative(rows, save_dir)

    return rows


def _plot_cumulative(rows, save_dir):
    thresholds = [r['threshold_pts']  for r in rows]
    pcts       = [r['cumulative_pct'] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, pcts, 'o-', color='#2c7fb8', lw=2, ms=6)
    for t, p in zip(thresholds, pcts):
        ax.annotate(f'{p:.1f}%', (t, p),
                    textcoords='offset points', xytext=(0, 7),
                    fontsize=8, ha='center', color='#2c7fb8')
    ax.set_xscale('log')
    ax.set_xticks(thresholds)
    ax.set_xticklabels([str(t) for t in thresholds], rotation=45, fontsize=8)
    ax.set_xlabel('Error Threshold (pts, original space)')
    ax.set_ylabel('Cumulative Accuracy (%)')
    ax.set_title('AFM V15 Baseline — Original Space Cumulative Accuracy')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.35)
    ax.axhline(90, color='gray', ls='--', alpha=0.5, label='90%')
    ax.axhline(95, color='gray', ls=':',  alpha=0.5, label='95%')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'orig_cumulative_accuracy.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {fig_path}")


def save_train_summary(save_dir, train_minutes):
    best_path = os.path.join(save_dir, 'best_mae.pth')
    summary   = {'train_minutes': train_minutes}

    if os.path.exists(best_path):
        try:
            ck = torch.load(best_path, map_location='cpu', weights_only=False)
        except TypeError:
            ck = torch.load(best_path, map_location='cpu')

        summary.update({
            'best_epoch':              ck.get('epoch'),
            'val_mae_orig':            ck.get('val_mae_orig'),
            'val_mae_soft_orig':       ck.get('val_mae_soft_orig'),
            'val_mae_soft_orig_ema':   ck.get('val_mae_soft_orig_ema'),
            'val_mae_soft_rs':         ck.get('val_mae_soft_rs'),
            'val_mae_soft_ema_rs':     ck.get('val_mae_soft_ema_rs'),
            'ema_is_better':           ck.get('ema_is_better'),
        })

        print("\n── 训练摘要 ─────────────────────────────────────────")
        for k, v in summary.items():
            print(f"  {k:<30}: {v}")
        print("─────────────────────────────────────────────────────")

    out_path = os.path.join(save_dir, 'baseline_train_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Saved] {out_path}")
    return summary


def run_baseline():
    from run_v15      import train_model, evaluate_model
    from evaluate_v15 import Evaluator

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  AFM V15 ①+③ — 训练开始 (HeatmapLoss + OrdinalRanking)")
    print("=" * 60)
    print(f"  DATA_DIR      : {DATA_DIR}")
    print(f"  LABELS_EXCEL  : {LABELS_EXCEL}")
    print(f"  SAVE_DIR      : {SAVE_DIR}")
    print(f"  target_length : {CONFIG['target_length']}")
    print(f"  epochs        : {CONFIG['epochs']}  (patience={CONFIG['patience']})")
    print(f"  batch_size    : {CONFIG['batch_size']}")
    print(f"  fast_mode     : {CONFIG['fast_mode']}")
    print(f"  use_tta       : {CONFIG['use_tta']}")
    print(f"  ema_decay     : {CONFIG['ema_decay']}")
    print(f"  channels      : 2 (deflection + d(deflection)/dx)")
    print("=" * 60 + "\n")

    t0 = time.time()

    model, test_files, _ = train_model(
        data_dir=DATA_DIR,
        labels_excel=LABELS_EXCEL,
        save_dir=SAVE_DIR,
        config=CONFIG,
    )
    train_minutes = round((time.time() - t0) / 60, 1)
    print(f"\n训练完成，耗时 {train_minutes} 分钟")

    summary = save_train_summary(SAVE_DIR, train_minutes)

    best_path   = os.path.join(SAVE_DIR, 'best_mae.pth')
    results_dir = os.path.join(SAVE_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  测试集评估（含 TTA）")
    print("=" * 60)

    _, eval_results, _ = evaluate_model(
        model_path=best_path,
        data_dir=DATA_DIR,
        labels_excel=LABELS_EXCEL,
        test_files=test_files,
        save_dir=SAVE_DIR,
        num_visualize=CONFIG['num_visualize'],
    )

    errors_orig = np.array([
        abs(r['pred_idx'] - r['gt_idx'])
        for r in eval_results
        if r.get('gt_idx') is not None and r.get('pred_idx') is not None
    ])

    cumulative_rows = print_orig_cumulative(errors_orig, save_dir=results_dir)

    final_summary = {
        'config':               CONFIG,
        'train_summary':        summary,
        'orig_cumulative_pct': {
            f'<={r["threshold_pts"]}pts': r['cumulative_pct']
            for r in cumulative_rows
        },
        'orig_mae':    float(errors_orig.mean())             if len(errors_orig) else None,
        'orig_median': float(np.median(errors_orig))         if len(errors_orig) else None,
        'orig_std':    float(errors_orig.std())              if len(errors_orig) else None,
        'orig_p90':    float(np.percentile(errors_orig, 90)) if len(errors_orig) else None,
        'orig_p95':    float(np.percentile(errors_orig, 95)) if len(errors_orig) else None,
    }
    final_path = os.path.join(SAVE_DIR, 'baseline_final_results.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Saved] {final_path}")

    total_minutes = round((time.time() - t0) / 60, 1)
    print("\n" + "=" * 60)
    print(f"  全流程完成，总耗时 {total_minutes} 分钟")
    print(f"  结果根目录 : {SAVE_DIR}")
    print(f"  评估结果   : {results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    missing = []
    if not os.path.exists(DATA_DIR):
        missing.append(f"DATA_DIR     : {DATA_DIR}")
    if not os.path.exists(LABELS_EXCEL):
        missing.append(f"LABELS_EXCEL : {LABELS_EXCEL}")
    if missing:
        print("\n[错误] 以下路径不存在，请修改脚本顶部的路径配置：")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    run_baseline()
