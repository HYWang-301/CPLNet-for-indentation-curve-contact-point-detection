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
LABELS_EXCEL = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\组织+细胞\contact_points_labels.xlsx"
BASE_SAVE_DIR = r"C:\Users\admin\Desktop\下游数据智能化分析\力曲线智能识别训练数据\AAA20260527\v11 ablation_v15\最终版multiseed 双通道且不要T及log概率转换基础上且只有2项Loss（KL和Ord）"

SEEDS = [42, 99, 123, 314, 2025, 2026, 7777]

BASE_CONFIG = {
    'epochs':             200,
    'num_epochs':         200,
    'batch_size':         32,
    'learning_rate':      0.0002,
    'weight_decay':       0.001,
    'patience':           30,

    'target_length':      8192,
    'num_workers':        4,
    'drop_path':          0.15,
    'dropout':            0.18,
    'boundary_window':    64,
    'softargmax_window':  100,
    'use_transformer':    True,
    'use_boundary':       True,
    'use_multiscale':     True,

    'use_amp':            True,
    'fast_mode':          False,
    'lr_warmup_epochs':   5,
    'ema_decay':          0.999,
    'grad_accum_steps':   1,
    'use_tta':            True,

    'group_aware_split':       True,
    'use_stratified_sampler':  True,
    'stratified_bins':         20,
    'cutout_prob':             0.6,
    'resample_mode':           'poly',

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


def run_one_seed(seed, save_dir):
    from run_v15      import train_model, evaluate_model
    from evaluate_v15 import Evaluator

    config = {**BASE_CONFIG, 'seed': seed}
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed {seed} — 训练开始")
    print(f"{'='*60}")

    t0 = time.time()

    model, test_files, _ = train_model(
        data_dir=DATA_DIR,
        labels_excel=LABELS_EXCEL,
        save_dir=save_dir,
        config=config,
    )
    train_minutes = round((time.time() - t0) / 60, 1)
    print(f"\nSeed {seed} 训练完成，耗时 {train_minutes} 分钟")

    best_path   = os.path.join(save_dir, 'best_mae.pth')
    results_dir = os.path.join(save_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    _, eval_results, _ = evaluate_model(
        model_path=best_path,
        data_dir=DATA_DIR,
        labels_excel=LABELS_EXCEL,
        test_files=test_files,
        save_dir=save_dir,
        num_visualize=0,
    )

    errors_orig = np.array([
        abs(r['pred_idx'] - r['gt_idx'])
        for r in eval_results
        if r.get('gt_idx') is not None and r.get('pred_idx') is not None
    ])

    stats = {
        'seed':         seed,
        'n_samples':    len(errors_orig),
        'mae':          float(errors_orig.mean()),
        'median':       float(np.median(errors_orig)),
        'std':          float(errors_orig.std()),
        'p90':          float(np.percentile(errors_orig, 90)),
        'p95':          float(np.percentile(errors_orig, 95)),
        'train_minutes': train_minutes,
    }
    for t in ORIG_THRESHOLDS:
        stats[f'pct_le_{t}'] = float((errors_orig <= t).mean() * 100)

    np.save(os.path.join(results_dir, 'errors_orig.npy'), errors_orig)

    with open(os.path.join(results_dir, 'seed_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSeed {seed} 结果: MAE={stats['mae']:.2f} | "
          f"Median={stats['median']:.1f} | P95={stats['p95']:.1f}")

    return stats, errors_orig


def summarize(all_stats, all_errors, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)

    maes    = [s['mae']    for s in all_stats]
    medians = [s['median'] for s in all_stats]
    p95s    = [s['p95']    for s in all_stats]
    seeds   = [s['seed']   for s in all_stats]

    print(f"\n{'='*60}")
    print("  多种子汇总结果 (原始空间误差, pts)")
    print(f"{'='*60}")
    print(f"{'Seed':>8} {'MAE':>8} {'Median':>8} {'Std':>8} "
          f"{'P90':>8} {'P95':>8} {'≤10pts':>8} {'≤50pts':>8}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['seed']:>8} {s['mae']:>8.2f} {s['median']:>8.1f} "
              f"{s['std']:>8.2f} {s['p90']:>8.1f} {s['p95']:>8.1f} "
              f"{s['pct_le_10']:>7.1f}% {s['pct_le_50']:>7.1f}%")
    print("-" * 60)
    print(f"{'Mean':>8} {np.mean(maes):>8.2f} {np.mean(medians):>8.1f} "
          f"{'':>8} {np.mean(p95s):>8.1f}±{np.std(p95s):.1f}")
    print(f"{'Std':>8} {np.std(maes):>8.2f} {np.std(medians):>8.1f}")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    seed_labels = [f"Seed\n{s}" for s in seeds]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for ax, (metric, vals, title) in zip(axes, [
        ('MAE',    maes,    'Original Space MAE (pts)'),
        ('Median', medians, 'Original Space Median (pts)'),
        ('P95',    p95s,    'Original Space P95 (pts)'),
    ]):
        bars = ax.bar(seed_labels, vals, color=colors[:len(seeds)],
                      edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.axhline(np.mean(vals), color='red', ls='--', lw=1.5,
                   label=f'Mean={np.mean(vals):.2f}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (pts)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('AFM V15 ①+③  Multi-Seed Stability (Original Space)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'multi_seed_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] multi_seed_bar.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    max_err = max(np.percentile(e, 99) for e in all_errors)
    bins = np.linspace(0, max_err, 60)
    for errs, seed, col in zip(all_errors, seeds, colors):
        ax.hist(errs, bins=bins, alpha=0.45, color=col, label=f'Seed {seed}',
                density=True, edgecolor='none')
    ax.set_xlabel('Error (pts, original space)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Error Distribution (density)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for errs, seed, col in zip(all_errors, seeds, colors):
        se  = np.sort(errs)
        cdf = np.arange(1, len(se) + 1) / len(se)
        ax.plot(se, cdf, color=col, lw=2, label=f'Seed {seed} (MAE={np.mean(errs):.1f})')
    for pct, ls, label in [(0.90, '--', '90%'), (0.95, ':', '95%')]:
        ax.axhline(pct, color='gray', ls=ls, lw=1.2, alpha=0.7, label=label)
    ax.set_xlabel('Error threshold (pts, original space)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('CDF of Errors', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.suptitle('AFM V15 ①+③  Error Distribution across Seeds',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'multi_seed_error_dist.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] multi_seed_error_dist.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for s, errs, col in zip(all_stats, all_errors, colors):
        pcts = [(errs <= t).mean() * 100 for t in ORIG_THRESHOLDS]
        ax.plot(ORIG_THRESHOLDS, pcts, 'o-', color=col, lw=2,
                label=f"Seed {s['seed']} (MAE={s['mae']:.1f})", ms=5)
    mean_pcts = [
        np.mean([(errs <= t).mean() * 100 for errs in all_errors])
        for t in ORIG_THRESHOLDS
    ]
    ax.plot(ORIG_THRESHOLDS, mean_pcts, 's--', color='black', lw=2.5,
            label='Mean', ms=7, zorder=5)
    ax.set_xscale('log')
    ax.set_xticks(ORIG_THRESHOLDS)
    ax.set_xticklabels([str(t) for t in ORIG_THRESHOLDS], rotation=45, fontsize=9)
    ax.set_xlabel('Error Threshold (pts, original space)', fontsize=11)
    ax.set_ylabel('Cumulative Accuracy (%)', fontsize=11)
    ax.set_title('AFM V15 ①+③  Cumulative Accuracy across Seeds', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(90, color='gray', ls='--', alpha=0.5)
    ax.axhline(95, color='gray', ls=':', alpha=0.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'multi_seed_cumulative.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] multi_seed_cumulative.png")

    csv_path = os.path.join(summary_dir, 'multi_seed_summary.csv')
    fieldnames = list(all_stats[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
        mean_row = {'seed': 'MEAN'}
        std_row  = {'seed': 'STD'}
        for k in fieldnames:
            if k == 'seed':
                continue
            vals = [s[k] for s in all_stats if isinstance(s[k], (int, float))]
            if vals:
                mean_row[k] = round(float(np.mean(vals)), 4)
                std_row[k]  = round(float(np.std(vals)),  4)
        writer.writerow(mean_row)
        writer.writerow(std_row)
    print(f"[Saved] {csv_path}")

    summary = {
        'seeds':        seeds,
        'per_seed':     all_stats,
        'across_seeds': {
            'mae_mean':    round(float(np.mean(maes)),    2),
            'mae_std':     round(float(np.std(maes)),     2),
            'median_mean': round(float(np.mean(medians)), 2),
            'median_std':  round(float(np.std(medians)),  2),
            'p95_mean':    round(float(np.mean(p95s)),    2),
            'p95_std':     round(float(np.std(p95s)),     2),
        }
    }
    json_path = os.path.join(summary_dir, 'multi_seed_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Saved] {json_path}")

    return summary


def main():
    missing = []
    if not os.path.exists(DATA_DIR):
        missing.append(f"DATA_DIR     : {DATA_DIR}")
    if not os.path.exists(LABELS_EXCEL):
        missing.append(f"LABELS_EXCEL : {LABELS_EXCEL}")
    if missing:
        print("\n[错误] 以下路径不存在，请修改脚本顶部的路径配置：")
        for m in missing: print(f"  {m}")
        sys.exit(1)

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    t_total = time.time()

    all_stats  = []
    all_errors = []

    for seed in SEEDS:
        seed_dir = os.path.join(BASE_SAVE_DIR, f'seed_{seed}')
        stats, errors = run_one_seed(seed, seed_dir)
        all_stats.append(stats)
        all_errors.append(errors)

    summary_dir = os.path.join(BASE_SAVE_DIR, 'summary')
    summary = summarize(all_stats, all_errors, summary_dir)

    total_h = round((time.time() - t_total) / 3600, 2)
    print(f"\n{'='*60}")
    print(f"  全部完成！4 个种子总耗时 {total_h} 小时")
    print(f"  结果根目录 : {BASE_SAVE_DIR}")
    print(f"  汇总目录   : {summary_dir}")
    print(f"{'='*60}")
    print(f"\n  MAE:  {summary['across_seeds']['mae_mean']:.2f} ± "
          f"{summary['across_seeds']['mae_std']:.2f} pts")
    print(f"  P95:  {summary['across_seeds']['p95_mean']:.2f} ± "
          f"{summary['across_seeds']['p95_std']:.2f} pts")


if __name__ == '__main__':
    main()
