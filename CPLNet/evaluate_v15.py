import os
import csv
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_v15 import load_labels
from model_v15   import AFMContactDetector, windowed_soft_argmax

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class Evaluator:
    def __init__(self, model_path, data_dir, labels_excel,
                 target_length=4096, device='cuda'):
        self.data_dir          = data_dir
        self.target_length     = target_length
        self.softargmax_window = 100
        self.device            = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        self.labels_dict     = load_labels(labels_excel)
        self._last_save_dir  = None
        self._coords         = torch.linspace(0, 1, self.target_length,
                                              device=self.device)

    def _load_model(self, model_path):
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt.get('config', {})
        self.target_length     = cfg.get('target_length', self.target_length)
        self.softargmax_window = cfg.get('softargmax_window', self.softargmax_window)
        model = AFMContactDetector(
            in_channels=2, target_length=self.target_length,
            drop_path=cfg.get('drop_path', 0.15), dropout=cfg.get('dropout', 0.18),
            boundary_window=cfg.get('boundary_window', 64),
            softargmax_window=self.softargmax_window)

        if ckpt.get('ema_is_better', False) and 'ema_state_dict' in ckpt:
            sd = ckpt['ema_state_dict']
            print("  [INFO] Loading EMA state_dict (ema_is_better=True)")
        else:
            sd = ckpt['model_state_dict']
            print("  [INFO] Loading raw model state_dict")

        if any(k.startswith('module.') for k in sd):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model = model.to(self.device)
        print(f"Model loaded: {model_path}")
        print(f"  Epoch: {ckpt.get('epoch','N/A')}  "
              f"Val MAE: {ckpt.get('val_mae','N/A')}")
        print(f"  softargmax_window: {self.softargmax_window}")
        return model

    def _load_raw_curve(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data.append(float(line))
                except ValueError:
                    pass
        data = np.array(data)
        d = data[:len(data) // 2]
        return d

    def _smooth_derivative(self, data, poly=3):
        window = 11
        if not _HAS_SCIPY or len(data) < window + 1:
            return np.gradient(data)
        try:
            return savgol_filter(data, window_length=window, polyorder=poly, deriv=1)
        except Exception:
            return np.gradient(data)

    def _preprocess(self, deflection):
        L    = self.target_length
        orig = len(deflection)

        try:
            from scipy.signal import resample_poly
            from scipy.interpolate import PchipInterpolator
            from math import gcd

            def _rs(arr):
                if len(arr) == L:
                    return arr.copy()
                if len(arr) > L:
                    g = gcd(len(arr), L)
                    return resample_poly(arr, L // g, len(arr) // g)[:L]
                x_old = np.linspace(0, 1, len(arr))
                x_new = np.linspace(0, 1, L)
                return PchipInterpolator(x_old, arr)(x_new)
        except ImportError:
            def _rs(arr):
                return np.interp(np.linspace(0, 1, L),
                                 np.linspace(0, 1, len(arr)), arr)

        def _norm(arr):
            s = arr.std()
            return (arr - arr.mean()) / (s if s > 1e-8 else 1.0)

        d_resampled = _rs(deflection)
        dd = self._smooth_derivative(d_resampled)

        sig = np.stack([_norm(d_resampled), _norm(dd)], axis=0)
        return torch.tensor(sig, dtype=torch.float32).unsqueeze(0), orig

    @torch.no_grad()
    def _predict_single(self, filename, use_tta=True, tta_n=5):
        d        = self._load_raw_curve(filename)
        orig_len = len(d)
        sig, _   = self._preprocess(d)
        sig      = sig.to(self.device)
        L        = self.target_length

        if use_tta:
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            hm_sum = torch.zeros(1, L, device=self.device)
            for _ in range(tta_n):
                hm, _ = self.model(sig, compute_position=False)
                hm_sum += hm
            self.model.eval()
            pred_hm = hm_sum / tta_n
        else:
            pred_hm, _ = self.model(sig, compute_position=False)

        pos = windowed_soft_argmax(pred_hm, self._coords, window=self.softargmax_window)
        pred_rs      = int(np.clip(round(pos.item() * (L - 1)), 0, L - 1))
        pred_rs_hard = int(pred_hm.argmax(dim=1).item())
        pred_hm_vis  = (pred_hm / (pred_hm.amax(dim=1, keepdim=True) + 1e-8)
                        ).cpu().numpy().squeeze()

        denom_o   = max(orig_len - 1, 1)
        gt_orig   = self.labels_dict.get(filename)
        gt_rs     = (int(np.clip(round(gt_orig * (L - 1) / denom_o), 0, L - 1))
                     if gt_orig is not None else None)
        pred_orig = int(np.clip(round(pred_rs * denom_o / (L - 1)), 0, orig_len - 1))

        return dict(pred_idx=pred_orig, pred_idx_rs=pred_rs,
                    pred_idx_rs_hard=pred_rs_hard,
                    gt_idx=gt_orig, gt_idx_rs=gt_rs,
                    deflection=d,
                    pred_heatmap=pred_hm_vis, original_length=orig_len)

    def evaluate_testset(self, test_files, use_tta=True, tta_n=5,
                         outlier_thresh=500, save_dir=None):
        if save_dir is not None:
            self._last_save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        results, errors_rs, errors_orig, rel_errors = [], [], [], []
        print(f"\nEvaluating {len(test_files)} samples "
              f"(TTA={'ON' if use_tta else 'OFF'})...")
        for fn in test_files:
            try:
                r = self._predict_single(fn, use_tta=use_tta, tta_n=tta_n)
                results.append({'filename': fn,
                                'pred_idx': r['pred_idx'],
                                'pred_idx_rs': r['pred_idx_rs'],
                                'gt_idx': r['gt_idx'],
                                'gt_idx_rs': r['gt_idx_rs'],
                                'original_length': r['original_length']})
                if r['gt_idx_rs'] is not None and r['gt_idx'] is not None:
                    errors_rs.append(abs(r['pred_idx_rs'] - r['gt_idx_rs']))
                    errors_orig.append(abs(r['pred_idx'] - r['gt_idx']))
                    rel_errors.append(
                        abs(r['pred_idx'] - r['gt_idx']) / r['original_length'] * 100)
            except Exception as ex:
                print(f"Error {fn}: {ex}")

        errors_rs    = np.array(errors_rs)
        errors_orig  = np.array(errors_orig)
        rel_errors   = np.array(rel_errors)
        orig_lengths = np.array([r['original_length'] for r in results])
        outlier_mask = errors_rs > outlier_thresh
        n_outliers   = outlier_mask.sum()
        errors_clean = errors_rs[~outlier_mask]

        print(f"\n=== V15 Evaluation Results ===")
        print(f"Samples: {len(results)}")
        print(f"\n[Original Signal Length]")
        print(f"  Min: {orig_lengths.min():>8d}   Max: {orig_lengths.max():>8d}")
        print(f"  Mean: {orig_lengths.mean():>7.0f}   "
              f"Median: {np.median(orig_lengths):>8.0f}")
        ratios = orig_lengths / self.target_length
        print(f"  Resample ratio: min={ratios.min():.2f} "
              f"mean={ratios.mean():.2f} max={ratios.max():.2f}")
        print(f"\n[Resampled Space - {self.target_length} pts]")
        print(f"  MAE: {errors_rs.mean():>8.2f}  "
              f"Median: {np.median(errors_rs):>7.2f}  Std: {errors_rs.std():.2f}")
        if n_outliers > 0:
            print(f"  Clean: {errors_clean.mean():>8.2f} "
                  f"[excluded {n_outliers} outliers > {outlier_thresh}]")
        for t in [5, 10, 20, 50, 100, 200, 500]:
            print(f"    <= {t:4d} pts (rs): {(errors_rs <= t).mean()*100:>5.1f}%")
        print(f"\n[Original Space]")
        print(f"  MAE: {errors_orig.mean():>8.1f}  "
              f"Median: {np.median(errors_orig):>7.1f}  Std: {errors_orig.std():.1f}")
        for p in [50, 75, 90, 95, 99]:
            print(f"  P{p:>2d}: {np.percentile(errors_orig, p):>8.1f}")
        for t in [5, 10, 20, 50, 100, 200, 500]:
            print(f"    <= {t:4d} pts (orig): {(errors_orig <= t).mean()*100:>5.1f}%")
        print(f"  Relative MAE: {rel_errors.mean():.3f}% of signal length")

        n_show  = min(30, len(errors_orig))
        top_idx = np.argsort(errors_orig)[-n_show:][::-1]
        print(f"\n[Top-{n_show} Largest Errors]")
        print(f"  {'Rank':>4} {'Err(orig)':>10} {'Err(rs)':>9} "
              f"{'Pred':>11} {'GT':>9} {'OrigLen':>8}  Filename")
        for rank, i in enumerate(top_idx, 1):
            r = results[i]
            print(f"  {rank:>4} {errors_orig[i]:>10.0f} {errors_rs[i]:>9.0f} "
                  f"{r['pred_idx']:>11d} {r['gt_idx']:>9d} "
                  f"{r['original_length']:>8d}  {r['filename']}")

        if self._last_save_dir is not None and os.path.isdir(self._last_save_dir):
            self._save_error_report(results, errors_rs, errors_orig,
                                    self._last_save_dir)
        return results, errors_rs

    def _save_error_report(self, results, errors_rs, errors_orig, save_dir):
        try:
            full_csv = os.path.join(save_dir, 'evaluation_errors_full.csv')
            order    = np.argsort(errors_orig)[::-1]
            with open(full_csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['rank', 'filename', 'err_orig', 'err_rs',
                            'pred_orig', 'gt_orig', 'pred_rs', 'gt_rs',
                            'original_length', 'resample_ratio'])
                for rank, i in enumerate(order, 1):
                    r = results[i]
                    w.writerow([rank, r['filename'],
                                int(errors_orig[i]), int(errors_rs[i]),
                                r['pred_idx'], r['gt_idx'],
                                r['pred_idx_rs'], r['gt_idx_rs'],
                                r['original_length'],
                                f"{r['original_length']/self.target_length:.3f}"])
            print(f"[Saved] {full_csv}")
        except Exception as e:
            print(f"[Warn] Failed to save error report: {e}")

    def visualize_predictions(self, test_files, num_samples=20,
                              save_dir='./results', use_tta=True):
        if num_samples <= 0:
            return
        os.makedirs(save_dir, exist_ok=True)
        self._last_save_dir = save_dir
        selected  = random.sample(test_files, min(num_samples, len(test_files)))
        nc, nr    = 4, (len(selected) + 3) // 4
        fig, axes = plt.subplots(nr, nc, figsize=(20, 4 * nr))
        axes = (np.array(axes).flatten() if nr > 1 else
                (np.array([axes]) if not isinstance(axes, np.ndarray)
                 else axes.flatten()))
        for i, fn in enumerate(selected):
            ax = axes[i]
            try:
                r = self._predict_single(fn, use_tta=use_tta, tta_n=3)
                x = np.arange(len(r['deflection']))
                ax.plot(x, r['deflection'], 'b-', lw=0.8)
                ax.axvline(r['pred_idx'], color='g', ls='--', lw=1.5,
                           label=f"Pred:{r['pred_idx']}")
                ax.plot(r['pred_idx'], r['deflection'][r['pred_idx']], 'go', ms=10)
                if r['gt_idx'] is not None:
                    ax.axvline(r['gt_idx'], color='r', ls=':', lw=1.5,
                               label=f"GT:{r['gt_idx']}")
                    ax.plot(r['gt_idx'], r['deflection'][r['gt_idx']],
                            'ro', ms=8, mfc='none', mew=2)
                    ax.set_title(f"{fn}\n"
                                 f"Err={abs(r['pred_idx']-r['gt_idx'])}(orig)",
                                 fontsize=8)
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            except Exception:
                ax.set_title(f"{fn}\nError"); ax.axis('off')
        for i in range(len(selected), len(axes)):
            axes[i].axis('off')
        plt.suptitle('V15 AFM Contact Point Detection', fontsize=13, y=1.01)
        plt.tight_layout()
        sp = os.path.join(save_dir, 'prediction_visualization.png')
        plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
        print(f"[Saved] {sp}")
