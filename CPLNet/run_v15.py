import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_v15 import create_dataloaders
from model_v15   import AFMContactDetector, windowed_soft_argmax
from losses_v15  import ImprovedCombinedLoss, AdaptiveSigmaScheduler, generate_heatmap_batch


def _sanitize_config(cfg):
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in cfg.items()}


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay       = decay
        self.num_updates = 0
        self.module      = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.data, alpha=1.0 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def update_bn(self, loader, device, n_batches=20):
        self.module.train()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            self.module(batch['signal'].to(device))
        self.module.eval()
        print(f"[INFO] EMA BN stats updated ({min(n_batches, len(loader))} batches)")

    def state_dict(self):
        return self.module.state_dict()


@torch.no_grad()
def compute_mae(model, loader, device, target_length,
                use_tta=False, tta_n=5, window=100, amp_dtype=None):
    model.eval()
    if use_tta:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    coords = torch.linspace(0, 1, target_length, device=device)
    hard_rs, soft_rs   = [], []
    hard_orig, soft_orig = [], []

    use_amp = amp_dtype is not None and device.type == 'cuda'

    for batch in loader:
        signal   = batch['signal'].to(device, non_blocking=True)
        gt_rs    = batch['resampled_contact_idx']
        orig_len = batch['original_length']
        gt_orig  = batch['original_contact_idx']

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            if use_tta:
                hm_sum = torch.zeros(signal.shape[0], target_length, device=device)
                for _ in range(tta_n):
                    hm, _ = model(signal, compute_position=False)
                    hm_sum += hm
                hm_avg = hm_sum / tta_n
                pos = windowed_soft_argmax(hm_avg, coords, window=window)
            else:
                hm_avg, pos_direct = model(signal)
                pos = pos_direct.squeeze(1)

        pred_hard = hm_avg.argmax(dim=1).float().cpu()
        pred_soft = (pos * (target_length - 1)).round().clamp(0, target_length - 1).float().cpu()

        for ph, ps, g_rs, ol, g_o in zip(pred_hard, pred_soft, gt_rs,
                                          orig_len, gt_orig):
            ol_i = int(ol.item()); g_o_i = int(g_o.item())
            denom = max(ol_i - 1, 1)
            hard_rs.append(abs(ph.item() - g_rs.item()))
            soft_rs.append(abs(ps.item() - g_rs.item()))
            ph_o = min(round(ph.item() * denom / (target_length - 1)), denom)
            ps_o = min(round(ps.item() * denom / (target_length - 1)), denom)
            hard_orig.append(abs(ph_o - g_o_i))
            soft_orig.append(abs(ps_o - g_o_i))

    model.eval()
    return {
        'hard_rs':   float(np.mean(hard_rs)),
        'soft_rs':   float(np.mean(soft_rs)),
        'hard_orig': float(np.mean(hard_orig)),
        'soft_orig': float(np.mean(soft_orig)),
        'soft_orig_median': float(np.median(soft_orig)),
    }


@torch.no_grad()
def compute_train_loss_aligned(model, loader, criterion, device, target_length,
                                sigma_min, sigma_max, heatmap_type, n_batches=25):
    model.eval()
    accum = {'loss': 0.0, 'hm': 0.0, 'ord': 0.0}
    cnt   = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        signal          = batch['signal'].to(device)
        gt_rs           = batch['resampled_contact_idx'].to(device)
        target_heatmap  = generate_heatmap_batch(
            gt_rs, target_length, sigma_min, device, htype=heatmap_type)
        target_position = gt_rs.float() / (target_length - 1)
        pred_heatmap, _ = model(signal, compute_position=False)
        loss, losses = criterion(
            pred_heatmap, target_heatmap, target_position,
            current_sigma=sigma_min, sigma_max=sigma_max, sigma_min=sigma_min)
        accum['loss'] += loss.item()
        accum['hm']   += losses['heatmap'].item()
        accum['ord']  += losses['ordinal'].item()
        cnt += 1
    return {k: v / max(cnt, 1) for k, v in accum.items()}


def train_model(data_dir, labels_excel, save_dir, config):
    print("=" * 60)
    print("AFM Contact Point Detection - Training V15")
    print("=" * 60)

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fast_mode = config.get('fast_mode', False)
    if fast_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_amp = config.get('use_amp', True) and torch.cuda.is_available()
    amp_dtype = None
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] Random seed: {seed} "
          f"(cudnn.deterministic={not fast_mode}, benchmark={fast_mode})")
    print(f"[INFO] AMP: {'ON ('+str(amp_dtype).split('.')[-1]+')' if amp_dtype else 'OFF'}"
          f" | fast_mode={fast_mode}")
    os.makedirs(save_dir, exist_ok=True)

    target_length     = config['target_length']
    num_epochs        = config['num_epochs']
    heatmap_type      = config.get('heatmap_type', 'laplacian')
    softargmax_window = config.get('softargmax_window', 100)

    print("\n[INFO] Loading data...")
    train_loader, val_loader, test_loader, test_files, labels_dict = create_dataloaders(
        data_dir=data_dir, labels_excel=labels_excel,
        batch_size=config['batch_size'], target_length=target_length,
        num_workers=config['num_workers'],
        use_stratified_sampler=config.get('use_stratified_sampler', True),
        stratified_bins=config.get('stratified_bins', 20),
        group_aware_split=config.get('group_aware_split', True),
        random_seed=seed,
        cutout_prob=config.get('cutout_prob', 0.6),
        resample_mode=config.get('resample_mode', 'poly'),
    )

    with open(os.path.join(save_dir, 'test_files.json'), 'w', encoding='utf-8') as f:
        json.dump(test_files, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(_sanitize_config(config), f, ensure_ascii=False, indent=2)

    print("\n[INFO] Creating model V15 (2-channel: deflection + d/dx)...")
    model = AFMContactDetector(
        in_channels=2, target_length=target_length,
        drop_path=config.get('drop_path', 0.15), dropout=config.get('dropout', 0.18),
        boundary_window=config.get('boundary_window', 64),
        softargmax_window=softargmax_window,
        use_transformer=config.get('use_transformer', True),
        use_boundary=config.get('use_boundary', True),
        use_multiscale=config.get('use_multiscale', True),
    ).to(device)
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ema_decay = config.get('ema_decay', 0.999)
    ema       = EMAModel(model, decay=ema_decay)
    print(f"[INFO] EMA enabled (decay={ema_decay}, bias-corrected)")

    sigma_max_val = config.get('sigma_max', 40)
    sigma_min_val = config.get('sigma_min', 5)

    train_coords  = torch.linspace(0, 1, target_length, device=device)

    criterion = ImprovedCombinedLoss(
        target_length=target_length,
        heatmap_weight=config.get('heatmap_weight', 1.0),
        ordinal_weight=config.get('ordinal_weight', 0.15),
        sharpness_radius=config.get('sharpness_radius', 40),
        ordinal_n_neg=config.get('ordinal_n_neg', 20),
        ordinal_margin=config.get('ordinal_margin', 4.0),
        ordinal_neg_thresh=config.get('ordinal_neg_thresh', 0.1),
        sharpness_weight=config.get('sharpness_weight', 0.5),
    ).to(device)

    sigma_sched_epochs = config.get('sigma_schedule_epochs',
                                     min(num_epochs, 70))
    sigma_scheduler = AdaptiveSigmaScheduler(
        sigma_max=sigma_max_val, sigma_min=sigma_min_val,
        total_epochs=sigma_sched_epochs,
        warmup_epochs=config.get('sigma_warmup', 2))
    print(f"[INFO] Sigma schedule: {sigma_max_val}→{sigma_min_val} "
          f"over {sigma_sched_epochs} epochs (reaches min at ep {sigma_sched_epochs})")

    optimizer     = optim.AdamW(model.parameters(),
                                lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    warmup_epochs = config.get('lr_warmup_epochs', 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(1e-6 / config['learning_rate'],
                   0.5 * (1 + np.cos(np.pi * progress)))

    scheduler   = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    accum_steps = config.get('grad_accum_steps', 1)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_loss_aligned',
        'train_hm_aligned', 'train_ord_aligned',
        'train_heatmap', 'val_heatmap',
        'train_ordinal', 'val_ordinal',
        'val_mae_hard', 'val_mae_soft', 'train_mae_soft',
        'val_mae_soft_ema',
        'val_mae_soft_orig', 'val_mae_soft_orig_ema',
        'lr', 'sigma',
        'eff_w_heatmap', 'eff_w_ordinal', 'peak_weight']}

    best_val_mae     = float('inf')
    best_val_loss    = float('inf')
    best_epoch       = 0
    patience_counter = 0
    save_dict_base   = {}

    print("\n[INFO] Starting training...\n" + "-" * 60)

    for epoch in range(num_epochs):
        start_time    = time.time()
        current_sigma = sigma_scheduler.get_sigma(epoch)

        model.train(); criterion.train()
        accum = {k: 0.0 for k in
                 ['loss', 'hm', 'ord', 'w_hm', 'w_ord', 'peak_w', 'train_mae']}
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for step, batch in enumerate(pbar):
            signal          = batch['signal'].to(device, non_blocking=True)
            gt_rs           = batch['resampled_contact_idx'].to(device, non_blocking=True)
            target_heatmap  = generate_heatmap_batch(
                gt_rs, target_length, current_sigma, device, htype=heatmap_type)
            target_position = gt_rs.float() / (target_length - 1)

            with torch.autocast(device_type='cuda', dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
                pred_heatmap, _ = model(signal, compute_position=False)

                pred_position = windowed_soft_argmax(
                    pred_heatmap, train_coords, window=softargmax_window
                ).unsqueeze(1)

                loss, losses = criterion(
                    pred_heatmap, target_heatmap, target_position,
                    current_sigma=current_sigma, sigma_max=sigma_max_val,
                    sigma_min=sigma_min_val)

            with torch.no_grad():
                _pred_idx = (pred_position.detach().squeeze(1) * (target_length - 1)
                             ).round().long().clamp(0, target_length - 1)
                accum['train_mae'] += (_pred_idx.cpu() - gt_rs.cpu()).float().abs().mean().item()

            scaler.scale(loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            accum['loss']   += loss.item()
            accum['hm']     += losses['heatmap'].item()
            accum['ord']    += losses['ordinal'].item()
            accum['w_hm']   += losses['eff_w_heatmap']
            accum['w_ord']  += losses['eff_w_ordinal']
            accum['peak_w'] += losses['peak_weight']
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'peak_w': f'{losses["peak_weight"]:.2f}'})

        train_loss = accum['loss']      / num_batches
        train_hm   = accum['hm']        / num_batches
        train_ord  = accum['ord']       / num_batches
        train_mae  = accum['train_mae'] / num_batches
        avg_w      = {k: accum[k] / num_batches for k in ['w_hm', 'w_ord']}
        avg_peak_w = accum['peak_w'] / num_batches

        model.eval(); criterion.eval()
        val_accum   = {k: 0.0 for k in ['loss', 'hm', 'ord']}
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                signal          = batch['signal'].to(device, non_blocking=True)
                gt_rs           = batch['resampled_contact_idx'].to(device, non_blocking=True)
                target_heatmap  = generate_heatmap_batch(
                    gt_rs, target_length, sigma_min_val, device, htype=heatmap_type)
                target_position = gt_rs.float() / (target_length - 1)
                with torch.autocast(device_type='cuda', dtype=amp_dtype,
                                    enabled=(amp_dtype is not None)):
                    pred_heatmap, _ = model(signal, compute_position=False)
                    loss, losses = criterion(
                        pred_heatmap, target_heatmap, target_position,
                        current_sigma=sigma_min_val, sigma_max=sigma_max_val,
                        sigma_min=sigma_min_val)
                val_accum['loss'] += loss.item()
                val_accum['hm']   += losses['heatmap'].item()
                val_accum['ord']  += losses['ordinal'].item()
                val_batches += 1

        val_loss = val_accum['loss'] / val_batches
        val_hm   = val_accum['hm']   / val_batches
        val_ord  = val_accum['ord']  / val_batches

        aligned = compute_train_loss_aligned(
            model, train_loader, criterion, device, target_length,
            sigma_min=sigma_min_val, sigma_max=sigma_max_val,
            heatmap_type=heatmap_type, n_batches=25)

        mae_raw = compute_mae(
            model, val_loader, device, target_length, use_tta=False,
            window=softargmax_window, amp_dtype=amp_dtype)
        mae_ema = compute_mae(
            ema.module, val_loader, device, target_length, use_tta=False,
            window=softargmax_window, amp_dtype=amp_dtype)

        val_mae_hard          = mae_raw['hard_rs']
        val_mae_soft          = mae_raw['soft_rs']
        val_mae_soft_ema      = mae_ema['soft_rs']
        val_mae_soft_orig     = mae_raw['soft_orig']
        val_mae_soft_orig_ema = mae_ema['soft_orig']

        scheduler.step()

        val_mae_for_selection = min(val_mae_soft_orig, val_mae_soft_orig_ema)
        ema_is_better         = val_mae_soft_orig_ema < val_mae_soft_orig

        for k, v in [('train_loss', train_loss), ('val_loss', val_loss),
                     ('train_loss_aligned', aligned['loss']),
                     ('train_hm_aligned',   aligned['hm']),
                     ('train_ord_aligned',  aligned['ord']),
                     ('train_heatmap', train_hm), ('val_heatmap', val_hm),
                     ('train_ordinal', train_ord), ('val_ordinal', val_ord),
                     ('val_mae_hard', val_mae_hard), ('val_mae_soft', val_mae_soft),
                     ('val_mae_soft_ema', val_mae_soft_ema),
                     ('val_mae_soft_orig', val_mae_soft_orig),
                     ('val_mae_soft_orig_ema', val_mae_soft_orig_ema),
                     ('train_mae_soft', train_mae),
                     ('lr', optimizer.param_groups[0]['lr']),
                     ('sigma', current_sigma),
                     ('eff_w_heatmap', avg_w['w_hm']),
                     ('eff_w_ordinal', avg_w['w_ord']),
                     ('peak_weight', avg_peak_w)]:
            history[k].append(v)

        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
              f"sigma={current_sigma:.1f} | peak_w={avg_peak_w:.2f}")
        print(f"  Train (dyn sigma):     {train_loss:.4f} "
              f"(hm:{train_hm:.4f} ord:{train_ord:.4f})")
        print(f"  Train (aligned sigma): {aligned['loss']:.4f} "
              f"(hm:{aligned['hm']:.4f} ord:{aligned['ord']:.4f})")
        print(f"  Val:                   {val_loss:.4f} "
              f"(hm:{val_hm:.4f} ord:{val_ord:.4f})")
        print(f"  MAE: Train={train_mae:.2f} | Val={val_mae_soft:.2f} | "
              f"Val-EMA={val_mae_soft_ema:.2f} | Val-Hard={val_mae_hard:.2f}")
        print(f"  MAE[orig]: Val={val_mae_soft_orig:.2f} | "
              f"Val-EMA={val_mae_soft_orig_ema:.2f} | "
              f"sel={val_mae_for_selection:.2f} ({'EMA' if ema_is_better else 'RAW'})")

        save_dict_base = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_state_dict':   ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae_for_selection,
            'val_mae_orig': val_mae_for_selection,
            'val_mae_soft_rs':     val_mae_soft,
            'val_mae_soft_ema_rs': val_mae_soft_ema,
            'val_mae_soft_orig':     val_mae_soft_orig,
            'val_mae_soft_orig_ema': val_mae_soft_orig_ema,
            'ema_is_better':    ema_is_better,
            'config': _sanitize_config(config)}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(save_dict_base, os.path.join(save_dir, 'best_loss.pth'))
            print("  -> Saved best loss model!")

        if val_mae_for_selection < best_val_mae:
            best_val_mae     = val_mae_for_selection
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save(save_dict_base, os.path.join(save_dir, 'best_mae.pth'))
            tag = "EMA" if ema_is_better else "RAW"
            print(f"  -> Saved best MAE model! ({tag} orig-soft={val_mae_for_selection:.2f})")
        else:
            patience_counter += 1
            print(f"  patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print(f"\n[INFO] Early stopping at epoch {epoch+1}")
            break

    if save_dict_base:
        torch.save(save_dict_base, os.path.join(save_dir, 'final.pth'))
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, save_dir, best_epoch=best_epoch)

    _best_path = os.path.join(save_dir, 'best_mae.pth')
    if os.path.exists(_best_path):
        print("\n[INFO] Post-training TTA val check on best_mae.pth ...")
        try:
            _ckpt = torch.load(_best_path, map_location=device, weights_only=False)
            if _ckpt.get('ema_is_better', False):
                model.load_state_dict(_ckpt['ema_state_dict'])
                src = "EMA"
                ema.module.load_state_dict(_ckpt['ema_state_dict'])
                ema.update_bn(train_loader, device, n_batches=20)
                _ckpt['ema_state_dict'] = ema.module.state_dict()
                torch.save(_ckpt, _best_path)
                print("[INFO] best_mae.pth updated with BN-corrected EMA weights.")
            else:
                model.load_state_dict(_ckpt['model_state_dict'])
                src = "RAW"
            tta = compute_mae(
                model, val_loader, device, target_length,
                use_tta=True, tta_n=5,
                window=softargmax_window, amp_dtype=amp_dtype)
            tta_soft = tta['soft_orig']; tta_hard = tta['hard_orig']
            gap = abs(tta_soft - best_val_mae)
            print(f"[INFO] Best ({src}) TTA val MAE[orig]: soft={tta_soft:.2f} | "
                  f"hard={tta_hard:.2f}")
            print(f"[INFO] Early-stop no-TTA MAE[orig]: {best_val_mae:.2f} | gap={gap:.2f} pts")
            if gap > 5.0:
                print(f"[WARN] TTA vs no-TTA gap ({gap:.1f} pts) > 5 pts.")
        except Exception as _e:
            print(f"[WARN] Post-training check failed: {_e}")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best orig-space soft MAE: {best_val_mae:.2f} (epoch {best_epoch})")
    print("=" * 60)
    return model, test_files, labels_dict


def plot_training_curves(history, save_dir, best_epoch=None):
    fig, axes = plt.subplots(4, 2, figsize=(15, 24))
    ep = range(1, len(history['train_loss']) + 1)

    def _plot(ax, keys_labels, title, ylog=False):
        for key, label, color in keys_labels:
            if key in history and history[key]:
                ax.plot(ep, history[key][:len(ep)], color,
                        label=label, linewidth=1.5)
        ax.set_title(title); ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3); ax.set_xlabel('Epoch')
        if ylog:
            ax.set_yscale('log')
        if best_epoch is not None:
            ax.axvline(best_epoch, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    _plot(axes[0, 0],
          [('train_loss_aligned', 'Train (aligned: sigma=sigma_min)', 'b-'),
           ('val_loss',           'Val (sigma=sigma_min)',            'r-')],
          'Total Loss (①+③, Train/Val same protocol)')
    _plot(axes[0, 1],
          [('train_hm_aligned', 'Train (aligned)', 'b-'),
           ('val_heatmap',      'Val',             'r-')],
          '① Heatmap Loss (same protocol)')
    _plot(axes[1, 0],
          [('train_ord_aligned', 'Train (aligned)', 'b-'),
           ('val_ordinal',       'Val',             'r-')],
          '③ Ordinal Ranking Loss (same protocol)')
    _plot(axes[1, 1],
          [('val_mae_soft_orig',     'Val orig (raw)', 'm-'),
           ('val_mae_soft_orig_ema', 'Val orig (EMA)', 'c-'),
           ('val_mae_soft',          'Val rs (raw)',   'm--'),
           ('val_mae_soft_ema',      'Val rs (EMA)',   'c--'),
           ('train_mae_soft',        'Train rs (in-loop)', 'b-')],
          'MAE (pts): orig=main metric, rs=resampled', ylog=True)
    _plot(axes[2, 0],
          [('eff_w_heatmap', 'Heatmap ①', 'b-'),
           ('eff_w_ordinal', 'Ordinal ③', 'g-')], 'Effective Task Weights')
    _plot(axes[2, 1],
          [('sigma',       'Sigma',       'c-'),
           ('peak_weight', 'Peak Weight', 'k:')], 'Sigma / Peak-w')
    _plot(axes[3, 0], [('lr', 'LR', 'm-')], 'Learning Rate')
    axes[3, 1].axis('off')

    plt.suptitle(f'AFM V15 ①+③ Training Curves'
                 + (f' | Best epoch: {best_epoch}' if best_epoch else ''),
                 fontsize=13, y=1.001)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"[Saved] training_curves.png")


def evaluate_model(model_path, data_dir, labels_excel, test_files,
                   save_dir, num_visualize=20):
    print("\n" + "=" * 60)
    print("AFM Contact Point Detection - Evaluation V15")
    print("=" * 60)
    from evaluate_v15 import Evaluator

    evaluator   = Evaluator(model_path=model_path, data_dir=data_dir,
                            labels_excel=labels_excel)
    results_dir = os.path.join(save_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    results, errors = evaluator.evaluate_testset(
        test_files, use_tta=True, tta_n=5, save_dir=results_dir)

    with open(os.path.join(results_dir, 'predictions.json'), 'w',
              encoding='utf-8') as f:
        json.dump([{'filename': r['filename'],
                    'pred_idx': int(r['pred_idx']),
                    'gt_idx':   int(r['gt_idx']) if r['gt_idx'] is not None else None,
                    'original_length': int(r['original_length'])}
                   for r in results],
                  f, indent=2, ensure_ascii=False)

    if num_visualize > 0:
        evaluator.visualize_predictions(test_files, num_samples=num_visualize,
                                        save_dir=results_dir)

    errors = np.array(errors)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(errors.mean(), color='r', linestyle='--',
                    label=f'Mean: {errors.mean():.1f}')
    axes[0].axvline(np.median(errors), color='g', linestyle=':',
                    label=f'Median: {np.median(errors):.1f}')
    axes[0].set_title('Error Distribution (resampled space)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    se  = np.sort(errors); cdf = np.arange(1, len(se) + 1) / len(se)
    axes[1].plot(se, cdf, 'b-')
    for pct, col in [(0.9, 'r'), (0.95, 'g'), (0.99, 'purple')]:
        axes[1].axhline(y=pct, color=col, linestyle='--', alpha=0.7,
                        label=f'{int(pct*100)}%: {np.percentile(errors, pct*100):.1f} pts')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_distribution.png'), dpi=150)
    plt.close()
    print(f"\n[INFO] Results saved to: {results_dir}")
    return evaluator, results, errors
