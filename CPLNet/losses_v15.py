import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_heatmap_batch(contact_idx, length, sigma, device, htype='laplacian'):
    x = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
    c = contact_idx.to(device).float().unsqueeze(1)
    if htype == 'laplacian':
        scale = max(sigma / math.sqrt(2), 1e-6)
        return torch.exp(-(x - c).abs() / scale)
    return torch.exp(-((x - c) ** 2) / (2 * max(sigma, 1e-6) ** 2))


class HeatmapLoss(nn.Module):
    def __init__(self, ce_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight

    def forward(self, pred, target, peak_weight=1.0, current_sigma=None):
        t_dist = target / (target.sum(dim=1, keepdim=True) + 1e-8)
        log_p  = torch.log(pred.clamp(min=1e-8))

        ce = -(t_dist * log_p).sum(dim=1).mean()
        with torch.no_grad():
            log_t = torch.log(t_dist.clamp(min=1e-8))
            ht    = -(t_dist * log_t).sum(dim=1).mean()
        kl_norm = ce - ht

        return self.ce_weight * kl_norm


class OrdinalRankingLoss(nn.Module):
    def __init__(self, n_neg=20, margin=4.0, neg_thresh=0.1):
        super().__init__()
        self.n_neg      = n_neg
        self.margin     = margin
        self.neg_thresh = neg_thresh

    def forward(self, pred_heatmap, target_position, target_heatmap):
        B, L   = pred_heatmap.shape
        device = pred_heatmap.device
        log_p  = torch.log(pred_heatmap.clamp(min=1e-8))

        contact_idx = (target_position * (L - 1)).round().long().clamp(0, L - 1)
        contact_val = log_p[torch.arange(B, device=device), contact_idx]

        tgt_norm = target_heatmap / (target_heatmap.amax(dim=1, keepdim=True) + 1e-8)

        n_hard = self.n_neg // 2
        n_rand = self.n_neg - n_hard

        neg_probs = pred_heatmap.detach().clone()
        neg_probs.masked_fill_(tgt_norm >= self.neg_thresh, 0.0)

        avail_min  = int((neg_probs > 0).long().sum(dim=1).min().clamp(min=1).item())
        pool_k     = max(min(n_hard * 2, avail_min), 1)
        n_hard_eff = min(n_hard, pool_k)
        _, topk_idx  = neg_probs.topk(pool_k, dim=1)
        rand_sel     = torch.randint(0, pool_k, (B, n_hard_eff), device=device)
        hard_neg_idx = topk_idx.gather(1, rand_sel)

        rand_neg_idx = torch.randint(0, L, (B, n_rand), device=device)
        neg_idx      = torch.cat([hard_neg_idx, rand_neg_idx], dim=1)
        valid_mask   = tgt_norm.gather(1, neg_idx) < self.neg_thresh

        neg_val     = log_p.gather(1, neg_idx)
        hinge       = F.relu(neg_val - contact_val.unsqueeze(1) + self.margin) * valid_mask.float()
        valid_count = valid_mask.float().sum(dim=1).clamp(min=1.0)
        return (hinge.sum(dim=1) / valid_count).mean()


class AdaptiveSigmaScheduler:
    def __init__(self, sigma_max=40, sigma_min=5, total_epochs=100, warmup_epochs=2):
        self.sigma_max     = sigma_max
        self.sigma_min     = sigma_min
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_sigma(self, epoch):
        if epoch < self.warmup_epochs:
            return self.sigma_max
        progress = min(
            (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs),
            1.0)
        return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (
            1 + math.cos(math.pi * progress))


class ImprovedCombinedLoss(nn.Module):
    def __init__(self,
                 target_length=4096,
                 heatmap_weight=1.0,
                 ordinal_weight=0.15,
                 sharpness_radius=40,
                 ordinal_n_neg=20, ordinal_margin=4.0, ordinal_neg_thresh=0.1,
                 sharpness_weight=0.5,
                 soft_argmax_weight=None, regression_weight=None):
        super().__init__()
        self.target_length   = target_length
        self.heatmap_loss_fn = HeatmapLoss(ce_weight=1.0)
        self.ordinal_loss_fn = OrdinalRankingLoss(
            n_neg=ordinal_n_neg, margin=ordinal_margin, neg_thresh=ordinal_neg_thresh)
        self.w_heatmap = heatmap_weight
        self.w_ordinal = ordinal_weight

    @staticmethod
    def _compute_peak_weight(current_sigma, sigma_max, sigma_min):
        if sigma_max <= sigma_min:
            return 1.0
        ratio = float(np.clip((sigma_max - current_sigma) / (sigma_max - sigma_min),
                               0.0, 1.0))
        return 0.5 * (1.0 - math.cos(math.pi * ratio))

    def forward(self, pred_heatmap,
                target_heatmap, target_position,
                current_sigma=None, sigma_max=None, sigma_min=None, **kwargs):
        peak_weight = (self._compute_peak_weight(current_sigma, sigma_max,
                                                  sigma_min if sigma_min is not None else 5.0)
                       if current_sigma is not None and sigma_max is not None else 1.0)

        loss_hm = self.heatmap_loss_fn(
            pred_heatmap, target_heatmap,
            peak_weight=peak_weight,
            current_sigma=current_sigma)

        loss_ord = self.ordinal_loss_fn(pred_heatmap, target_position, target_heatmap)

        total_loss = self.w_heatmap * loss_hm + self.w_ordinal * loss_ord

        if torch.isnan(total_loss):
            print("[WARNING] NaN in loss!")
            total_loss = torch.zeros((), device=pred_heatmap.device, requires_grad=True)

        losses = {
            'heatmap':       loss_hm,
            'ordinal':       loss_ord,
            'eff_w_heatmap': self.w_heatmap,
            'eff_w_ordinal': self.w_ordinal,
            'peak_weight':   peak_weight,
        }
        return total_loss, losses

    def epoch_end_update(self, *args, **kwargs):
        pass


CombinedLossV2     = ImprovedCombinedLoss
SimpleCombinedLoss = ImprovedCombinedLoss


if __name__ == "__main__":
    B, L = 4, 4096
    print("=== KL-only + OrdinalRanking 消融版损失测试 ===")
    for sigma in [40, 20, 5]:
        pred_hm  = torch.softmax(torch.randn(B, L), dim=1)
        tgt_idx  = torch.randint(1000, 3000, (B,))
        tgt_hm   = generate_heatmap_batch(tgt_idx, L, sigma, 'cpu')
        tgt_pos  = tgt_idx.float() / (L - 1)
        fn = ImprovedCombinedLoss(target_length=L)
        total, losses = fn(pred_hm, tgt_hm, tgt_pos,
                           current_sigma=sigma, sigma_max=40, sigma_min=5)
        print(f"sigma={sigma:2d}: total={total.item():.4f} "
              f"hm(kl_only)={losses['heatmap'].item():.4f} "
              f"ord={losses['ordinal'].item():.4f} "
              f"peak_w={losses['peak_weight']:.2f}")
    print("OK!")
