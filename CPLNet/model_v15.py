import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def windowed_soft_argmax(heatmap, coords, window=100):
    B, L = heatmap.shape
    with torch.no_grad():
        peak = heatmap.argmax(dim=1)
    idx  = torch.arange(L, device=heatmap.device).unsqueeze(0)
    pk   = peak.unsqueeze(1)
    mask  = (idx >= (pk - window).clamp(0)) & (idx <= (pk + window).clamp(max=L - 1))
    masked = heatmap.masked_fill(~mask, 0.0)
    w = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (w * coords.unsqueeze(0)).sum(dim=1)


class WindowedSoftArgmax(nn.Module):
    def __init__(self, length=8192, window=100):
        super().__init__()
        self.window = window
        self.register_buffer('coords', torch.linspace(0, 1, length))

    def forward(self, heatmap):
        return windowed_soft_argmax(heatmap, self.coords, self.window)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand      = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * torch.floor(rand + keep_prob) / keep_prob


class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        assert out_channels % 4 == 0
        bc = out_channels // 4
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, bc, 3,  padding=1),  nn.BatchNorm1d(bc), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, bc, 7,  padding=3),  nn.BatchNorm1d(bc), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, bc, 15, padding=7),  nn.BatchNorm1d(bc), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.Conv1d(in_channels, bc, 31, padding=15), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.bn   = nn.BatchNorm1d(out_channels)
        self.se   = SEBlock1d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.drop(self.se(self.bn(out)))


class SingleScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, kernel_size=13):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1  = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU(True)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.se   = SEBlock1d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.bn1(self.conv(x)))
        return self.drop(self.se(self.bn(out)))


def _make_msblock(in_channels, out_channels, dropout=0.1, multiscale=True):
    if multiscale:
        return MultiScaleCNNBlock(in_channels, out_channels, dropout=dropout)
    return SingleScaleCNNBlock(in_channels, out_channels, dropout=dropout)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, drop_path=0.15, dropout=0.18, use_se=False):
        super().__init__()
        self.conv1     = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1       = nn.BatchNorm1d(channels)
        self.conv2     = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2       = nn.BatchNorm1d(channels)
        self.relu      = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path)
        self.dropout   = nn.Dropout(dropout)
        self.se        = SEBlock1d(channels) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.dropout(residual)
        residual = self.se(self.bn2(self.conv2(residual)))
        return self.relu(x + self.drop_path(residual))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=96, output_channels=256,
                 drop_path=0.15, dropout=0.18, use_multiscale=True):
        super().__init__()
        half = base_channels // 2
        self.stage1 = nn.Sequential(
            _make_msblock(in_channels, half, dropout=dropout, multiscale=use_multiscale),
            DilatedResidualBlock(half, dilation=1, drop_path=drop_path, dropout=dropout, use_se=False),
            DilatedResidualBlock(half, dilation=2, drop_path=drop_path, dropout=dropout, use_se=False))
        self.stage2 = nn.Sequential(
            _make_msblock(half, base_channels, dropout=dropout, multiscale=use_multiscale),
            DilatedResidualBlock(base_channels, dilation=4,  drop_path=drop_path, dropout=dropout, use_se=False),
            DilatedResidualBlock(base_channels, dilation=8,  drop_path=drop_path, dropout=dropout, use_se=False),
            DilatedResidualBlock(base_channels, dilation=16, drop_path=drop_path, dropout=dropout, use_se=False),
            DilatedResidualBlock(base_channels, dilation=32, drop_path=drop_path, dropout=dropout, use_se=True))
        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2), nn.ReLU(True),
            SEBlock1d(base_channels * 2), nn.Dropout(dropout))
        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, output_channels, 4, stride=2, padding=1),
            nn.BatchNorm1d(output_channels), nn.ReLU(True),
            SEBlock1d(output_channels), nn.Dropout(dropout))

    def forward(self, x):
        s1        = self.stage1(x)
        feat_full = self.stage2(s1)
        feat_mid  = self.down1(feat_full)
        feat_down = self.down2(feat_mid)
        return feat_down, feat_mid, feat_full


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=768, dropout=0.18):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        x = self.pos_encoder(x.permute(0, 2, 1))
        return self.transformer(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(True), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(True),
            SEBlock1d(out_channels))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    def __init__(self, transformer_dim=256, mid_channels=192, base_channels=96, dropout=0.1):
        super().__init__()
        self.up1 = UpsampleBlock(transformer_dim, mid_channels, mid_channels, dropout)
        self.up2 = UpsampleBlock(mid_channels,    base_channels, base_channels, dropout)

    def forward(self, x, feat_mid, feat_full):
        x = self.up1(x, feat_mid)
        x = self.up2(x, feat_full)
        return x


class BoundaryAwareModule(nn.Module):
    def __init__(self, channels, window=64, dropout=0.1):
        super().__init__()
        self.window = window
        self.pool   = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        w = self.window
        left_ctx  = self.pool(F.pad(x, (w - 1, 0), mode='replicate'))
        right_ctx = self.pool(F.pad(x, (0, w - 1), mode='replicate'))
        return self.fusion(torch.cat([x, left_ctx, right_ctx], dim=1))


class HeatmapHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm1d(hidden_channels), nn.ReLU(True), nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2), nn.ReLU(True),
            nn.Conv1d(hidden_channels // 2, 1, 1))

    def forward(self, x):
        logits = self.head(x).squeeze(1)
        return F.softmax(logits, dim=1), logits


class AFMContactDetector(nn.Module):
    def __init__(self, in_channels=2, cnn_base_channels=96,
                 transformer_d_model=256, transformer_nhead=8,
                 transformer_layers=2, target_length=4096,
                 drop_path=0.15, dropout=0.18,
                 boundary_window=64, softargmax_window=100,
                 use_transformer=True, use_boundary=True, use_multiscale=True):
        super().__init__()
        self.target_length   = target_length
        self.use_transformer = use_transformer
        self.use_boundary    = use_boundary
        mid_channels = cnn_base_channels * 2
        self.cnn_encoder = CNNEncoder(
            in_channels=in_channels, base_channels=cnn_base_channels,
            output_channels=transformer_d_model, drop_path=drop_path, dropout=dropout,
            use_multiscale=use_multiscale)
        self.transformer = TransformerEncoder(
            d_model=transformer_d_model, nhead=transformer_nhead,
            num_layers=transformer_layers, dim_feedforward=768, dropout=dropout
        ) if use_transformer else None
        self.decoder = Decoder(
            transformer_dim=transformer_d_model, mid_channels=mid_channels,
            base_channels=cnn_base_channels, dropout=dropout)
        self.boundary_module = BoundaryAwareModule(
            channels=cnn_base_channels, window=boundary_window, dropout=dropout
        ) if use_boundary else None
        self.heatmap_head = HeatmapHead(
            in_channels=cnn_base_channels, hidden_channels=cnn_base_channels, dropout=dropout)
        self.windowed_softargmax = WindowedSoftArgmax(
            length=target_length, window=softargmax_window)

    def forward(self, x, return_logits=False, compute_position=True):
        feat_down, feat_mid, feat_full = self.cnn_encoder(x)
        if self.transformer is not None:
            trans_out = self.transformer(feat_down).permute(0, 2, 1)
        else:
            trans_out = feat_down
        decoded   = self.decoder(trans_out, feat_mid, feat_full)
        refined   = self.boundary_module(decoded) if self.boundary_module is not None else decoded
        heatmap_prob, heatmap_logits = self.heatmap_head(refined)
        position  = (self.windowed_softargmax(heatmap_prob).unsqueeze(1)
                     ) if compute_position else None
        if return_logits:
            return heatmap_prob, position, heatmap_logits
        return heatmap_prob, position

    def predict_contact_point(self, x):
        self.eval()
        with torch.no_grad():
            _, position = self.forward(x)
            idx = (position.squeeze(1) * (self.target_length - 1)).round().long()
            return idx.clamp(0, self.target_length - 1)


if __name__ == "__main__":
    model = AFMContactDetector(in_channels=2, target_length=4096)
    x = torch.randn(4, 2, 4096)
    hm, pos = model(x)
    print(f"Heatmap: {hm.shape}  Position: {pos.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("OK!")
