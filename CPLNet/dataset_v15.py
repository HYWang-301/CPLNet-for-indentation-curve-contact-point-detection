import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.signal import savgol_filter, resample_poly
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[WARN] scipy not installed; falling back to np.interp / np.gradient.")


def parse_group_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if '_' in base:
        return base.rsplit('_', 1)[0]
    return base


class AFMDataset(Dataset):
    def __init__(self, file_list, labels_dict, data_dir,
                 target_length=8192, augment=False,
                 cutout_prob=0.6, resample_mode='poly'):
        self.file_list      = file_list
        self.labels_dict    = labels_dict
        self.data_dir       = data_dir
        self.target_length  = target_length
        self.augment        = augment
        self.cutout_prob    = cutout_prob
        self.resample_mode  = resample_mode
        self._lengths_cache = {}

    def __len__(self):
        return len(self.file_list)

    def load_curve(self, filename):
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
                    continue
        data = np.array(data)
        defl = data[:len(data) // 2]
        self._lengths_cache[filename] = len(defl)
        return defl

    def _get_original_length(self, filename):
        if filename in self._lengths_cache:
            return self._lengths_cache[filename]
        filepath = os.path.join(self.data_dir, filename)
        count = 0
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        float(line)
                        count += 1
                    except ValueError:
                        continue
        n = count // 2
        self._lengths_cache[filename] = n
        return n

    def _resample_1d(self, data, target_len):
        orig_len = len(data)
        if orig_len == target_len:
            return data.copy()
        if self.resample_mode == 'linear' or not _HAS_SCIPY:
            return np.interp(np.linspace(0, 1, target_len),
                             np.linspace(0, 1, orig_len), data)
        if orig_len > target_len:
            from math import gcd
            g = gcd(orig_len, target_len)
            out = resample_poly(data, target_len // g, orig_len // g)
            return out[:target_len]
        else:
            x_old = np.linspace(0, 1, orig_len)
            x_new = np.linspace(0, 1, target_len)
            return PchipInterpolator(x_old, data)(x_new)

    def resample(self, data, original_contact_idx, original_length):
        L       = self.target_length
        data_rs = self._resample_1d(data, L)
        denom   = max(original_length - 1, 1)
        new_idx = int(np.clip(round(original_contact_idx * (L - 1) / denom), 0, L - 1))
        return data_rs, new_idx

    def resample_signal_only(self, data):
        return self._resample_1d(data, self.target_length)

    def normalize(self, data):
        mean = np.mean(data)
        std  = np.std(data)
        if std < 1e-8:
            std = 1.0
        return (data - mean) / std

    def compute_smooth_derivative(self, data, poly=3):
        window = 11
        if not _HAS_SCIPY or len(data) < window + 1:
            return np.gradient(data)
        try:
            return savgol_filter(data, window_length=window, polyorder=poly, deriv=1)
        except Exception:
            return np.gradient(data)

    def add_gaussian_noise(self, data, scale_range=(0.005, 0.02)):
        noise_level = np.random.uniform(*scale_range) * np.std(data)
        return data + np.random.normal(0, noise_level, len(data))

    def add_linear_drift(self, data):
        slope = np.random.uniform(-0.00005, 0.00005)
        x     = np.arange(len(data)) - (len(data) - 1) / 2.0
        return data + slope * x

    def cutout_1d(self, data, n_holes=3, max_len_frac=0.08):
        data   = data.copy()
        length = len(data)
        max_hl = max(1, int(length * max_len_frac))
        for _ in range(n_holes):
            hl    = np.random.randint(1, max_hl + 1)
            if length - hl - 2 <= 1:
                continue
            start = np.random.randint(1, length - hl - 1)
            left  = data[start - 1]
            right = data[start + hl]
            data[start:start + hl] = np.linspace(left, right, hl + 2)[1:-1]
        return data

    @staticmethod
    def _shift_pad(arr, shift):
        n = len(arr)
        if n < 2:
            return arr
        if shift > 0:
            slope = arr[1] - arr[0]
            pad   = arr[0] - slope * np.arange(shift, 0, -1)
            return np.concatenate([pad, arr[:-shift]])
        else:
            s     = -shift
            slope = arr[-1] - arr[-2]
            pad   = arr[-1] + slope * np.arange(1, s + 1)
            return np.concatenate([arr[s:], pad])

    def random_time_shift(self, deflection, contact_idx, max_shift_frac=0.03):
        length    = len(deflection)
        max_shift = int(length * max_shift_frac)
        if max_shift <= 0:
            return deflection, contact_idx
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return deflection, contact_idx
        new_idx = contact_idx + shift
        if not (0 <= new_idx < length):
            return deflection, contact_idx
        return self._shift_pad(deflection, shift), new_idx

    def augment_smooth(self, deflection, contact_idx):
        if np.random.random() < 0.7:
            deflection = self.add_gaussian_noise(deflection)
        if np.random.random() < 0.5:
            deflection = self.add_linear_drift(deflection)
        if np.random.random() < 0.3:
            deflection, contact_idx = self.random_time_shift(
                deflection, contact_idx, max_shift_frac=0.03)
        return deflection, contact_idx

    def __getitem__(self, idx):
        filename    = self.file_list[idx]
        contact_idx = self.labels_dict[filename]

        deflection = self.load_curve(filename)
        original_length = len(deflection)

        deflection, new_contact_idx = self.resample(
            deflection, contact_idx, original_length)

        if self.augment:
            deflection, new_contact_idx = self.augment_smooth(
                deflection, new_contact_idx)
            new_contact_idx = int(np.clip(new_contact_idx, 0, self.target_length - 1))

        if self.augment and self.cutout_prob > 0 and np.random.random() < self.cutout_prob:
            deflection = self.cutout_1d(deflection, n_holes=3, max_len_frac=0.08)

        d_deflection = self.compute_smooth_derivative(deflection)

        deflection   = self.normalize(deflection)
        d_deflection = self.normalize(d_deflection)

        position_label = new_contact_idx / max(self.target_length - 1, 1)

        signal_tensor = torch.tensor(
            np.stack([deflection, d_deflection], axis=0),
            dtype=torch.float32)

        return {
            'signal':                signal_tensor,
            'position':              torch.tensor(position_label, dtype=torch.float32),
            'filename':              filename,
            'original_length':       original_length,
            'original_contact_idx':  self.labels_dict[filename],
            'resampled_contact_idx': new_contact_idx,
        }


def make_stratified_sampler(dataset, n_bins=20):
    positions = []
    for f in dataset.file_list:
        orig_len = dataset._get_original_length(f)
        denom    = max(orig_len - 1, 1)
        positions.append(np.clip(dataset.labels_dict[f] / denom, 0.0, 1.0))
    positions  = np.asarray(positions)
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_ids    = np.digitize(positions, bin_edges[1:-1])
    bin_counts = np.maximum(np.bincount(bin_ids, minlength=n_bins), 1)
    weights    = torch.tensor(1.0 / bin_counts[bin_ids], dtype=torch.float32)
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    return {str(row['FileName']): int(row['ContactIndex']) for _, row in df.iterrows()}


def _group_split(files, test_size, seed):
    groups   = np.array([parse_group_id(f) for f in files])
    n_groups = len(set(groups))
    print(f"[INFO] Group-aware split: {n_groups} groups, {len(files)} files "
          f"(avg {len(files)/max(n_groups,1):.1f} files/group)")
    if n_groups < 2 or n_groups <= int(round(1.0 / max(test_size, 1e-6))):
        print("[WARN] Groups too few for GroupShuffleSplit, falling back to random split.")
        return train_test_split(files, test_size=test_size, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(files, groups=groups))
    files = np.asarray(files)
    return list(files[tr_idx]), list(files[te_idx])


def create_dataloaders(data_dir, labels_excel,
                       batch_size=32, target_length=4096,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       num_workers=0, random_seed=42,
                       use_stratified_sampler=True, stratified_bins=20,
                       group_aware_split=True, **kwargs):

    labels_dict = load_labels(labels_excel)
    all_files   = list(labels_dict.keys())
    valid_files = [f for f in all_files if os.path.exists(os.path.join(data_dir, f))]
    print(f"Total valid files: {len(valid_files)}")

    np.random.seed(random_seed)
    split_fn = _group_split if group_aware_split else (
        lambda fs, ts, sd: train_test_split(fs, test_size=ts, random_state=sd))

    train_val_files, test_files = split_fn(valid_files, test_ratio, random_seed)
    val_size = val_ratio / (train_ratio + val_ratio)
    train_files, val_files = split_fn(train_val_files, val_size, random_seed)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    ckw = dict(labels_dict=labels_dict, data_dir=data_dir, target_length=target_length)
    cutout_prob   = kwargs.get('cutout_prob', 0.6)
    resample_mode = kwargs.get('resample_mode', 'poly')
    dkw = dict(cutout_prob=cutout_prob, resample_mode=resample_mode)
    train_dataset = AFMDataset(train_files, augment=True,  **ckw, **dkw)
    val_dataset   = AFMDataset(val_files,   augment=False, **ckw, **dkw)
    test_dataset  = AFMDataset(test_files,  augment=False, **ckw, **dkw)

    use_pin = torch.cuda.is_available()
    lkw = dict(num_workers=num_workers, pin_memory=use_pin)
    if num_workers > 0:
        lkw['persistent_workers'] = True
        lkw['prefetch_factor']    = 4

    if use_stratified_sampler:
        sampler      = make_stratified_sampler(train_dataset, n_bins=stratified_bins)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, **lkw)
        print(f"[INFO] Stratified sampler: {stratified_bins} bins")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, **lkw)

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, **lkw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **lkw)
    return train_loader, val_loader, test_loader, test_files, labels_dict
