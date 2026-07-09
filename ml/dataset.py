"""
Dataset для TorchSig 2.0.
- Кэш в .npy (без h5py)
    <stem>_data.npy   : float32 [N, 2, num_iq_samples]
    <stem>_labels.npy : int32   [N]
- Файлы создаются сразу на диске через memmap (как HDF5)
- Train и Val генерируются отдельно
- Аугментации только на train
- Визуализация созвездий после генерации
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc

import config.settings as cfg

try:
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.datasets.dataset_metadata import DatasetMetadata
    TORCHSIG_AVAILABLE = True
    print("[✓] TorchSig 2.0 успешно импортирован")
except ImportError as e:
    TORCHSIG_AVAILABLE = False
    print(f"[✗] TorchSig 2.0 не найден: {e}")

# SignalTransform
class SignalTransform:
    def __init__(self, modulation_list):
        self.mod_to_idx = {m: i for i, m in enumerate(modulation_list)}

    _debug_printed = False

    def __call__(self, signal):
        if not SignalTransform._debug_printed:
            SignalTransform._debug_printed = True
            try:
                print("\n[DEBUG] type(signal):", type(signal).__name__)
                print("[DEBUG] attrs:", [a for a in dir(signal) if not a.startswith('_')])
                for attr in ('iq_data', 'data', 'samples', 'metadata',
                             'component_signals', 'class_name', 'label'):
                    if hasattr(signal, attr):
                        val = getattr(signal, attr)
                        print(f"[DEBUG]   signal.{attr} = "
                              f"{type(val).__name__}: {repr(val)[:200]}")
                cs = getattr(signal, 'component_signals', None)
                if cs is not None and len(cs) > 0:
                    comp = cs[0]
                    print("[DEBUG] component_signals[0] attrs:",
                          [a for a in dir(comp) if not a.startswith('_')])
                    meta = getattr(comp, 'metadata', None)
                    if meta is not None:
                        for a in dir(meta):
                            if not a.startswith('_'):
                                try:
                                    print(f"[DEBUG]     metadata.{a} ="
                                          f" {repr(getattr(meta, a))[:120]}")
                                except Exception:
                                    pass
                print("[DEBUG] --- end ---\n")
            except Exception as e:
                print(f"[DEBUG] crashнул: {e}\n")

        #IQ data 
        raw = getattr(signal, 'iq_data', None)
        if raw is None: raw = getattr(signal, 'data',    None)
        if raw is None: raw = getattr(signal, 'samples', None)
        if raw is None: raw = np.zeros(cfg.NUM_IQ_SAMPLES, dtype=np.complex64)

        if hasattr(raw, '__len__') and len(raw) != cfg.NUM_IQ_SAMPLES:
            if len(raw) > cfg.NUM_IQ_SAMPLES:
                raw = raw[:cfg.NUM_IQ_SAMPLES]
            else:
                pad = np.zeros(cfg.NUM_IQ_SAMPLES - len(raw), dtype=np.complex64)
                raw = np.concatenate([np.asarray(raw), pad])

        if isinstance(raw, np.ndarray) and np.iscomplexobj(raw):
            c      = torch.from_numpy(raw.astype(np.complex64))
            tensor = torch.stack([c.real, c.imag], dim=0)
        else:
            tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
            if tensor.dim() == 1:
                tensor = tensor.view(2, -1)

        #Label
        class_name = None
        try:
            cs = getattr(signal, 'component_signals', None)
            if cs is not None and len(cs) > 0:
                class_name = cs[0].metadata.class_name
        except Exception:
            pass
        if class_name is None:
            try:   class_name = signal.metadata.class_name
            except Exception: pass
        if class_name is None: class_name = getattr(signal, 'class_name', None)
        if class_name is None: class_name = getattr(signal, 'label',      None)
        if isinstance(class_name, (list, tuple)):
            class_name = class_name[0] if len(class_name) > 0 else None

        label = self.mod_to_idx.get(str(class_name), 0) if class_name is not None else 0
        return tensor, label

# NPY helpers
def _npy_stem(impairment_level, snr_min, snr_max, num_samples, num_classes, split):
    return (f"ts_{split}"
            f"_lvl{int(impairment_level)}"
            f"_snr{int(snr_min)}to{int(snr_max)}"
            f"_n{num_samples}_c{num_classes}")


def _npy_paths(cache_dir, stem):
    return (os.path.join(cache_dir, f"{stem}_data.npy"),
            os.path.join(cache_dir, f"{stem}_labels.npy"))


def _npy_exists(cache_dir, stem):
    dp, lp = _npy_paths(cache_dir, stem)
    return os.path.exists(dp) and os.path.exists(lp)


def _load_memmap_data(path, num_iq_samples):
    """
    Загружает data-файл созданный через np.memmap (float32, raw binary).
    Форма: [N, 2, num_iq_samples]
    """
    raw  = np.memmap(path, dtype=np.float32, mode='r')
    n    = raw.size // (2 * num_iq_samples)
    return raw.reshape(n, 2, num_iq_samples)


def _load_memmap_labels(path):
    """
    Загружает labels-файл созданный через np.memmap (int32, raw binary).
    Форма: [N]
    """
    return np.memmap(path, dtype=np.int32, mode='r')


# Dataset

class CachedTorchSigDataset(Dataset):
    """
    Читает из двух бинарных файлов созданных np.memmap.
    Используем np.memmap для загрузки (не np.load — файлы без .npy заголовка).
    .copy() обязателен: mmap read-only, torch не принимает.
    """

    def __init__(self, data_path, labels_path, modulations, augment=False):
        self.modulations = modulations
        self.augment     = augment
        self.data        = _load_memmap_data(data_path, cfg.NUM_IQ_SAMPLES)
        self.labels      = _load_memmap_labels(labels_path)
        assert len(self.data) == len(self.labels), (
            f"Mismatch: data {len(self.data)} vs labels {len(self.labels)}")

    @staticmethod
    def _augment(t):
        N = t.shape[1]

        if np.random.rand() < 0.7:
            df   = np.random.uniform(-0.15, 0.15)
            phi  = 2 * np.pi * df * torch.arange(N).float()
            c, s = torch.cos(phi), torch.sin(phi)
            t    = torch.stack([c*t[0] - s*t[1], s*t[0] + c*t[1]])

        if np.random.rand() < 0.7:
            shift = np.random.randint(-N // 3, N // 3)
            t = torch.roll(t, shift, dims=1)

        if np.random.rand() < 0.7:
            a    = np.random.uniform(0, 2 * np.pi)
            c, s = float(np.cos(a)), float(np.sin(a))
            t    = torch.stack([c*t[0] - s*t[1], s*t[0] + c*t[1]])

        if np.random.rand() < 0.6:
            t = t * float(np.random.uniform(0.6, 1.4))

        if np.random.rand() < 0.7:
            sp  = t.pow(2).mean().clamp(min=1e-10)
            snr = float(np.random.uniform(10, 30))
            t   = t + torch.randn_like(t) * (sp / 10 ** (snr / 10)).sqrt()

        return t

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data  = torch.from_numpy(self.data[idx].copy())
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.augment:
            data = self._augment(data)
        return data, label



# Визуализация
def visualize_dataset(data_path, labels_path, modulations, snr_min, snr_max, tag=''):
    num_classes = len(modulations)
    print(f"[*] Визуализация ({tag}): {num_classes} классов…")
    try:
        # Используем memmap — файлы созданы без .npy заголовка
        data   = _load_memmap_data(data_path, cfg.NUM_IQ_SAMPLES)
        labels = _load_memmap_labels(labels_path)

        collected = {}
        for idx in np.random.permutation(len(labels)):
            lbl = int(labels[idx])
            if lbl not in collected:
                collected[lbl] = data[idx].copy()
            if len(collected) == num_classes:
                break

        found = len(collected)
        print(f"    Найдено {found}/{num_classes} классов")

        cols = 8
        rows = (found + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
        axes = axes.flatten()

        for plot_idx, label in enumerate(sorted(collected.keys())):
            ax   = axes[plot_idx]
            sig  = collected[label]
            name = modulations[label]
            ax.scatter(sig[0], sig[1], s=1, alpha=0.5, c='royalblue')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.set_title(name, fontsize=7, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

        for i in range(found, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f'Constellation Diagrams [{tag}] — SNR {snr_min:.0f}–{snr_max:.0f} dB',
            fontsize=13, fontweight='bold')
        plt.tight_layout()

        out_dir  = os.path.dirname(data_path) or '.'
        out_path = os.path.join(out_dir,
            f'viz_{tag}_snr{int(snr_min)}to{int(snr_max)}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[✓] Визуализация: {out_path}")

    except Exception as e:
        print(f"[!] Ошибка визуализации: {e}")


# Генерация + memmap кэш (файлы появляются СРАЗУ на диске)
def generate_torchsig_dataset(modulations, impairment_level, num_samples,
                               num_iq_samples, snr_min, snr_max, split='train'):
    """
    Генерирует датасет TorchSig и сохраняет через np.memmap.
    Файлы *_data.npy и *_labels.npy появляются на диске СРАЗУ
    и растут по мере генерации — как HDF5.

    Возвращает: (data_path, labels_path)
    """
    if not TORCHSIG_AVAILABLE:
        raise ImportError("TorchSig 2.0 не установлен!")

    ts_level  = int(np.clip(impairment_level, 0, 2))
    cache_dir = cfg.DATASET_CONFIG['save_dir']
    os.makedirs(cache_dir, exist_ok=True)

    stem                   = _npy_stem(ts_level, snr_min, snr_max,
                                       num_samples, len(modulations), split)
    data_path, labels_path = _npy_paths(cache_dir, stem)

    if _npy_exists(cache_dir, stem):
        print(f"[✓] Кэш [{split}] найден: {stem}")
        return data_path, labels_path

    print(f"\n{'='*70}")
    print(f" ГЕНЕРАЦИЯ [{split.upper()}] ДАТАСЕТА")
    print(f"{'='*70}")
    print(f"  Модуляций: {len(modulations)}  |  Сэмплов: {num_samples:,}")
    print(f"  Level: {ts_level}  |  SNR: {snr_min}–{snr_max} dB")
    print(f"  Файлы появятся на диске СРАЗУ (memmap)")
    print(f"{'='*70}\n")

    sr   = 200e3
    meta = DatasetMetadata(
        sample_rate=sr,
        num_iq_samples_dataset=num_iq_samples,
        fft_size=256,
        num_signals_min=1, num_signals_max=1,
        snr_db_min=snr_min,  snr_db_max=snr_max,
        signal_duration_min=0.9 * num_iq_samples / sr,
        signal_duration_max=1.0 * num_iq_samples / sr,
        signal_bandwidth_min=sr / 4, signal_bandwidth_max=sr / 2,
        cochannel_overlap_probability=0,
        class_list=modulations,
        level=ts_level,
    )

    ds = TorchSigIterableDataset(dataset_metadata=meta,
                                  transforms=[SignalTransform(modulations)])
    it = iter(ds)

    print(f"[*] Создаём файлы на диске (memmap)…")
    buf_data = np.memmap(
        data_path, dtype=np.float32, mode='w+',
        shape=(num_samples, 2, num_iq_samples))
    buf_labels = np.memmap(
        labels_path, dtype=np.int32, mode='w+',
        shape=(num_samples,))
    print(f"[✓] Файлы созданы:")
    print(f"    {data_path}")
    print(f"    {labels_path}")
    print(f"[*] Генерация сэмплов…\n")

    FLUSH_EVERY = 50_000

    for i in tqdm(range(num_samples), desc=f"Генерация [{split}]", unit="samples"):
        try:
            tensor, label = next(it)
        except StopIteration:
            it = iter(ds)
            tensor, label = next(it)

        buf_data[i]   = tensor.numpy()
        buf_labels[i] = int(label)

        if (i + 1) % FLUSH_EVERY == 0:
            buf_data.flush()
            buf_labels.flush()
            gc.collect()

    buf_data.flush()
    buf_labels.flush()
    del buf_data, buf_labels
    gc.collect()

    print(f"\n[✓] [{split}] генерация завершена")
    print(f"    {data_path}")
    print(f"    {labels_path}\n")

    return data_path, labels_path


# DataLoaders
def get_dataloaders(batch_size, impairment_level=0, snr_min=20.0, snr_max=40.0,
                    augment_train=True):
    """
    Возвращает (train_loader, val_loader, modulations).
    Train и Val генерируются независимо — честная валидация.
    """
    modulations = cfg.SELECTED_MODS
    total       = cfg.DATASET_CONFIG['num_samples']
    n_phases    = len(cfg.CURRICULUM_PHASES)
    val_ratio   = cfg.DATASET_CONFIG['val_ratio']

    train_n = total // n_phases
    val_n   = int(train_n * val_ratio)

    print(f"\n{'='*70}")
    print(" СОЗДАНИЕ DATALOADERS")
    print(f"{'='*70}")
    print(f"  Train: {train_n:,} сэмплов"
          f" (Level {impairment_level}, SNR {snr_min:.0f}–{snr_max:.0f} dB)")
    print(f"  Val:   {val_n:,} сэмплов  (независимая генерация)")
    print(f"{'='*70}\n")

    train_data, train_labels = generate_torchsig_dataset(
        modulations, impairment_level, train_n,
        cfg.NUM_IQ_SAMPLES, snr_min, snr_max, split='train')

    val_data, val_labels = generate_torchsig_dataset(
        modulations, impairment_level, val_n,
        cfg.NUM_IQ_SAMPLES, snr_min, snr_max, split='val')

    ds_train = CachedTorchSigDataset(
        train_data, train_labels, modulations, augment=augment_train)
    ds_val   = CachedTorchSigDataset(
        val_data, val_labels, modulations, augment=False)

    nw      = cfg.DATASET_CONFIG['num_workers']
    pin     = cfg.DATASET_CONFIG['pin_memory'] and torch.cuda.is_available()
    persist = nw > 0

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin,
        persistent_workers=persist, drop_last=True,
        prefetch_factor=2 if nw > 0 else None)

    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=persist,
        prefetch_factor=2 if nw > 0 else None)

    print(f"  Train: {len(ds_train):,} сэмплов → {len(train_loader)} батчей")
    print(f"  Val:   {len(ds_val):,} сэмплов  → {len(val_loader)} батчей")
    print(f"{'='*70}\n")

    return train_loader, val_loader, modulations
