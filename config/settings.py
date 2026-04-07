"""
Конфигурация XCiT Small — RF модуляции
21M сэмплов / 57 классов / 3 фазы curriculum
"""

import torch
import numpy as np
import random

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    'num_samples':    5000000,   # делится на 3 фазы и на 57 классов без остатка
    'val_ratio':      0.20,         # 20% → отдельно сгенерированный val set
    'num_iq_samples': 1024,
    'num_classes':    57,
    'save_dir':       './torchsig_datasets',
    'num_workers':    10,           # все 10 ядер для DataLoader
    'pin_memory':     True,
}


TORCHSIG_ALL_MODS = [
    'am-lsb', 'am-usb',
    'ook',
    'bpsk', 'qpsk',
    'fm',
    '2fsk', '2gfsk', '2msk', '2gmsk',
]

SELECTED_MODS  = TORCHSIG_ALL_MODS
NUM_CLASSES    = len(SELECTED_MODS)    # 57
NUM_IQ_SAMPLES = DATASET_CONFIG['num_iq_samples']


# ── Auto batch ────────────────────────────────────────────────────────────────
def _auto_batch():
    if not torch.cuda.is_available():
        return 8
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem < 4:  return 16
    if mem < 8:  return 32
    if mem >= 8: return 128   # RTX 4070 Ti 12GB → 128
    return 128


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'model_name':     'xcit_small_12_p16_224',
    'num_classes':    NUM_CLASSES,
    'pretrained':     False,
    'drop_path_rate': 0.15,
}


# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    'batch_size':                  _auto_batch(),
    'gradient_accumulation_steps': 4 if torch.cuda.is_available() else 8,
    'use_mixed_precision':         torch.cuda.is_available(),

    'learning_rate':  1e-4,
    'epochs':         30,
    'device':         torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers':    DATASET_CONFIG['num_workers'],

    'weight_decay':   0.05,
    'betas':          (0.9, 0.999),

    'mixup_alpha':         0.2,
    'mixup_warmup_epoch':  5,
    'label_smoothing':     0.05,

    'warmup_epochs':  5,
    'lr_schedule':    'cosine',
    'min_lr':         1e-6,

    'gradient_clip_norm':          1.0,
    'early_stopping_patience':     30,
    'save_frequency':              5,
    'empty_cache_every_n_batches': 500,
}


# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM_PHASES = [
    {
        'name':        'Phase 1: Clean signals (high SNR)',
        'start_epoch': 0,  'end_epoch': 9,
        'impairment':  0,  'snr_min': 20, 'snr_max': 40,
    },
    {
        'name':        'Phase 2: Medium impairments',
        'start_epoch': 10, 'end_epoch': 19,
        'impairment':  1,  'snr_min': 10,  'snr_max': 20,
    },
    {
        'name':        'Phase 3: Strong impairments',
        'start_epoch': 20, 'end_epoch': 29,
        'impairment':  2,  'snr_min': 0, 'snr_max': 10,
    },
]


# ── Info ──────────────────────────────────────────────────────────────────────
def print_config():
    total     = DATASET_CONFIG['num_samples']
    n_phases  = len(CURRICULUM_PHASES)
    per_phase = total // n_phases
    val_n     = int(per_phase * DATASET_CONFIG['val_ratio'])

    print(f"\n{'='*70}")
    print(" XCiT Small  —  RF Modulation Classification")
    print(f"{'='*70}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:    {torch.cuda.get_device_name(0)}  ({mem:.1f} GB)")
    else:
        print("  Device: CPU")
    print(f"  Batch:  {TRAIN_CONFIG['batch_size']}"
          f" × {TRAIN_CONFIG['gradient_accumulation_steps']} accum ="
          f" {TRAIN_CONFIG['batch_size']*TRAIN_CONFIG['gradient_accumulation_steps']} effective")
    print(f"  AMP:    {'✓' if TRAIN_CONFIG['use_mixed_precision'] else '✗'}")
    print(f"  Samples:{total:,}  ({per_phase:,} train + {val_n:,} val / фазу × {n_phases} фазы)")
    print(f"  Classes:{NUM_CLASSES}")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Workers:{TRAIN_CONFIG['num_workers']}")
    print(f"  Reshape: [B,3,1024] → [B,3,8,128] → [B,3,112,112]  ← исправлено")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print_config()
