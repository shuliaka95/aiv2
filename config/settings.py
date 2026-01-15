"""
Configuration file for RF modulation classification project
"""

import torch
import numpy as np
import random

# Seed for reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset configuration
DATASET_CONFIG = {
    'num_samples': 15000000,       
    'num_iq_samples': 1024,        # IQ sample length
    'num_classes': 57,             # Number of modulation classes
    'save_dir': './torchsig_datasets',
    'num_workers': 4,              # DataLoader workers
    'pin_memory': True,
}

# All 57 modulation types supported by TorchSig 2.0
TORCHSIG_ALL_MODS = [
    # AM signals
    'am-dsb-sc', 'am-dsb', 'am-lsb', 'am-usb',
    
    # Chirp SS
    'chirpss',
    
    # Constellation signals
    'ook', '4ask', '8ask', '16ask', '32ask', '64ask',
    'bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk',
    '16qam', '32qam', '32qam_cross', '64qam', '128qam_cross',
    '256qam', '512qam_cross', '1024qam',
    
    # FM signals
    'fm',
    
    # FSK signals
    '2fsk', '2gfsk', '2msk', '2gmsk',
    '4fsk', '4gfsk', '4msk', '4gmsk',
    '8fsk', '8gfsk', '8msk', '8gmsk',
    '16fsk', '16gfsk', '16msk', '16gmsk',
    
    # LFM signals
    'lfm_data', 'lfm_radar',
    
    # OFDM signals
    'ofdm-64', 'ofdm-72', 'ofdm-128', 'ofdm-180', 'ofdm-256',
    'ofdm-300', 'ofdm-512', 'ofdm-600', 'ofdm-900', 'ofdm-1024',
    'ofdm-1200', 'ofdm-2048',
    
    # Tone
    'tone'
]

# Selected modulations for training
SELECTED_MODS = TORCHSIG_ALL_MODS  

NUM_CLASSES = len(SELECTED_MODS)
NUM_IQ_SAMPLES = DATASET_CONFIG['num_iq_samples']


# Model configuration
MODEL_CONFIG = {
    'model_name': 'xcit_small_12_p16_224',
    'num_classes': NUM_CLASSES,
    'pretrained': False,
    'drop_path_rate': 0.1,
}


# Training configuration for 15M samples
TRAIN_CONFIG = {
    'batch_size': 128,              
    'learning_rate': 5e-4,          
    'epochs': 120,                
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': DATASET_CONFIG['num_workers'],
    
    # Optimization
    'weight_decay': 0.05,           # L2 regularization
    'momentum': 0.9,                # For SGD (if used)
    'betas': (0.9, 0.999),          # Adam betas
    
    # Regularization
    'mixup_alpha': 0.2,             # Mixup augmentation strength
    'label_smoothing': 0.1,         # Label smoothing for better generalization
    'dropout': 0.1,                 # Dropout rate (if applicable)
    
    # Learning rate schedule
    'warmup_epochs': 10,            # Longer warmup for large dataset
    'lr_schedule': 'cosine',        # 'cosine' or 'step'
    'min_lr': 1e-6,                 # Minimum learning rate
    
    # Gradient control
    'gradient_clip_norm': 1.0,      # Gradient clipping for stability
    'gradient_accumulation': 1,     # Accumulate gradients over N batches
    
    # Mixed precision training
    'use_amp': True,                # Automatic Mixed Precision for speed
    'amp_dtype': 'float16',         # 'float16' or 'bfloat16'
    
    # Checkpointing
    'save_frequency': 5,            # Save checkpoint every N epochs
    'keep_last_n': 3,               # Keep last N checkpoints
}


# Curriculum learning phases for 120 epochs
CURRICULUM_PHASES = [
    {
        'name': 'Phase 1: Clean signals (high SNR)',
        'start_epoch': 0,
        'end_epoch': 39,
        'impairment': 0,
        'snr_min': 20, 'snr_max':40,  # SNR 20-30 dB
        'description': 'Learn fundamental modulation patterns on clean signals'
    },
    {
        'name': 'Phase 2: Medium impairments',
        'start_epoch': 40,
        'end_epoch': 79,
        'impairment': 1,
        'snr_min': 0, 'snr_max':20,  # SNR 10-20 dB
        'description': 'Adapt to realistic channel conditions'
    },
    {
        'name': 'Phase 3: Strong impairments',
        'start_epoch': 80,
        'end_epoch': 120,
        'impairment': 2,
        'snr_min': -20, 'snr_max':0,  # SNR 0-10 dB
        'description': 'Achieve robustness in harsh conditions'
    }
]


def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("Project Configuration")
    print("="*70)
    print(f"Dataset:")
    print(f"  Samples: {DATASET_CONFIG['num_samples']:,}")
    print(f"  IQ length: {NUM_IQ_SAMPLES}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Modulations: {', '.join(SELECTED_MODS[:5])}... ({len(SELECTED_MODS)} total)")
    print(f"\nModel:")
    print(f"  Architecture: {MODEL_CONFIG['model_name']}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Drop path rate: {MODEL_CONFIG['drop_path_rate']}")
    print(f"\nTraining:")
    print(f"  Device: {TRAIN_CONFIG['device']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Weight decay: {TRAIN_CONFIG['weight_decay']}")
    print(f"  Label smoothing: {TRAIN_CONFIG['label_smoothing']}")
    print(f"\nCurriculum Learning:")
    for i, phase in enumerate(CURRICULUM_PHASES, 1):
        snr_min = (phase['impairment'] - 1.0) * 10
        snr_max = snr_min + 10
        print(f"  Phase {i}: epochs {phase['start_epoch']}-{phase['end_epoch']}, "
              f"SNR={snr_min:.0f}-{snr_max:.0f}dB")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
