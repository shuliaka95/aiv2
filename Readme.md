# RF Signal Modulation Classification using Deep Learning

Professional implementation of automatic modulation classification (AMC) system using XCiT transformer architecture and TorchSig 2.0 framework.

## Overview

This system classifies 57 different types of RF signal modulations using deep learning. It employs a Vision Transformer (XCiT) adapted for 1D IQ signal processing, trained with curriculum learning strategy for robust performance across varying SNR conditions.

## Features

- **57 Modulation Types**: Supports all TorchSig 2.0 modulation classes
- **Scalable Dataset**: Handles datasets from 100K to 15M+ samples with HDF5 caching
- **Curriculum Learning**: Progressive training from clean to noisy signals
- **Data Augmentation**: Frequency shift, time shift, phase rotation, amplitude scaling
- **Efficient Training**: HDF5 caching, multi-worker data loading, mixed precision support

## Supported Modulations

### Digital Modulations (48)
- **PSK**: BPSK, QPSK, 8PSK, 16PSK, 32PSK, 64PSK
- **QAM**: 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, 512QAM, 1024QAM
- **FSK/GFSK**: 2/4/8/16-FSK, 2/4/8/16-GFSK
- **MSK/GMSK**: 2/4/8/16-MSK, 2/4/8/16-GMSK
- **ASK**: OOK, 4ASK, 8ASK, 16ASK, 32ASK, 64ASK
- **OFDM**: 64/72/128/180/256/300/512/600/900/1024/1200/2048 subcarriers

### Analog Modulations (6)
- **AM**: DSB, DSB-SC, LSB, USB
- **FM**, **Tone**

### Special Signals (3)
- **Chirp SS**
- **LFM**: Data, Radar

## Architecture

```
Input: [batch, 2, 1024] (I/Q channels)
    ↓
Preprocessing CNN: Conv1d layers
    [2, 1024] → [64, 256]
    ↓
Reshape: [64, 256] → [64, 16, 16]
    ↓
XCiT Transformer: 12 layers, patch size 16
    ↓
Output: [batch, 57] (class logits)
```

## Requirements

```
torch >= 2.0.0
timm >= 0.9.0
h5py >= 3.8.0
numpy >= 1.24.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scikit-learn >= 1.2.0
tqdm >= 4.65.0
```

## Installation

```bash
# Clone repository
git clone https://github.com/shuliaka95/aiv2.git
cd aiv2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

#Install torchsig
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install .
```

## Quick Start

### 1. Configuration

Edit `config/settings.py`:

```python
DATASET_CONFIG = {
    'num_samples': 100000,  # Start small, scale to 15M
    'num_classes': 57,      # Use all modulations
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 80,
}
```

### 2. Training

```bash
python ml/train.py
```

## Training Strategy

### Curriculum Learning (3 Phases)

**Phase 1 (Epochs 0-26): Clean Signals**
- SNR: 20-30 dB
- Goal: Learn basic modulation features
- Expected Accuracy: 90-95%

**Phase 2 (Epochs 27-53): Medium Impairments**
- SNR: 10-20 dB
- Goal: Adapt to realistic channel conditions
- Expected Accuracy: 85-92%

**Phase 3 (Epochs 54-80): Strong Impairments**
- SNR: 0-10 dB
- Goal: Robust performance in noisy conditions
- Expected Accuracy: 80-85%

### Data Augmentation

Applied to training data only:
- Frequency shift: ±10% normalized frequency
- Time shift: Cyclic roll ±25% of signal length
- Phase rotation: Random 0-2π rotation
- Amplitude scaling: 0.8-1.2x
- Additive noise: 1% of signal power

## Performance


### Scaling to 15M Samples

For production-scale training:


1. Training time (NVIDIA A100):
   - Dataset generation: ~2-3 hours
   - Training (80 epochs): ~24-36 hours

2. Storage requirements:
   - HDF5 cache: ~150-200 GB
   - Model checkpoints: ~100 MB each

## Project Structure

```
project/
├── config/
    ├── __init__.py
│   └── settings.py          # All configuration parameters
├── ml/
    ├── __init__.py
│   ├── dataset.py           # TorchSig data generation and loading
│   ├── model.py             # XCiT architecture
│   ├── train.py             # Training script
│   └── inference.py         # Inference and evaluation
├── torchsig_datasets/       # Generated datasets (auto-created)
├── best_model.pth           # Best model checkpoint
├── training_report.png      # Training curves
├── confusion_matrix.png     # Classification confusion matrix
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Files

### config/settings.py
- Dataset parameters
- Model configuration
- Training hyperparameters
- Curriculum learning phases
- Random seed for reproducibility

### ml/dataset.py
- TorchSig 2.0 integration
- HDF5 caching for efficient storage
- Data augmentation pipeline
- Multi-worker data loading

### ml/model.py
- XCiT architecture adapted for 1D signals
- CNN preprocessing layers
- Flexible number of output classes

### ml/train.py
- Training loop with curriculum learning
- Automatic phase switching
- Model checkpointing
- Training visualization

## Advanced Usage

### Custom Modulation Subset

```python
# In config/settings.py
SELECTED_MODS = [
    'bpsk', 'qpsk', '8psk', '16psk',
    '16qam', '64qam', '256qam',
    '2fsk', '4fsk', 'fm'
]
```

### Adjust Training Parameters

```python
TRAIN_CONFIG = {
    'batch_size': 128,          # Larger batch for faster training
    'learning_rate': 2e-4,      # Higher LR for faster convergence
    'weight_decay': 0.01,       # Less regularization
    'mixup_alpha': 0.1,         # Less aggressive mixup
    'label_smoothing': 0.05,    # Reduce label smoothing
}
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
model = nn.DataParallel(model)
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
TRAIN_CONFIG['batch_size'] = 32
```

### Slow Data Loading
```python
# Increase workers (not for Windows)
DATASET_CONFIG['num_workers'] = 8
```

### Low Accuracy
- Increase dataset size
- Reduce mixup_alpha
- Increase learning rate
- Add more curriculum phases

## Citation

If you use this code in your research, please cite:

```
@software{rf_modulation_classifier,
  title={RF Signal Modulation Classification using XCiT},
  author={Efanasiy},
  year={2026},
  url={https://github.com/shuliaka95/aiv2/}
}
```

## License

MIT License

## References

- TorchSig: https://github.com/torchdsp/torchsig
- XCiT: https://arxiv.org/abs/2106.09681
- timm: https://github.com/huggingface/pytorch-image-models

## Contact

For questions or issues, please open an issue on GitHub or contact [shulyakrabota@gmail.com]
