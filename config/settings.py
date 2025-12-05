# config/settings.py
import torch
import numpy as np

NUM_CLASSES = 58
NUM_IQ_SAMPLES = 1024

MODEL_CONFIG = {
    'n_classes': NUM_CLASSES,
    'n_samples': NUM_IQ_SAMPLES,
    'model_path': 'model_25M_best.pth'
}

TRAIN_CONFIG = {
    'batch_size': 32,           # Оптимально для 25M
    'learning_rate': 0.0003,    # Средний learning rate
    'num_epochs': 150,          # Достаточно эпох
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}