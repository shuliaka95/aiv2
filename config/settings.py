# config/settings.py
import torch
import numpy as np

NUM_CLASSES = 57
NUM_IQ_SAMPLES = 1024

MODEL_CONFIG = {
    'n_classes': NUM_CLASSES,
    'n_samples': NUM_IQ_SAMPLES,
    'model_path': 'model_25M_best.pth'
}

TRAIN_CONFIG = {
    'batch_size': 32,       
    'learning_rate': 0.000055,    
    'num_epochs': 70,      
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
