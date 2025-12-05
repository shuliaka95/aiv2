# ml/dataset.py
import torch
from torch.utils.data import DataLoader
from torchsig.datasets import TorchSigIterableDataset, DatasetMetadata

class SignalToTensorWithLabel:
    """Преобразует Signal в (данные, метка)."""
    def __call__(self, signal):
        # 1. Извлекаем данные
        data_c = torch.from_numpy(signal.data)
        real = data_c.real.float()
        imag = data_c.imag.float()
        data_tensor = torch.stack([real, imag], dim=0)
        
        # 2. Извлекаем метку (0-57)
        label = 0
        
        if hasattr(signal, 'component_signals') and signal.component_signals:
            component = signal.component_signals[0]
            if hasattr(component, 'metadata') and component.metadata is not None:
                metadata = component.metadata
                if hasattr(metadata, 'class_idx'):
                    label = int(metadata.class_idx)
                elif hasattr(metadata, 'class_index'):
                    label = int(metadata.class_index)
        
        return data_tensor, label

def get_dataloaders(batch_size=32, num_iq_samples=1024):
    """Создает загрузчики для всех 58 классов."""
    transform = SignalToTensorWithLabel()
    
    dataset_metadata_kwargs = {
        'num_iq_samples_dataset': num_iq_samples,
        'fft_size': num_iq_samples,
        'class_list': None,
        'sample_rate': 10e6,
        'snr_db_min': 0.0,
        'snr_db_max': 30.0,
    }
    
    train_metadata = DatasetMetadata(**dataset_metadata_kwargs)
    val_metadata = DatasetMetadata(**dataset_metadata_kwargs)
    
    # Для валидации немного другие SNR
    val_metadata.snr_db_min = 5.0
    val_metadata.snr_db_max = 25.0
    
    train_dataset = TorchSigIterableDataset(
        dataset_metadata=train_metadata,
        transforms=[transform],
    )
    
    val_dataset = TorchSigIterableDataset(
        dataset_metadata=val_metadata,
        transforms=[transform],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader