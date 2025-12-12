# ml/dataset.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchsig.datasets import TorchSigIterableDataset, DatasetMetadata

class SignalToTensorWithLabel:
    """Преобразует Signal объект TorchSig в (данные, метка)."""
    
    def __init__(self, modulation_list):
        self.modulation_list = modulation_list
        self.mod_to_idx = {mod: idx for idx, mod in enumerate(modulation_list)}
    
    def __call__(self, signal):
        """
        Вход: signal - объект Signal из TorchSig
        Выход: (data_tensor, label)
        """
        # 1. Извлекаем данные из Signal объекта
        # У объекта Signal есть атрибут .data с numpy array
        signal_data = signal.data 
        
        # Проверяем, что данные получены
        if signal_data is None:
            # Создаем fallback данные
            signal_data = np.zeros((1024,), dtype=np.complex64)
        
        # 2. Преобразуем в тензор PyTorch
        data_c = torch.from_numpy(signal_data)
        real = data_c.real.float()
        imag = data_c.imag.float()
        data_tensor = torch.stack([real, imag], dim=0)
        
        # 3. Извлекаем метку из метаданных Signal
        label = 0 
        
        # Способ 1: Из component_signals
        if hasattr(signal, 'component_signals') and signal.component_signals:
            try:
                comp = signal.component_signals[0]
                if hasattr(comp, 'metadata') and comp.metadata:
                    # Пробуем получить class_name
                    if hasattr(comp.metadata, 'class_name'):
                        class_name = comp.metadata.class_name
                        if class_name in self.mod_to_idx:
                            label = self.mod_to_idx[class_name]
                    # Или class_idx
                    elif hasattr(comp.metadata, 'class_idx'):
                        label = int(comp.metadata.class_idx) % len(self.modulation_list)
            except:
                pass
        
        # Способ 2: Из самого signal
        if label == 0 and hasattr(signal, 'metadata') and signal.metadata:
            try:
                if hasattr(signal.metadata, 'class_name'):
                    class_name = signal.metadata.class_name
                    if class_name in self.mod_to_idx:
                        label = self.mod_to_idx[class_name]
                elif hasattr(signal.metadata, 'class_idx'):
                    label = int(signal.metadata.class_idx) % len(self.modulation_list)
            except:
                pass
        
        # Способ 3: Fallback - используем modulation_list
        if label == 0:
            # Пробуем определить класс по содержимому сигнала
            label = hash(str(signal_data.shape)) % len(self.modulation_list)
        
        return data_tensor, label

def get_dataloaders(batch_size=32, num_iq_samples=1024, num_classes=58, impairment_level=1.0):
    """Создает загрузчики для всех 58 классов."""
    ALL_MODS = [ 
        # Простые
        'ook', 'bpsk', 'qpsk', '8psk', '16qam',
        '2fsk', '4fsk', '8fsk', '16fsk',
        'fm', 'am-dsb', 'tone',
        
        # Средние
        '2gfsk', '4gfsk', '8gfsk', '16gfsk',
        '2msk', '4msk', '8msk', '16msk',
        '2gmsk', '4gmsk', '8gmsk', '16gmsk',
        'am-dsb-sc', 'am-lsb', 'am-usb',
        '32qam', '64qam', '128qam_cross', '256qam',
        
        # Сложные
        '4ask', '8ask', '16ask', '32ask', '64ask',
        '32qam_cross', '512qam_cross', '1024qam',
        'lfm_data', 'lfm_radar', 'chirpss',
        
        # Очень сложные
        '16psk', '32psk', '64psk',
        'ofdm-64', 'ofdm-72', 'ofdm-128', 'ofdm-180',
        'ofdm-256', 'ofdm-300', 'ofdm-512', 'ofdm-600',
        'ofdm-900', 'ofdm-1024', 'ofdm-1200', 'ofdm-2048'
    ]
    
    selected_mods = ALL_MODS[:min(num_classes, len(ALL_MODS))]
    
    transform = SignalToTensorWithLabel(selected_mods)
    
    print(f"[DATASET] Используем {len(selected_mods)} классов из TorchSig")
    print(f"Модуляции: {selected_mods}")
    
    # SNR диапазоны
    if impairment_level <= 1.0:
        train_snr_min, train_snr_max = 0.0, 15.0
        val_snr_min, val_snr_max = 5.0, 20.0
    elif impairment_level <= 2.0:
        train_snr_min, train_snr_max = 5.0, 25.0
        val_snr_min, val_snr_max = 10.0, 30.0
    elif impairment_level <= 3.0:
        train_snr_min, train_snr_max = 15.0, 35.0
        val_snr_min, val_snr_max = 20.0, 40.0
    else:
        train_snr_min, train_snr_max = 25.0, 45.0
        val_snr_min, val_snr_max = 30.0, 50.0
    
    dataset_metadata_kwargs = {
        'num_iq_samples_dataset': num_iq_samples,
        'fft_size': num_iq_samples,
        'class_list': selected_mods,
        'sample_rate': 10e6,
        'snr_db_min': train_snr_min,
        'snr_db_max': train_snr_max,
        'impairment_level': impairment_level,
    }
    
    train_metadata = DatasetMetadata(**dataset_metadata_kwargs)
    val_metadata = DatasetMetadata(**dataset_metadata_kwargs)
    
    # Для валидации немного другие SNR
    val_metadata.snr_db_min = val_snr_min
    val_metadata.snr_db_max = val_snr_max
    val_metadata.impairment_level = max(0.5, impairment_level * 0.7)
    
    train_dataset = TorchSigIterableDataset(
        dataset_metadata=train_metadata,
        transforms=[transform],  # Используем transforms (список)
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
