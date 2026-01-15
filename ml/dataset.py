"""
Dataset module for RF signal modulation classification using TorchSig 2.0
Supports large-scale training with HDF5 caching
"""

import os
import torch
import numpy as np
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config.settings as cfg

# TorchSig 2.0 imports
try:
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.datasets.dataset_metadata import DatasetMetadata
    TORCHSIG_AVAILABLE = True
except ImportError as e:
    TORCHSIG_AVAILABLE = False
    print(f"ERROR: TorchSig 2.0 not available: {e}")
    print("Install with: pip install torchsig")


class SignalTransform:
    """Transform TorchSig Signal objects to tensors with labels"""
    
    def __init__(self, modulation_list):
        self.modulation_list = modulation_list
        self.mod_to_idx = {mod: idx for idx, mod in enumerate(modulation_list)}
        
    def __call__(self, signal):
        """
        Args:
            signal: TorchSig Signal object
            
        Returns:
            tuple: (data_tensor [2, N], label)
        """
        # Extract IQ data
        signal_data = signal.data if hasattr(signal, 'data') else signal.iq_data
        
        if signal_data is None:
            signal_data = np.zeros((cfg.NUM_IQ_SAMPLES,), dtype=np.complex64)
        
        # Convert to [2, N] tensor (I and Q channels)
        if isinstance(signal_data, np.ndarray):
            if np.iscomplexobj(signal_data):
                data_c = torch.from_numpy(signal_data)
                real = data_c.real.float()
                imag = data_c.imag.float()
                data_tensor = torch.stack([real, imag], dim=0)
            else:
                data_tensor = torch.from_numpy(signal_data).float()
                if data_tensor.dim() == 1:
                    data_tensor = data_tensor.view(2, -1)
        else:
            data_tensor = signal_data
        
        # Extract label
        label = 0
        class_name = None
        
        # Try component_signals
        if hasattr(signal, 'component_signals') and signal.component_signals:
            try:
                comp = signal.component_signals[0]
                if hasattr(comp, 'metadata') and comp.metadata:
                    if hasattr(comp.metadata, 'class_name'):
                        class_name = comp.metadata.class_name
                    elif hasattr(comp.metadata, 'class_idx'):
                        label = int(comp.metadata.class_idx) % len(self.modulation_list)
            except:
                pass
        
        # Try signal.metadata
        if class_name is None and hasattr(signal, 'metadata') and signal.metadata:
            try:
                if hasattr(signal.metadata, 'class_name'):
                    class_name = signal.metadata.class_name
                elif hasattr(signal.metadata, 'class_idx'):
                    label = int(signal.metadata.class_idx) % len(self.modulation_list)
            except:
                pass
        
        # Convert class name to index
        if class_name is not None:
            if isinstance(class_name, list):
                class_name = class_name[0]
            if class_name in self.mod_to_idx:
                label = self.mod_to_idx[class_name]
        
        return data_tensor, label


class CachedTorchSigDataset(Dataset):
    """Dataset with HDF5 caching for efficient data loading"""
    
    def __init__(self, h5_path, modulations, augment=False):
        self.h5_path = h5_path
        self.modulations = modulations
        self.mod_to_idx = {mod: idx for idx, mod in enumerate(modulations)}
        self.augment = augment
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['data'])
    
    def apply_augmentations(self, data_tensor):
        """Apply data augmentations to IQ signal [2, N]"""
        
        # Frequency shift
        if np.random.rand() < 0.5:
            freq_shift = np.random.uniform(-0.1, 0.1)
            t = torch.arange(data_tensor.shape[1]).float()
            phase = 2 * np.pi * freq_shift * t
            rotation = torch.stack([torch.cos(phase), torch.sin(phase)])
            i_new = rotation[0] * data_tensor[0] - rotation[1] * data_tensor[1]
            q_new = rotation[1] * data_tensor[0] + rotation[0] * data_tensor[1]
            data_tensor = torch.stack([i_new, q_new])
        
        # Time shift
        if np.random.rand() < 0.5:
            shift = np.random.randint(-data_tensor.shape[1]//4, data_tensor.shape[1]//4)
            data_tensor = torch.roll(data_tensor, shifts=shift, dims=1)
        
        # Phase rotation
        if np.random.rand() < 0.5:
            phase_rot = np.random.uniform(0, 2*np.pi)
            cos_p = np.cos(phase_rot)
            sin_p = np.sin(phase_rot)
            i_new = cos_p * data_tensor[0] - sin_p * data_tensor[1]
            q_new = sin_p * data_tensor[0] + cos_p * data_tensor[1]
            data_tensor = torch.stack([i_new, q_new])
        
        # Amplitude scaling
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            data_tensor = data_tensor * scale
        
        # Additional noise
        if np.random.rand() < 0.3:
            noise_power = 0.01 * torch.mean(data_tensor**2)
            noise = torch.sqrt(noise_power/2) * torch.randn_like(data_tensor)
            data_tensor = data_tensor + noise
        
        return data_tensor
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            data_bytes = f['data'][idx]
            label = int(f['labels'][idx])
        
        data_tensor = pickle.loads(data_bytes.tobytes())
        
        if self.augment:
            data_tensor = self.apply_augmentations(data_tensor)
        
        return data_tensor, torch.tensor(label, dtype=torch.long)


def generate_torchsig_dataset(modulations, impairment_level, num_samples, num_iq_samples, snr_min, snr_max):
    """
    Генерация датасета с использованием TorchSig 2.0 API и кэширование в HDF5.
    
    Args:
        modulations (list): Список типов модуляций.
        impairment_level (float): Уровень искажений (будет приведен к 0, 1 или 2).
        num_samples (int): Количество генерируемых примеров.
        num_iq_samples (int): Размер IQ-окна.
        snr_min (float): Минимальный SNR в дБ.
        snr_max (float): Максимальный SNR в дБ.
    """
    if not TORCHSIG_AVAILABLE:
        raise ImportError("TorchSig 2.0 is required but not installed")
    
    # Приводим уровень к допустимому в TorchSig диапазону [0, 2]
    # 0 - чистый, 1 - кабель, 2 - беспроводной канал
    ts_level = int(min(max(impairment_level, 0), 2))
    
    cache_dir = cfg.DATASET_CONFIG['save_dir']
    os.makedirs(cache_dir, exist_ok=True)
    
    # Формируем имя файла так, чтобы оно зависело от SNR и уровня
    # Это важно: если SNR изменится, должен создаться новый кэш!
    cache_filename = f"ts_lvl{ts_level}_snr{int(snr_min)}to{int(snr_max)}_n{num_samples}.h5"
    cache_path = os.path.join(cache_dir, cache_filename)
    classes_path = os.path.join(cache_dir, f"classes_c{len(modulations)}.pkl")
    
    if os.path.exists(cache_path):
        print(f"[*] Найден кэшированный датасет: {cache_filename}")
        return cache_path, classes_path
    
    print(f"\n[!] Генерация нового датасета TorchSig:")
    print(f"  Модуляций: {len(modulations)}")
    print(f"  Примеров: {num_samples}")
    print(f"  Уровень (TorchSig Level): {ts_level}")
    print(f"  Диапазон SNR: {snr_min:.1f} - {snr_max:.1f} dB")
    
    sample_rate = 200e3
    # Здесь используется ваш класс трансформации
    transform = SignalTransform(modulations)
    
    # Создание метаданных TorchSig 2.0
    metadata = DatasetMetadata(
        sample_rate=sample_rate,
        num_iq_samples_dataset=num_iq_samples,
        fft_size=256,
        num_signals_min=1,
        num_signals_max=1,
        snr_db_min=snr_min,
        snr_db_max=snr_max,
        signal_duration_min=0.9 * num_iq_samples / sample_rate,
        signal_duration_max=1.0 * num_iq_samples / sample_rate,
        signal_bandwidth_min=sample_rate / 4,
        signal_bandwidth_max=sample_rate / 2,
        cochannel_overlap_probability=0,
        class_list=modulations,
        level=ts_level, 
    )
    
    # Инициализация итерируемого датасета
    dataset = TorchSigIterableDataset(
        dataset_metadata=metadata,
        transforms=[transform],
    )
    
    print(f"Запись в HDF5: {cache_path}...")
    
    with h5py.File(cache_path, 'w') as f:
        # vlen=np.dtype('uint8') для хранения сериализованных pickle данных
        dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        h5_data = f.create_dataset('data', (num_samples,), dtype=dtype)
        h5_labels = f.create_dataset('labels', (num_samples,), dtype='i4')
        
        iterator = iter(dataset)
        
        for i in tqdm(range(num_samples), desc="Генерация"):
            try:
                data_tensor, label = next(iterator)
            except StopIteration:
                iterator = iter(dataset)
                data_tensor, label = next(iterator)
            
            # Сериализуем тензор для записи в HDF5
            data_bytes = pickle.dumps(data_tensor)
            h5_data[i] = np.frombuffer(data_bytes, dtype='uint8')
            h5_labels[i] = int(label)
    
    # Сохраняем список классов
    with open(classes_path, 'wb') as f:
        pickle.dump(modulations, f)
    
    print(f"[+] Датасет успешно сохранен.\n")
    
    return cache_path, classes_path

def get_dataloaders(batch_size, impairment_level=0, snr_min=20.0, snr_max=30.0, train_split=0.8, augment_train=True):
    """
    Создание загрузчиков данных для обучения и валидации.
    
    Args:
        batch_size: размер батча
        impairment_level: уровень искажений (0, 1, 2)
        snr_min: минимальный SNR для генерации
        snr_max: максимальный SNR для генерации
        train_split: доля данных для обучения
        augment_train: применять ли аугментации к тренировочным данным
    """
    modulations = cfg.SELECTED_MODS
    num_samples = cfg.DATASET_CONFIG['num_samples']
    num_iq_samples = cfg.NUM_IQ_SAMPLES
    
    print(f"\nDataLoader Creation (2026 Edition)")
    print(f"{'='*70}")
    print(f"  Level: {impairment_level} | SNR: {snr_min} to {snr_max} dB")
    
    # ТЕПЕРЬ ПЕРЕДАЕМ ВСЕ ПАРАМЕТРЫ В ГЕНЕРАТОР
    cache_path, classes_path = generate_torchsig_dataset(
        modulations=modulations, 
        impairment_level=impairment_level, 
        num_samples=num_samples, 
        num_iq_samples=num_iq_samples,
        snr_min=snr_min,
        snr_max=snr_max
    )
    
    # Создаем объекты датасета на основе кэша
    full_dataset_train = CachedTorchSigDataset(cache_path, modulations, augment=augment_train)
    full_dataset_val = CachedTorchSigDataset(cache_path, modulations, augment=False)
    
    total_size = len(full_dataset_train)
    train_size = int(train_split * total_size)
    
    # Фиксируем seed для воспроизводимого разделения на train/val
    generator = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)
    
    num_workers = cfg.DATASET_CONFIG.get('num_workers', 4)
    pin_memory = cfg.DATASET_CONFIG.get('pin_memory', True)
    prefetch_factor = cfg.DATASET_CONFIG.get('prefetch_factor', 4) if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
    )
    
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, modulations



def visualize_dataset_samples(h5_path, modulations, impairment_level, num_samples=16):
    """
    Visualize constellation diagrams from generated dataset
    
    Args:
        h5_path: path to HDF5 cache file
        modulations: list of modulation types
        impairment_level: impairment level
        num_samples: number of samples to visualize
    """
    print(f"Generating dataset visualization...")
    
    try:
        # Load samples
        with h5py.File(h5_path, 'r') as f:
            total_samples = len(f['data'])
            indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
            
            samples = []
            labels = []
            for idx in indices:
                data_bytes = f['data'][idx]
                label = int(f['labels'][idx])
                data_tensor = pickle.loads(data_bytes.tobytes())
                samples.append(data_tensor.numpy())
                labels.append(label)
        
        # Create visualization
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, (signal, label) in enumerate(zip(samples, labels)):
            if idx >= num_samples:
                break
            
            ax = axes[idx]
            class_name = modulations[label]
            
            i_signal = signal[0]
            q_signal = signal[1]
            
            ax.scatter(i_signal, q_signal, s=2, alpha=0.4, c='blue')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('I', fontsize=9)
            ax.set_ylabel('Q', fontsize=9)
            ax.set_title(f'{class_name.upper()}', fontweight='bold', fontsize=11)
            ax.tick_params(labelsize=8)
            
            max_val = max(np.abs(i_signal).max(), np.abs(q_signal).max())
            ax.set_xlim(-max_val*1.1, max_val*1.1)
            ax.set_ylim(-max_val*1.1, max_val*1.1)
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        snr_min = (impairment_level - 1.0) * 10
        snr_max = snr_min + 10
        plt.suptitle(f'Dataset Constellation Diagrams - Impairment {impairment_level:.1f} '
                    f'(SNR {snr_min:.0f}-{snr_max:.0f} dB)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.dirname(h5_path)
        filename = os.path.join(output_dir, f'dataset_visualization_lvl{impairment_level:.1f}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Dataset visualization saved: {filename}\n")
        
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}\n")
