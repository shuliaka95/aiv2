# check_modulation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata
import numpy as np

# Используем 3 модуляции для теста
mods = ['ook', 'bpsk', 'qpsk']

print("🔍 Проверка REAL генерации модуляций TorchSig")

# Создаем датасет с обработкой по умолчанию
metadata = DatasetMetadata(
    num_iq_samples_dataset=1024,
    fft_size=256,
    impairment_level=2.0,  # Добавим шум
    num_signals_max=1,
    num_signals_min=1,
    sample_rate=1e6,
    class_list=mods,
    enable_class_encoding=True,
)

dataset = TorchSigIterableDataset(
    dataset_metadata=metadata,
    processing_pipeline=[]  # Без кастомной обработки
)

print(f"Используем модуляции: {mods}")
labels_seen = []
raw_signals = []

# Собираем 50 сигналов
for i in range(50):
    try:
        signal = next(iter(dataset))
        raw_signals.append(signal)
        
        # Пробуем разные способы получить метку
        label = -1
        
        # Способ 1: проверяем атрибуты сигнала
        print(f"\nSignal {i+1} attributes:")
        print(f"  Type: {type(signal)}")
        print(f"  Dir: {[x for x in dir(signal) if not x.startswith('_')][:10]}")
        
        if hasattr(signal, 'class_idx'):
            label = signal.class_idx
            print(f"  class_idx: {label}")
        
        if hasattr(signal, 'component_signals'):
            print(f"  component_signals: {len(signal.component_signals)}")
            if signal.component_signals:
                comp = signal.component_signals[0]
                print(f"  Component type: {type(comp)}")
                if hasattr(comp, 'metadata'):
                    print(f"  Metadata dir: {[x for x in dir(comp.metadata) if not x.startswith('_')]}")
                    if hasattr(comp.metadata, 'class_idx'):
                        label = comp.metadata.class_idx
                        print(f"  metadata.class_idx: {label}")
                    if hasattr(comp.metadata, 'class_name'):
                        class_name = comp.metadata.class_name
                        print(f"  metadata.class_name: {class_name}")
                        # Пробуем найти в нашем списке
                        if class_name in mods:
                            label = mods.index(class_name)
        
        if label == -1:
            print(f"  ⚠️ Не нашел метку")
            label = 0
        
        labels_seen.append(label)
        print(f"  Final label: {label} -> {mods[label] if label < len(mods) else 'UNKNOWN'}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        break

if labels_seen:
    print(f"\n📊 Статистика меток из {len(labels_seen)} сигналов:")
    label_counts = np.bincount(labels_seen, minlength=len(mods))
    for i, mod in enumerate(mods):
        print(f"  {mod}: {label_counts[i]} ({label_counts[i]/len(labels_seen)*100:.1f}%)")
else:
    print("❌ Не получилось собрать статистику")
