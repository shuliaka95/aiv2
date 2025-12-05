# debug_signal.py
import sys
sys.path.insert(0, '.')
from torchsig.datasets import TorchSigIterableDataset, DatasetMetadata

# Минимальный датасет для диагностики
metadata = DatasetMetadata(
    num_iq_samples_dataset=1024,
    fft_size=1024,
    class_list=['ook', 'bpsk', 'qpsk'],
    sample_rate=10e6,
    kwargs={}
)

dataset = TorchSigIterableDataset(
    dataset_metadata=metadata,
    transforms=[]
)

# Получаем первый сигнал
signal = next(iter(dataset))
print("=" * 60)
print("ДИАГНОСТИКА СТРУКТУРЫ SIGNAL")
print("=" * 60)

# 1. Основные атрибуты
print(f"Тип объекта: {type(signal)}")
print(f"\nВсе атрибуты (первые 20):")
attrs = [attr for attr in dir(signal) if not attr.startswith('_')]
for attr in attrs[:20]:
    try:
        value = getattr(signal, attr)
        print(f"  {attr}: {type(value)} = {repr(value)[:100]}")
    except:
        print(f"  {attr}: <не удалось получить>")

# 2. Проверка metadata
print(f"\nПроверка signal.metadata:")
if hasattr(signal, 'metadata'):
    if signal.metadata is None:
        print("  signal.metadata = None")
    else:
        print(f"  Тип metadata: {type(signal.metadata)}")
        meta_attrs = [attr for attr in dir(signal.metadata) if not attr.startswith('_')]
        for attr in meta_attrs[:15]:
            try:
                value = getattr(signal.metadata, attr)
                print(f"    {attr}: {type(value)} = {repr(value)[:80]}")
            except:
                print(f"    {attr}: <не удалось получить>")
else:
    print("  signal не имеет атрибута metadata")

# 3. Проверка data
print(f"\nПроверка signal.data:")
print(f"  Тип: {type(signal.data)}")
print(f"  Форма: {signal.data.shape}")
print(f"  dtype: {signal.data.dtype}")

print("=" * 60)