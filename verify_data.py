import torch
import numpy as np
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms import SignalTransform

# --- Конфигурация ---
NUM_IQ_SAMPLES = 1024
CLASSES = ["bpsk", "qpsk", "16qam", "64qam"] # Классы для генерации

print(f"--- [TEST] Инициализация TorchSig (Проверка Rust-расширений) ---")
print(f"--- [TEST] Классы: {CLASSES}")
print(f"--- [TEST] Длина семпла: {NUM_IQ_SAMPLES}")

# Трансформация: комплексный массив NumPy -> комплексный тензор PyTorch
class ComplexToNumPy(SignalTransform):
    def __call__(self, signal):
        # signal.data - это np.array([I + jQ, ...])
        return signal.data

try:
    # 1. Создаем датасет (on-the-fly генерация)
    transform = ComplexToNumPy()
    
    ds = ModulationsDataset(
        classes=CLASSES,
        use_class_idx=True,
        level=1, 
        num_iq_samples=NUM_IQ_SAMPLES,
        transform=transform,
        include_snr=False
    )

    print("\n--- [TEST] Датасет создан. Генерируем 1-й пример... ---")

    # 2. Берем первый элемент (запускает генерацию)
    # data_cplx: numpy array [1024] типа complex64
    data_cplx, label_idx = ds[0]
    
    # 3. Выводим информацию
    print(f"--- [SUCCESS] Генерация прошла успешно! ---")
    print(f"Тип данных: {type(data_cplx)}")
    print(f"Форма массива: {data_cplx.shape}")
    print(f"DType массива: {data_cplx.dtype}")
    
    # 4. Выводим первые 10 комплексных сэмплов
    print("\n--- Первые 10 комплексных сэмплов (I + jQ) ---")
    print(data_cplx[:10])

    # 5. Выводим статистику
    print(f"\n--- Общая статистика ---")
    print(f"Среднее (Комплексное): {data_cplx.mean():.4f}")
    print(f"Макс. модуль: {np.abs(data_cplx).max():.4f}")

except Exception as e:
    print(f"\n!!! [ERROR] Ошибка при генерации: {e}")
    import traceback
    traceback.print_exc()