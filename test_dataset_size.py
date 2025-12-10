import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.dataset import get_dataloaders
import torch

print("🔧 ИСПРАВЛЕНИЕ КЛАССОВ")

# Тест 1: Создаем на 58 классов (как сейчас)
print("\n=== ТЕСТ 1: NUM_CLASSES=58 ===")
try:
    train_loader, val_loader = get_dataloaders(
        batch_size=8,
        num_iq_samples=1024,
        num_classes=58
    )
    
    # Проверяем метки
    train_iter = iter(train_loader)
    inputs, labels = next(train_iter)
    
    print(f"Max label: {labels.max().item()}")
    print(f"Min label: {labels.min().item()}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    if labels.max().item() >= 58:
        print("⚠️  ОШИБКА: метки >= 58!")
    elif labels.max().item() == 57:
        print("✅ Метки 0-57 (58 классов)")
    elif labels.max().item() == 56:
        print("⚠️  Метки 0-56 (только 57 классов)")
        
except Exception as e:
    print(f"Ошибка: {e}")

# Тест 2: Создаем на 57 классов
print("\n=== ТЕСТ 2: NUM_CLASSES=57 ===")
try:
    train_loader, val_loader = get_dataloaders(
        batch_size=8,
        num_iq_samples=1024,
        num_classes=57
    )
    
    train_iter = iter(train_loader)
    inputs, labels = next(train_iter)
    
    print(f"Max label: {labels.max().item()}")
    print(f"Should be 56: {'✅' if labels.max().item() == 56 else '❌'}")
    
except Exception as e:
    print(f"Ошибка: {e}")

print("\n🎯 РЕКОМЕНДАЦИЯ:")
print("1. В config/settings.py измени NUM_CLASSES = 57")
print("2. В ml/model.py проверь что model = ModulationNet25M(num_classes=57)")
print("3. Перезапусти обучение")
