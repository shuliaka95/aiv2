# train_optimized.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset import get_dataloaders
from model import MyCustomModel
from config import TRAIN_CONFIG, MODEL_CONFIG, NUM_CLASSES

def train_model():
    """Оптимизированное обучение с warmup и улучшенным мониторингом."""
    device = TRAIN_CONFIG['device']
    epochs = TRAIN_CONFIG['num_epochs']
    model_save_path = MODEL_CONFIG['model_path']
    
    print(f"=== ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ НА {NUM_CLASSES} КЛАССОВ ===")
    print(f"Устройство: {device}")
    print(f"Эпох: {epochs}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print("=" * 60)

    # Инициализация данных
    train_loader, val_loader = get_dataloaders(TRAIN_CONFIG['batch_size'])
    
    # Количество шагов (уменьшено для более частых эпох)
    STEPS_PER_EPOCH = 200
    VAL_STEPS = 50

    # Инициализация модели
    model = MyCustomModel(num_classes=NUM_CLASSES).to(device)
    
    # Временно уменьшаем Dropout для ускорения обучения
    model.dropout = nn.Dropout(p=0.4)
    
    # Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                          lr=TRAIN_CONFIG['learning_rate'] * 0.1,  # Начальный LR меньше
                          weight_decay=1e-5)
    
    # Scheduler с warmup
    WARMUP_EPOCHS = 5
    
    # Цикл обучения
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(epochs):
        # Learning rate warmup
        if epoch < WARMUP_EPOCHS:
            lr = TRAIN_CONFIG['learning_rate'] * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Постепенно уменьшаем LR
            lr = TRAIN_CONFIG['learning_rate'] * (0.95 ** (epoch - WARMUP_EPOCHS))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # === ОБУЧЕНИЕ ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(range(STEPS_PER_EPOCH),
                          desc=f"Эпоха {epoch+1}/{epochs}")
        train_iter = iter(train_loader)
        
        for step in progress_bar:
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if step % 20 == 0:
                accuracy = correct / total if total > 0 else 0
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}")

        epoch_loss = running_loss / STEPS_PER_EPOCH
        epoch_acc = 100 * correct / total
        
        # === ВАЛИДАЦИЯ ===
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        val_iter = iter(val_loader)
        with torch.no_grad():
            for _ in range(VAL_STEPS):
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    inputs, targets = next(val_iter)
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_running_loss / VAL_STEPS
        val_acc = 100 * val_correct / val_total
        
        # === ВЫВОД СТАТИСТИКИ ===
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch+1:3d}/{epochs}:")
        print(f"  Обучение  - Потеря: {epoch_loss:.4f}, Точность: {epoch_acc:6.2f}%")
        print(f"  Валидация - Потеря: {val_loss:.4f}, Точность: {val_acc:6.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, model_save_path)
            print(f"  ✓ Сохранена лучшая модель (точность: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Ранняя остановка
        if patience_counter >= patience_limit:
            print(f"\n⚠️  Ранняя остановка: точность не улучшалась {patience_limit} эпох")
            break
        
        # Каждые 5 эпох возвращаем нормальный Dropout
        if epoch == 10:
            model.dropout = nn.Dropout(p=0.6)
            print(f"  ↻ Установлен Dropout 0.6 для лучшей регуляризации")

    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()