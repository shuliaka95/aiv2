# train.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataset import get_dataloaders
from model import ModulationNet25M as MyCustomModel
from config import TRAIN_CONFIG, MODEL_CONFIG, NUM_CLASSES, NUM_IQ_SAMPLES

class VerboseReduceLROnPlateau(ReduceLROnPlateau):
    """Кастомный scheduler с выводом информации."""
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8):
        super().__init__(optimizer, mode, factor, patience, threshold,
                         threshold_mode, cooldown, min_lr, eps)
    
    def step(self, metrics):
        old_lr = self.optimizer.param_groups[0]['lr']
        super().step(metrics)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  ↻ Learning rate изменен: {old_lr:.2e} → {new_lr:.2e}")

def train_model():
    """Мощный пайплайн обучения с защитой от переобучения."""
    device = TRAIN_CONFIG['device']
    epochs = TRAIN_CONFIG['num_epochs']
    model_save_path = MODEL_CONFIG['model_path']
    
    print(f"=== ОБУЧЕНИЕ МОЩНОЙ МОДЕЛИ НА {NUM_CLASSES} КЛАССОВ ===")
    print(f"Устройство: {device}")
    print(f"Эпох: {epochs}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print("=" * 70)

    # Инициализация данных
    train_loader, val_loader = get_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        num_iq_samples=NUM_IQ_SAMPLES
    )
    
    # Проверка данных
    test_batch = next(iter(train_loader))
    print(f"Форма данных: {test_batch[0].shape}")
    print(f"Диапазон меток: [{test_batch[1].min().item()}, {test_batch[1].max().item()}]")
    print(f"Все метки < {NUM_CLASSES}? {test_batch[1].max().item() < NUM_CLASSES}")
    
    # Для IterableDataset используем фиксированное количество шагов
    STEPS_PER_EPOCH = 250  # Увеличил для большей модели
    VAL_STEPS = 75

    # Инициализация модели
    model = MyCustomModel(num_classes=NUM_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== АРХИТЕКТУРА МОДЕЛИ ===")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Пропорция обучаемых: {trainable_params/total_params:.2%}")
    
    # Функция потерь с label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Увеличил smoothing
    
    # Оптимизатор с большим weight decay
    optimizer = optim.AdamW(model.parameters(),
                          lr=TRAIN_CONFIG['learning_rate'],
                          weight_decay=2e-3,  # Увеличил weight decay
                          betas=(0.9, 0.999))
    
    # Комбинированные schedulers
    scheduler_plateau = VerboseReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                patience=5, min_lr=1e-7)  # Больше patience
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    # Цикл обучения
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 12  # Увеличил patience
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Для отслеживания прогресса
    train_iter = iter(train_loader)
    
    for epoch in range(epochs):
        # === ОБУЧЕНИЕ ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(range(STEPS_PER_EPOCH),
                          desc=f"Эпоха {epoch+1}/{epochs}",
                          leave=False)
        
        for step in progress_bar:
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Нормализация по батчу
            inputs = (inputs - inputs.mean(dim=(0, 2), keepdim=True)) / (inputs.std(dim=(0, 2), keepdim=True) + 1e-8)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Добавляем L2 регуляризацию вручную
            l2_lambda = 0.002  # Увеличил
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)  # Уменьшил
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            batch_total = targets.size(0)
            total += batch_total
            correct += (predicted == targets).sum().item()
            
            # Обновление прогресс-бара
            if step % 20 == 0:
                avg_loss = running_loss / (step + 1)
                accuracy = 100 * correct / total if total > 0 else 0
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}", 
                    acc=f"{accuracy:.2f}%",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}"
                )
        
        epoch_loss = running_loss / STEPS_PER_EPOCH
        epoch_acc = 100 * correct / total if total > 0 else 0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Обновление cosine scheduler
        scheduler_cosine.step()
        
        # === ВАЛИДАЦИЯ ===
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        val_iter = iter(val_loader)
        with torch.no_grad():
            for step in range(VAL_STEPS):
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    inputs, targets = next(val_iter)
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Нормализация
                inputs = (inputs - inputs.mean(dim=(0, 2), keepdim=True)) / (inputs.std(dim=(0, 2), keepdim=True) + 1e-8)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_running_loss / VAL_STEPS
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # === ВЫВОД СТАТИСТИКИ ===
        print(f"\n{'='*70}")
        print(f"Эпоха {epoch+1:3d}/{epochs}:")
        print(f"  Обучение  - Потеря: {epoch_loss:.4f}, Точность: {epoch_acc:6.2f}%")
        print(f"  Валидация - Потеря: {val_loss:.4f}, Точность: {val_acc:6.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Gap (train-val): {epoch_acc - val_acc:.1f}%")
        
        # Обновление plateau scheduler
        scheduler_plateau.step(val_acc)
        
        # Проверка на переобучение
        overfitting_warning = ""
        
        if epoch >= 5:
            # Разрыв между train и val accuracy
            if epoch_acc - val_acc > 12:  # Уменьшил порог
                overfitting_warning = f" ⚠️  Разрыв train-val: {epoch_acc-val_acc:.1f}%"
            
            # Растущая val loss
            if len(val_losses) >= 4 and all(val_losses[-i] > val_losses[-(i+1)] for i in range(1, 3)):
                overfitting_warning = " ⚠️  Val loss растет 2 эпохи подряд!"
            
            # Падающая val accuracy
            if len(val_accs) >= 4 and all(val_accs[-i] < val_accs[-(i+1)] for i in range(1, 3)):
                overfitting_warning = " ⚠️  Val accuracy падает 2 эпохи подряд!"
        
        if overfitting_warning:
            print(f"  {overfitting_warning}")
            
            # Автоматическое увеличение dropout при переобучении
            if hasattr(model, 'classifier'):
                for module in model.classifier:
                    if isinstance(module, nn.Dropout):
                        if module.p < 0.8:  # Максимум 80%
                            module.p = min(0.8, module.p + 0.03)
                            print(f"  ↻ Увеличен Dropout: {module.p-0.03:.2f} → {module.p:.2f}")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': epoch_acc,
                'train_loss': epoch_loss,
                'config': {
                    'num_classes': NUM_CLASSES,
                    'learning_rate': TRAIN_CONFIG['learning_rate'],
                    'batch_size': TRAIN_CONFIG['batch_size']
                }
            }, model_save_path)
            print(f"  ✓ Сохранена лучшая модель (точность: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Без улучшений: {patience_counter}/{max_patience}")
        
        # Сохранение чекпоинта
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': epoch_acc,
                'val_acc': val_acc,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  💾 Чекпоинт: {checkpoint_path}")
        
        # Ранняя остановка
        if patience_counter >= max_patience:
            print(f"\n{'='*70}")
            print(f"⚠️  РАННЯЯ ОСТАНОВКА на эпохе {epoch+1}")
            print(f"   Точность не улучшалась {max_patience} эпох")
            print(f"   Лучшая точность: {best_val_acc:.2f}%")
            break
        
        # Прогресс каждые 20 эпох
        if (epoch + 1) % 20 == 0:
            print(f"\n  === ПРОГРЕСС ЧЕРЕЗ {epoch+1} ЭПОХ ===")
            print(f"  Train accuracy: {train_accs[0]:.1f}% → {epoch_acc:.1f}%")
            print(f"  Val accuracy: {val_accs[0]:.1f}% → {val_acc:.1f}%")
            print(f"  Средний gap: {np.mean([t-v for t,v in zip(train_accs[-10:], val_accs[-10:])]):.1f}%")

    # Финальная статистика
    print(f"\n{'='*70}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Лучшая потеря на валидации: {best_val_loss:.4f}")
    print(f"Всего эпох: {len(train_accs)}")
    
    # Анализ переобучения
    if len(train_accs) > 10:
        final_gap = train_accs[-1] - val_accs[-1]
        avg_gap = np.mean([t - v for t, v in zip(train_accs[-10:], val_accs[-10:])])
        
        print(f"\n=== АНАЛИЗ ПЕРЕОБУЧЕНИЯ ===")
        print(f"Финальный разрыв (train-val): {final_gap:.2f}%")
        print(f"Средний разрыв (последние 10 эпох): {avg_gap:.2f}%")
        
        if avg_gap > 15:
            print("  ⚠️  СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ!")
            print("  Рекомендация: Увеличьте dropout, уменьшите модель")
        elif avg_gap > 8:
            print("  ⚠️  УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ")
        else:
            print("  ✓ ХОРОШАЯ ОБОБЩАЮЩАЯ СПОСОБНОСТЬ")
    
    # Загрузка лучшей модели для теста
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nЗагружена лучшая модель из эпохи {checkpoint['epoch']+1}")
        
        # Расширенный тест
        model.eval()
        all_predictions = []
        all_targets = []
        
        val_iter = iter(val_loader)
        with torch.no_grad():
            for _ in range(VAL_STEPS * 2):  # Больше данных для теста
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    inputs, targets = next(val_iter)
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Вычисление точности
        correct = sum([1 for p, t in zip(all_predictions, all_targets) if p == t])
        total = len(all_predictions)
        final_acc = 100 * correct / total if total > 0 else 0
        
        print(f"Финальная точность на валидации: {final_acc:.2f}%")
        print(f"Протестировано примеров: {total}")

if __name__ == "__main__":
    train_model()