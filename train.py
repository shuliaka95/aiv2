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
        num_iq_samples=NUM_IQ_SAMPLES,
        num_classes=NUM_CLASSES,
    )
    
    # Проверка данных
    test_batch = next(iter(train_loader))
    print(f"Форма данных: {test_batch[0].shape}")
    print(f"Диапазон меток: [{test_batch[1].min().item()}, {test_batch[1].max().item()}]")
    print(f"Все метки < {NUM_CLASSES}? {test_batch[1].max().item() < NUM_CLASSES}")
    
    # Для IterableDataset используем фиксированное количество шагов
    STEPS_PER_EPOCH = 15000  # Увеличил для большей модели
    VAL_STEPS = 1500

    # Инициализация модели
    model = MyCustomModel(num_classes=NUM_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== АРХИТЕКТУРА МОДЕЛИ ===")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Пропорция обучаемых: {trainable_params/total_params:.2%}")
    
    # Функция потерь с label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.3)  # Увеличил smoothing
    
    # Оптимизатор с большим weight decay
    optimizer = optim.AdamW(model.parameters(),
                          lr=TRAIN_CONFIG['learning_rate'],
                          weight_decay=1e-4,
                          betas=(0.9, 0.999),
                          eps=1e-8)
    
    # Комбинированные schedulers
    #scheduler_plateau = VerboseReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                             #   patience=5, min_lr=1e-7)  # Больше patience
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

    #  CURRICULUM LEARNING ПЛАН
    curriculum_phases = [
        {'name': 'Фаза 1: Много шума (базовые признаки)',
         'start_epoch': 0,
         'end_epoch': 15,
         'impairment': 1.0,
         'steps': 8000,
         'val_steps': 800},
         
        {'name': 'Фаза 2: Средний шум (тонкие различия)',
         'start_epoch': 16,
         'end_epoch': 35,
         'impairment': 2.0,
         'steps': 10000,
         'val_steps': 1000},
         
        {'name': 'Фаза 3: Мало шума (максимальная точность)',
         'start_epoch': 36,
         'end_epoch': 70,
         'impairment': 3.0,
         'steps': 12000,
         'val_steps': 1200}
    ]
    
    print("\n" + "="*70)
    print("🎯 CURRICULUM LEARNING ПЛАН:")
    print("="*70)
    for phase in curriculum_phases:
        print(f"{phase['name']}:")
        print(f"  Эпохи: {phase['start_epoch']+1}-{phase['end_epoch']}")
        print(f"  Уровень шума: {phase['impairment']}")
        print(f"  Шагов/эпоху: {phase['steps']:,}")
        print(f"  Val шагов: {phase['val_steps']:,}")
        print("-" * 70)
    
    # Сохраняем оригинальные загрузчики для первой фазы
    original_train_loader = train_loader
    original_val_loader = val_loader
    
    # ========== ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ ==========
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 12
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    current_phase_idx = 0
    current_phase = curriculum_phases[0]
    
    for epoch in range(epochs):
        # ===== ОПРЕДЕЛЯЕМ ТЕКУЩУЮ ФАЗУ CURRICULUM =====
        for i, phase in enumerate(curriculum_phases):
            if phase['start_epoch'] <= epoch <= phase['end_epoch']:
                if i != current_phase_idx:
                    current_phase_idx = i
                    current_phase = phase
                    
                    print(f"\n{'='*70}")
                    print(f"🚀 ПЕРЕХОД НА: {current_phase['name']}")
                    print(f"  Уровень шума: {current_phase['impairment']}")
                    print(f"  Шагов/эпоху: {current_phase['steps']:,}")
                    print(f"{'='*70}")
                    
                    # Освобождаем память
                    del train_loader, val_loader
                    torch.cuda.empty_cache()
                    
                    # Создаем новые загрузчики с новым уровнем шума
                    train_loader, val_loader = get_dataloaders(
                        batch_size=TRAIN_CONFIG['batch_size'],
                        num_iq_samples=NUM_IQ_SAMPLES,
                        num_classes=NUM_CLASSES,
                        impairment_level=current_phase['impairment']
                    )
                break
        
        # Получаем параметры текущей фазы
        phase_steps = current_phase['steps']
        phase_val_steps = current_phase['val_steps']
        
        # ===== ОБУЧЕНИЕ =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_iter = iter(train_loader)
        progress_bar = tqdm(range(phase_steps),
                          desc=f"Эпоха {epoch+1}/{epochs} [{current_phase['name'][:15]}...]",
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
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            batch_total = targets.size(0)
            total += batch_total
            correct += (predicted == targets).sum().item()
            
            # Обновление прогресс-бара
            if step % 50 == 0:
                avg_loss = running_loss / (step + 1)
                accuracy = 100 * correct / total if total > 0 else 0
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}", 
                    acc=f"{accuracy:.2f}%",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}"
                )
        
        epoch_loss = running_loss / phase_steps
        epoch_acc = 100 * correct / total if total > 0 else 0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Обновление scheduler
        scheduler_cosine.step()
        
        # ===== ВАЛИДАЦИЯ =====
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        val_iter = iter(val_loader)
        with torch.no_grad():
            for step in range(phase_val_steps):
                try:
                    inputs, targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    inputs, targets = next(val_iter)
                
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = (inputs - inputs.mean(dim=(0, 2), keepdim=True)) / (inputs.std(dim=(0, 2), keepdim=True) + 1e-8)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_running_loss / phase_val_steps
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # ===== ВЫВОД СТАТИСТИКИ =====
        print(f"\n{'='*70}")
        print(f"Эпоха {epoch+1:3d}/{epochs} [{current_phase['name']}]")
        print(f"  Обучение  - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:6.2f}%")
        print(f"  Валидация - Loss: {val_loss:.4f}, Acc: {val_acc:6.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}, Phase: {current_phase_idx+1}/3")
        print(f"  Gap: {epoch_acc - val_acc:.1f}%")
        
        # ===== СОХРАНЕНИЕ МОДЕЛИ =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': epoch_acc,
                'train_loss': epoch_loss,
                'phase': current_phase_idx + 1,
                'impairment': current_phase['impairment']
            }, model_save_path)
            
            print(f"  💾 Сохранена лучшая модель (Acc: {val_acc:.2f}%, Phase: {current_phase_idx+1})")
        else:
            patience_counter += 1
            print(f"  ⏳ Без улучшений: {patience_counter}/{max_patience}")
        
        # ===== РАННЯЯ ОСТАНОВКА =====
        if patience_counter >= max_patience:
            print(f"\n{'='*70}")
            print(f"⚠️  РАННЯЯ ОСТАНОВКА на эпохе {epoch+1}")
            print(f"   Фаза: {current_phase['name']}")
            print(f"   Лучшая точность: {best_val_acc:.2f}%")
            break
        
        # ===== ПРОГРЕСС КАЖДЫЕ 10 ЭПОХ =====
        if (epoch + 1) % 10 == 0:
            print(f"\n  📊 Прогресс через {epoch+1} эпох:")
            print(f"  Текущая фаза: {current_phase['name']}")
            print(f"  Средний Loss (последние 10): {np.mean(train_losses[-10:]):.4f}")
            print(f"  Средний Acc (последние 10): {np.mean(train_accs[-10:]):.2f}%")
    
    # ===== ФИНАЛЬНАЯ СТАТИСТИКА =====
    print(f"\n{'='*70}")
    print(f"🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Всего эпох: {len(train_accs)}")
    print(f"Лучшая точность: {best_val_acc:.2f}%")
    print(f"Лучший Loss: {best_val_loss:.4f}")
    print(f"{'='*70}")
    
    # Загрузка лучшей модели для финального теста
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n📦 Загружена лучшая модель:")
        print(f"  Эпоха: {checkpoint['epoch']+1}")
        print(f"  Фаза: {checkpoint['phase']}")
        print(f"  Accuracy: {checkpoint['val_acc']:.2f}%")
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
