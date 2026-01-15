import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.dataset import get_dataloaders
from ml.model import get_model
from config.settings import TRAIN_CONFIG, CURRICULUM_PHASES


def mixup_data(x, y, alpha=0.2):
    """
    Mixup аугментация: смешивание двух сигналов для регуляризации
    Дополнительно к frequency/time shift аугментациям в датасете
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss для Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_plots(history, save_path='training_report.png'):
    """Сохранение графиков обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График точности
    axes[0].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Acc', marker='x', linewidth=2)
    axes[0].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # График loss
    axes[1].plot(history['train_loss'], label='Train Loss', color='red', linewidth=2)
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_confusion_matrix(all_preds, all_labels, class_names, save_path='confusion_matrix.png'):
    """Генерация и сохранение confusion matrix"""
    plt.figure(figsize=(20, 16))
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(
        cm_norm, 
        annot=False, 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_current_phase(epoch):
    """Определяет текущую фазу curriculum learning"""
    for phase in CURRICULUM_PHASES:
        if phase['start_epoch'] <= epoch <= phase['end_epoch']:
            return phase
    return CURRICULUM_PHASES[-1]  # последняя фаза по умолчанию


def train():
    device = TRAIN_CONFIG['device']
    print(f"\n{'='*70}")
    print(f" ЗАПУСК ОБУЧЕНИЯ (TorchSig 2.0 API)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"{'='*70}\n")
    
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'phases': []
    }
    
    # 1. Инициализация ПЕРВОЙ фазы (используем 0 вместо epoch)
    current_phase = get_current_phase(0)
    
    print(f"[*] Начальная фаза: {current_phase['name']}")
    
    # Корректный вызов с новыми параметрами
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'], 
        impairment_level=current_phase['impairment'], 
        snr_min=current_phase['snr_min'],        
        snr_max=current_phase['snr_max'],
        augment_train=True
    )
    
    num_classes = len(class_names)
    print(f"   Количество классов: {num_classes}\n")
    
    # Инициализация модели
    model = get_model(model_type='xcit', num_classes=num_classes).to(device)
    
    # Оптимизатор
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG.get('weight_decay', 0.05)
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG.get('label_smoothing', 0.1))
    
    # Scheduler
    warmup_epochs = TRAIN_CONFIG.get('warmup_epochs', 10)
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / (TRAIN_CONFIG['epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc = 0.0

    # ==========================
    # ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
    # ==========================
    for epoch in range(TRAIN_CONFIG['epochs']):
        
        # Проверяем смену фазы (начиная со второй эпохи)
        new_phase = get_current_phase(epoch)
        if new_phase != current_phase:
            print(f"\n{'-'*40}")
            print(f" [!] СМЕНА ФАЗЫ: {new_phase['name']}")
            print(f" SNR: {new_phase['snr_min']} to {new_phase['snr_max']} dB")
            print(f"{'-'*40}\n")
            
            current_phase = new_phase
            # Обновляем загрузчики под новые условия
            train_loader, val_loader, _ = get_dataloaders(
                batch_size=TRAIN_CONFIG['batch_size'], 
                impairment_level=current_phase['impairment'],
                snr_min=current_phase['snr_min'],
                snr_max=current_phase['snr_max']
            )
        
        # TRAINING
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup
            mixup_alpha = TRAIN_CONFIG.get('mixup_alpha', 0.2)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.get('gradient_clip_norm', 1.0))
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            # Метрика с учетом Mixup
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.7f}"
            })
        
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"[*] Epoch {epoch+1} Result: Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # Сохранение истории
        history['train_loss'].append(total_loss/len(train_loader))
        history['train_acc'].append(100.*correct/total)
        history['val_acc'].append(val_acc)
        
        # Шаг шедулера
        scheduler.step()

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"   [+] New best model saved ({best_val_acc:.2f}%)")

    # Финальные графики
    save_plots(history)

    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"{'='*70}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: best_model.pth")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train()
