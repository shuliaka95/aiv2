"""
Тренировочный цикл XCiT Small для RF модуляций.
Mixed Precision (AMP) + Gradient Accumulation + Curriculum Learning.

Сохранение моделей:
  best_model.pth          ← лучшая за всё обучение
  best_model_phase0.pth   ← лучшая Phase 1 (SNR 20-40 dB)
  best_model_phase1.pth   ← лучшая Phase 2 (SNR 10-20 dB)
  best_model_phase2.pth   ← лучшая Phase 3 (SNR 0-10 dB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import gc
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.dataset  import get_dataloaders
from ml.model    import get_model, count_parameters
from config.settings import TRAIN_CONFIG, CURRICULUM_PHASES


# Утилиты 

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#Графики
def save_plots(history, save_path='training_report.png'):
    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Accuracy
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(history['train_acc'], label='Train')
    ax.plot(history['val_acc'],   label='Val')
    for ph in CURRICULUM_PHASES[1:]:
        if ph['start_epoch'] < len(history['val_acc']):
            ax.axvline(ph['start_epoch'], color='red', ls='--', alpha=0.4,
                       label=f"→ {ph['name']}")
    mixup_ep = TRAIN_CONFIG.get('mixup_warmup_epoch', 5)
    if mixup_ep < len(history['val_acc']):
        ax.axvline(mixup_ep, color='green', ls=':', alpha=0.6, label='Mixup ON')
    # Отмечаем лучшие точки по каждой фазе
    for phase_idx, (phase_name, (acc, ep)) in enumerate(history.get('best_per_phase', {}).items()):
        if ep < len(history['val_acc']):
            ax.scatter(ep, acc, s=120, zorder=5,
                       label=f"Best Ph{phase_idx+1}: {acc:.1f}%")
    ax.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('%')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Loss
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(history['train_loss'], label='Train', color='red')
    ax.plot(history['val_loss'],   label='Val',   color='orange')
    ax.set_title('Loss', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Overfitting gap
    ax = fig.add_subplot(gs[1, 0])
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax.plot(gap, color='purple')
    ax.axhline(10, color='orange', ls='--', alpha=0.5)
    ax.fill_between(range(len(gap)), 0, gap,
                    where=(gap > 10), alpha=0.3, color='red')
    ax.set_title('Overfitting Gap', fontsize=13, fontweight='bold')
    ax.set_ylabel('Train − Val (%)')
    ax.grid(alpha=0.3)

    # LR
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(history['learning_rates'], color='green')
    ax.set_title('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Memory
    ax = fig.add_subplot(gs[1, 2])
    if history.get('memory_usage'):
        ax.plot(history['memory_usage'], color='blue')
    ax.set_title('GPU Memory (GB)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Training Report — RF Modulation Classification',
                 fontsize=15, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] График: {save_path}")


def save_confusion_matrix(all_preds, all_labels, class_names,
                           save_path='confusion_matrix.png'):
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, None] + 1e-9)

    order        = np.argsort(cm_norm.diagonal())[::-1]
    cm_sorted    = cm_norm[order][:, order]
    names_sorted = [class_names[i] for i in order]

    plt.figure(figsize=(22, 18))
    sns.heatmap(cm_sorted, annot=False, cmap='Blues',
                xticklabels=names_sorted, yticklabels=names_sorted,
                cbar_kws={'label': 'Normalized'}, vmin=0, vmax=1)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion matrix: {save_path}")


#Curriculum

def get_current_phase(epoch):
    for phase in CURRICULUM_PHASES:
        if phase['start_epoch'] <= epoch <= phase['end_epoch']:
            return phase
    return CURRICULUM_PHASES[-1]


#Валидация

def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds, labels = [], []

    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            if use_amp:
                with autocast('cuda'):
                    out      = model(inp)
                    loss_sum += criterion(out, tgt).item()
            else:
                out      = model(inp)
                loss_sum += criterion(out, tgt).item()

            pred = out.argmax(1)
            correct += pred.eq(tgt).sum().item()
            total   += tgt.size(0)
            preds.extend(pred.cpu().numpy())
            labels.extend(tgt.cpu().numpy())

    return 100. * correct / total, loss_sum / len(loader), preds, labels


#Сохранение чекпоинта

def save_checkpoint(path, model, optimizer, epoch, val_acc, class_names,
                    history, phase_name):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc':              val_acc,
        'class_names':          class_names,
        'history':              history,
        'phase':                phase_name,
    }, path)


#Основной цикл

def train():
    device      = TRAIN_CONFIG['device']
    use_amp     = TRAIN_CONFIG.get('use_mixed_precision', False)
    accum_steps = TRAIN_CONFIG.get('gradient_accumulation_steps', 1)
    cache_freq  = TRAIN_CONFIG.get('empty_cache_every_n_batches', 5)

    mixup_alpha        = TRAIN_CONFIG['mixup_alpha']
    mixup_warmup_epoch = TRAIN_CONFIG.get('mixup_warmup_epoch', 5)
    label_smoothing    = TRAIN_CONFIG.get('label_smoothing', 0.05)

    print(f"\n{'='*70}")
    print(" OPTIMIZED TRAINING — XCiT Small  (57 классов)")
    print(f"{'='*70}")
    print(f"  Device:          {device}")
    print(f"  Mixed Precision: {'✓' if use_amp else '✗'}")
    print(f"  Grad Accum:      {accum_steps}x")
    print(f"  Effective Batch: {TRAIN_CONFIG['batch_size'] * accum_steps}")
    print(f"  Mixup ON после:  epoch {mixup_warmup_epoch}")
    print(f"  Label Smoothing: {label_smoothing}")
    print(f"  Чекпоинты:")
    print(f"    best_model.pth         ← лучшая за всё обучение")
    print(f"    best_model_phase0.pth  ← лучшая Phase 1 (SNR 20–40 dB)")
    print(f"    best_model_phase1.pth  ← лучшая Phase 2 (SNR 0–20 dB)")
    print(f"    best_model_phase2.pth  ← лучшая Phase 3 (SNR -20–0 dB)")
    print(f"{'='*70}\n")
    clear_memory()

    history = {
        'train_acc':      [],
        'val_acc':        [],
        'train_loss':     [],
        'val_loss':       [],
        'learning_rates': [],
        'grad_norms':     [],
        'memory_usage':   [],
        'phases':         [],
        # {phase_name: (best_acc, best_epoch)} — заполняется в процессе
        'best_per_phase': {},
    }

    # Трекеры лучших значений
    best_val_acc_global = 0.0         # лучшая за всё обучение
    best_per_phase      = {}          # {phase_name: best_acc}
    patience_counter    = 0

    # Первая фаза
    current_phase = get_current_phase(0)
    print(f"[*] Старт: {current_phase['name']}")
    print(f"    SNR {current_phase['snr_min']}–{current_phase['snr_max']} dB\n")

    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        impairment_level=current_phase['impairment'],
        snr_min=current_phase['snr_min'],
        snr_max=current_phase['snr_max'],
    )
    num_classes = len(class_names)

    # Модель
    print("[*] Инициализация XCiT Small…")
    model = get_model(num_classes=num_classes, pretrained=False).to(device)
    total_p, train_p = count_parameters(model)
    print(f"    Параметров: {total_p:,}  (обучаемых {train_p:,})\n")
    clear_memory()

    #Оптимизатор и scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['betas'],
    )

    scaler = GradScaler('cuda') if use_amp else None

    criterion_clean = nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_mixup = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    warmup       = TRAIN_CONFIG['warmup_epochs']
    total_epochs = TRAIN_CONFIG['epochs']

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    RESUME_FROM=''
    START_EPOCH=0
    	
    if RESUME_FROM and os.path.exists(RESUME_FROM):
    	ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
    	model.load_state_dict(ckpt['model_state_dict'])
    	optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    	
    #Основной цикл эпох
    start_epoch = START_EPOCH if (RESUME_FROM and os.path.exists(RESUME_FROM)) else 0
    for epoch in range(start_epoch, total_epochs):

        #Смена фазы curriculum
        new_phase = get_current_phase(epoch)
        if new_phase is not current_phase:
            print(f"\n{'='*70}")
            print(f" ПЕРЕХОД → {new_phase['name']}")
            print(f" SNR {new_phase['snr_min']}–{new_phase['snr_max']} dB")
            print(f"{'='*70}\n")
            current_phase = new_phase
            history['phases'].append(epoch)

            del train_loader, val_loader
            clear_memory()

            train_loader, val_loader, _ = get_dataloaders(
                batch_size=TRAIN_CONFIG['batch_size'],
                impairment_level=current_phase['impairment'],
                snr_min=current_phase['snr_min'],
                snr_max=current_phase['snr_max'],
            )

            # Сбрасываем patience при смене фазы —
            # модель должна заново адаптироваться к новым условиям
            patience_counter = 0
            phase_idx = CURRICULUM_PHASES.index(current_phase)
            if phase_idx > 0:
            	for param_group in optimizer.param_groups:
            		param_group['lr'] = param_group['lr'] * 0.1            		
            	current_lr = optimizer.param_groups[0]['lr']
            	
            print(f"[*] Patience counter сброшен для новой фазы\n")

        use_mixup = (epoch >= mixup_warmup_epoch)
        criterion = criterion_mixup if use_mixup else criterion_clean

        if epoch == mixup_warmup_epoch:
            print(f"\n[*] Epoch {epoch+1}: включаем Mixup"
                  f" (alpha={mixup_alpha}) + Label Smoothing ({label_smoothing})\n")

        # Train
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        grad_norms = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{total_epochs} "
                         f"[{current_phase['name'].split(':')[0]}]"
                         f" {'[mixup]' if use_mixup else '[clean]'}")

        for bi, (inp, tgt) in enumerate(pbar):
            inp, tgt = inp.to(device), tgt.to(device)

            if use_mixup:
                inp_m, tgt_a, tgt_b, lam = mixup_data(inp, tgt, mixup_alpha)
            else:
                inp_m = inp

            if use_amp:
                with autocast('cuda'):
                    out = model(inp_m)
                    if use_mixup:
                        loss = mixup_criterion(
                            criterion, out, tgt_a, tgt_b, lam) / accum_steps
                    else:
                        loss = criterion(out, tgt) / accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(inp_m)
                if use_mixup:
                    loss = mixup_criterion(
                        criterion, out, tgt_a, tgt_b, lam) / accum_steps
                else:
                    loss = criterion(out, tgt) / accum_steps
                loss.backward()

            if (bi + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAIN_CONFIG['gradient_clip_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAIN_CONFIG['gradient_clip_norm'])
                    optimizer.step()
                optimizer.zero_grad()
                grad_norms.append(
                    gn.item() if hasattr(gn, 'item') else float(gn))

            # Accuracy всегда по оригинальному tgt
            loss_sum += loss.item() * accum_steps
            pred      = out.argmax(1)
            total    += tgt.size(0)
            correct  += pred.eq(tgt).sum().item()

            if (bi + 1) % cache_freq == 0:
                clear_memory()

            pbar.set_postfix(
                loss=f"{loss.item()*accum_steps:.4f}",
                acc=f"{100.*correct/total:.2f}%")

        # Validation
        val_acc, val_loss, all_preds, all_labels = evaluate(
            model, val_loader, criterion_clean, device, use_amp)

        train_acc  = 100. * correct / total
        train_loss = loss_sum / len(train_loader)
        mem_gb     = (torch.cuda.memory_allocated() / 1e9
                      if torch.cuda.is_available() else 0.0)
        phase_name = current_phase['name']
        phase_idx  = current_phase['impairment']   # 0, 1, 2

        print(f"\n[Epoch {epoch+1:3d}]  "
              f"Train {train_loss:.4f} / {train_acc:.2f}%  |  "
              f"Val {val_loss:.4f} / {val_acc:.2f}%  |  "
              f"Gap {train_acc-val_acc:.2f}%  |  "
              f"LR {optimizer.param_groups[0]['lr']:.2e}  |  "
              f"Mem {mem_gb:.2f} GB")

        # История
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['grad_norms'].append(np.mean(grad_norms) if grad_norms else 0)
        history['memory_usage'].append(mem_gb)

        scheduler.step()

        # Сохранение: лучшая внутри текущей фазы
        phase_improved = (phase_name not in best_per_phase or
                          val_acc > best_per_phase[phase_name])

        if phase_improved:
            best_per_phase[phase_name] = val_acc
            history['best_per_phase'][phase_name] = (val_acc, epoch)

            phase_path = f'best_model_phase{phase_idx}.pth'
            save_checkpoint(phase_path, model, optimizer, epoch,
                            val_acc, class_names, history, phase_name)
            print(f"   [✓] NEW BEST [{phase_name}]: {val_acc:.2f}%"
                  f"  → {phase_path}")

            # Confusion matrix для этой фазы
            save_confusion_matrix(
                all_preds, all_labels, class_names,
                save_path=f'confusion_matrix_phase{phase_idx}.png')

            patience_counter = 0
        else:
            patience_counter += 1

        # Сохранение: лучшая за всё обучение
        if val_acc > best_val_acc_global:
            best_val_acc_global = val_acc
            save_checkpoint('best_model.pth', model, optimizer, epoch,
                            val_acc, class_names, history, phase_name)
            print(f"   [★] NEW GLOBAL BEST: {best_val_acc_global:.2f}%"
                  f"  → best_model.pth")

        # Early stopping
        patience = TRAIN_CONFIG['early_stopping_patience']
        if patience_counter >= patience:
            print(f"\n[!] Early stopping — {patience} эпох без улучшения"
                  f" в фазе [{phase_name}]")
            # Не прерываем весь трейн — просто логируем.
            # Если хочешь полный стоп — раскомментируй break:
            # break

        # Периодические графики
        if (epoch + 1) % 10 == 0:
            save_plots(history)

        clear_memory()

    # Итог
    print(f"\n{'='*70}")
    print(" ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"  Global Best Val Acc: {best_val_acc_global:.2f}%  (best_model.pth)")
    print(f"\n  По фазам:")
    for ph_name, (acc, ep) in history['best_per_phase'].items():
        ph_idx = next(p['impairment'] for p in CURRICULUM_PHASES
                      if p['name'] == ph_name)
        print(f"    {ph_name}")
        print(f"      Accuracy: {acc:.2f}%  |  Epoch: {ep+1}"
              f"  →  best_model_phase{ph_idx}.pth")
    print(f"\n  Final Val Acc:  {history['val_acc'][-1]:.2f}%")
    print(f"  Epochs done:    {epoch + 1}")
    print(f"{'='*70}\n")

    save_plots(history)
    print("\n Файлы:")
    print("  best_model.pth          ← лучшая за всё обучение")
    for i in range(len(CURRICULUM_PHASES)):
        if os.path.exists(f'best_model_phase{i}.pth'):
            print(f"  best_model_phase{i}.pth   ← лучшая Phase {i+1}")
    print("  training_report.png")
    print("  confusion_matrix_phase*.png")


if __name__ == "__main__":
    train()
