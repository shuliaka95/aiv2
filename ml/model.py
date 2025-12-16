# ml/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import os

class XCiT1d(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 57,
        xcit_version: str = "small_12_p16_224",
        drop_path_rate: float = 0.15,
        drop_rate: float = 0.5,
        ds_method: str = "chunk",
        ds_rate: int = 4
    ):
        super().__init__()
        
        model_name = f"xcit_{xcit_version}" if not xcit_version.startswith("xcit_") else xcit_version
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
        
        W = self.backbone.num_features
        
        self.grouper = nn.Conv1d(W, num_classes, kernel_size=1)
        
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(input_channels, W, ds_rate)
        elif ds_method == "chunk":
            self.backbone.patch_embed = Chunker(input_channels, W, ds_rate)
        else:
            raise ValueError(f"Unsupported downsampling method: {ds_method}")
        
        self.backbone.head = nn.Identity()
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] XCiT1d ({xcit_version}) создана")
        print(f"  Параметров: {total_params:,}")
        print(f"  Dropout: {drop_rate}, DropPath: {drop_path_rate}")
        print(f"  Метод понижения частоты: {ds_method} (x{ds_rate})")
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.grouper.weight, mode='fan_out', nonlinearity='relu')
        if self.grouper.bias is not None:
            nn.init.constant_(self.grouper.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        mdl = self.backbone
        B = x.shape[0]
        
        x = self.backbone.patch_embed(x)
        Hp, Wp = x.shape[-1], 1
        
        pos_encoding = mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp).permute(0, 2, 1)
        x = x.transpose(1, 2) + pos_encoding
        
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)
        
        cls_tokens = mdl.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in mdl.cls_attn_blocks:
            x = blk(x)
        
        x = mdl.norm(x)
        cls_token = x[:, 0, :].unsqueeze(-1)
        x = self.grouper(cls_token).squeeze(-1)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return x

class ConvDownSampler(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=ds_rate * 2 + 1,
            stride=ds_rate,
            padding=ds_rate,
            bias=False
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Chunker(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 4):
        super().__init__()
        self.ds_rate = ds_rate
        self.embed = nn.Conv1d(in_chans, embed_dim, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()
        self.pool = nn.AvgPool1d(kernel_size=ds_rate, stride=ds_rate)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class ModulationNet25M(nn.Module):
    def __init__(self, num_classes=57):
        super().__init__()
        
        self.model = XCiT1d(
            input_channels=2,
            num_classes=num_classes,
            xcit_version="small_12_p16_224",
            drop_path_rate=0.15,
            drop_rate=0.5,
            ds_method="chunk",
            ds_rate=4
        )
        
    def forward(self, x):
        return self.model(x)

def create_training_plots(train_losses, train_accs, val_losses, val_accs, lr_history, save_path="training_plots"):
    """Создает и сохраняет графики процесса обучения."""
    
    os.makedirs(save_path, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 1. График потерь
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.title('Loss во время обучения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 2. График точности
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy во время обучения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 3. Gap между train и val
    plt.subplot(2, 2, 3)
    if len(train_accs) == len(val_accs):
        gaps = [train_accs[i] - val_accs[i] for i in range(len(train_accs))]
        plt.plot(epochs, gaps, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Эпохи')
        plt.ylabel('Gap (%)')
        plt.title('Разрыв между Train и Val Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # 4. Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(lr_history) + 1), lr_history, 'purple', linewidth=2)
    plt.xlabel('Эпохи')
    plt.ylabel('Learning Rate')
    plt.title('Изменение Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_path}/training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Сводный график всех метрик
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, alpha=0.6, label='Train Loss')
    ax1.plot(epochs, val_losses, color=color, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(epochs, train_accs, color=color, alpha=0.6, label='Train Acc')
    ax2.plot(epochs, val_accs, color=color, linestyle='--', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title('Сводная статистика обучения')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Графики сохранены в папку: {save_path}/")


def save_training_report(train_losses, train_accs, val_losses, val_accs, lr_history):
    """Создает текстовый отчет для курсовой работы."""
    
    if not train_accs or not val_accs:
        print("  Нет данных для создания отчета")
        return
    
    report = f"""
ОТЧЕТ ПО ОБУЧЕНИЮ МОДЕЛИ КЛАССИФИКАЦИИ РАДИОСИГНАЛОВ
======================================================

1. ПАРАМЕТРЫ МОДЕЛИ
-------------------
- Архитектура: XCiT1d (small_12_p16_224)
- Количество классов: 57 модуляций
- Dropout: 0.5
- DropPath (Stochastic Depth): 0.15
- Метод понижения частоты: chunk (x4)
- Learning Rate: начальный {lr_history[0]:.2e}, финальный {lr_history[-1]:.2e}

2. РЕЗУЛЬТАТЫ ОБУЧЕНИЯ
----------------------
- Лучшая точность валидации: {max(val_accs):.2f}%
- Финальная точность валидации: {val_accs[-1]:.2f}%
- Финальная точность обучения: {train_accs[-1]:.2f}%
- Разрыв между train и val (Gap): {train_accs[-1] - val_accs[-1]:.2f}%
- Лучший Loss: {min(val_losses):.4f}
- Количество эпох обучения: {len(train_accs)}

3. АНАЛИЗ ПЕРЕОБУЧЕНИЯ
-----------------------
- Финальный разрыв (train-val): {train_accs[-1] - val_accs[-1]:.2f}%
- Средний разрыв (последние 10 эпох): {np.mean([train_accs[i] - val_accs[i] for i in range(-min(10, len(train_accs)), 0)]) if len(train_accs) >= 10 else 0:.2f}%
- Динамика обучения: {'Стабильная' if train_accs[-1] > train_accs[0] + 10 else 'Медленная'}

4. СТАТИСТИКА ПО ЭПОХАМ
------------------------
"""
    
    # Добавляем таблицу с ключевыми эпохами
    key_epochs = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    report += "\nЭпоха | Train Acc | Val Acc  | Loss   | LR\n"
    report += "------|-----------|----------|--------|---------\n"
    
    for epoch in key_epochs:
        if epoch - 1 < len(train_accs):
            idx = epoch - 1
            report += f"{epoch:5d} | {train_accs[idx]:8.2f}% | {val_accs[idx]:8.2f}% | {train_losses[idx]:6.4f} | {lr_history[idx]:.2e}\n"
    
    # Сохраняем в файл
    with open("training_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(" Текстовый отчет сохранен в 'training_report.txt'")

if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ МОДЕЛИ ===")
    
    model = ModulationNet25M(num_classes=57)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Всего параметров: {total_params:,}")
    
    batch_size = 4
    x = torch.randn(batch_size, 2, 1024)
    y = model(x)
    
    print(f"Вход: {x.shape}")
    print(f"Выход: {y.shape}")
    
    print("\n Модель готова к использованию!")
