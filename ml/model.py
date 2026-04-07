"""
XCiT Small для RF модуляций [2, 1024] -> 57 классов

ИСПРАВЛЕНИЕ: reshape [B, 3, 32, 32] → [B, 3, 8, 128]
  Было:  32 сэмпла в строке  → разрывы каждые 32 отсчёта
  Стало: 128 сэмплов в строке → разрывы каждые 128 отсчётов (в 4x лучше)
  Каждый патч XCiT захватывает ~18 последовательных точек сигнала вместо ~4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model


class XCiTSmall1d(nn.Module):

    def __init__(self, num_classes=57, pretrained=False):
        super().__init__()

        # [B, 2, 1024] → [B, 3, 1024]  (все stride=1 — длина сохраняется)
        self.preprocess = nn.Sequential(
            nn.Conv1d(2,  32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64,  3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(3),
            nn.GELU(),
        )

        # XCiT Small: img_size=112 → 7×7 = 49 патчей (patch_size=16)
        self.backbone = create_model(
            'xcit_small_12_p16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
            img_size=112,
            drop_path_rate=0.15,
        )

        if pretrained and hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Linear(
                self.backbone.head.in_features, num_classes)

    def forward(self, x):
        """
        x: [B, 2, 1024]  →  [B, num_classes]

        Reshape: [B, 3, 1024]
              → [B, 3, 8, 128]      ← 8 строк по 128 последовательных отсчётов
              → [B, 3, 112, 112]    ← bilinear interp (14x вверх, 0.875x вправо)
        """
        B = x.size(0)

        # [B, 2, 1024] → [B, 3, 1024]
        x = self.preprocess(x)

        # ↓ ИСПРАВЛЕНО: было view(B, 3, 32, 32) — разрывы каждые 32 отсчёта
        # Теперь 8 строк × 128 столбцов = 1024, разрывы каждые 128 отсчётов
        x = x.view(B, 3, 8, 128)

        # [B, 3, 8, 128] → [B, 3, 112, 112]
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

        # XCiT backbone
        x = self.backbone(x)
        return x


def get_model(num_classes=57, pretrained=False):
    return XCiTSmall1d(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = get_model(num_classes=57)
    x = torch.randn(2, 2, 1024)
    with torch.no_grad():
        out = model(x)
    total, trainable = count_parameters(model)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {total:,} (trainable {trainable:,})")
    # Проверяем reshape
    dummy = torch.randn(2, 3, 1024)
    reshaped = dummy.view(2, 3, 8, 128)
    print(f"Reshape check: {dummy.shape} → {reshaped.shape}  ✓")
