import torch
import torch.nn as nn
from timm.models import create_model
from config.settings import NUM_CLASSES


class XCiT1d(nn.Module):
    """
    Адаптированная модель XCiT для 1D сигнал1,    ов (I/Q).
    
    Архитектура:
    1. Preprocessing: [2, 1024] -> [64, 256] через Conv1d
    2. Reshape: [64, 256] -> [64, 16, 16] для совместимости с патчами
    3. XCiT backbone: обработка как 2D изображение 16x16
    """
    def __init__(self, num_classes=None, pretrained=False):
        super().__init__()
        
        # Если num_classes не указан, берем из настроек
        if num_classes is None:
            num_classes = NUM_CLASSES
        
        # Препроцессинг: 2 канала (I/Q) -> 64 каналов, длина 1024 -> 256
        self.preprocess = nn.Sequential(
            # Первый блок: 2 -> 16, 1024 -> 512
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            
            # Второй блок: 16 -> 32, 512 -> 256
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Третий блок: 32 -> 64, 256 -> 256 (без изменения длины)
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        # XCiT модель с правильными размерами
        # img_size должен быть кратен patch_size (16)
        # У нас будет: [batch, 64, 16, 16]
        self.backbone = create_model(
            'xcit_small_12_p16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=64,  # входные каналы после preprocessing
            img_size=(16, 16),  # размер "изображения" кратный patch_size=16
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, 2, 1024] - I/Q сигнал
            
        Returns:
            [batch, num_classes] - логиты классов
        """
        # [batch, 2, 1024] -> [batch, 64, 256]
        x = self.preprocess(x)
        
        # [batch, 64, 256] -> [batch, 64, 16, 16]
        # Разбиваем 256 точек на 16x16 сетку
        batch_size = x.size(0)
        x = x.view(batch_size, 64, 16, 16)
        
        # Пропускаем через XCiT
        x = self.backbone(x)
        
        return x


class SimpleCNN1d(nn.Module):
    """
    Простая CNN для сравнения (baseline модель)
    """
    def __init__(self, num_classes=57):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 2 -> 32
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2: 32 -> 64
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 4: 128 -> 256
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_type='xcit', num_classes=57, pretrained=False):
    if model_type == 'xcit':
        return XCiT1d(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'simple_cnn':
        return SimpleCNN1d(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
