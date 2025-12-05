# ml/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Оптимальный Residual блок для 25M модели."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout1d(p=0.1)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        
        return out

class ModulationNet25M(nn.Module):
    """Оптимальная модель на ~25 млн параметров для 58 классов модуляций."""
    def __init__(self, num_classes=58):
        super(ModulationNet25M, self).__init__()
        
        # Начальные слои
        self.init_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout1d(p=0.1)
        )

        # Основные ResNet слои
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2)    # 1024 -> 256
        self.layer2 = self._make_layer(128, 256, num_blocks=3, stride=2)   # 256 -> 128
        self.layer3 = self._make_layer(256, 512, num_blocks=4, stride=2)   # 128 -> 64
        self.layer4 = self._make_layer(512, 512, num_blocks=2, stride=2)   # 64 -> 32
        
        # Глобальное pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Сбалансированный классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            
            nn.Linear(512 * 2, 2048),  # *2 из-за avg + max pooling
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Вход: [batch, 2, 1024]
        x = self.init_conv(x)      # [batch, 64, 256]
        x = self.layer1(x)         # [batch, 128, 128]
        x = self.layer2(x)         # [batch, 256, 64]
        x = self.layer3(x)         # [batch, 512, 32]
        x = self.layer4(x)         # [batch, 512, 16]
        
        # Комбинированное pooling
        avg_out = self.avgpool(x).squeeze(-1)  # [batch, 512]
        max_out = self.maxpool(x).squeeze(-1)  # [batch, 512]
        x = torch.cat([avg_out, max_out], dim=1)  # [batch, 1024]
        
        x = self.classifier(x)
        
        return x

# Альтернативная компактная модель если нужна точная настройка
class Compact25MNet(nn.Module):
    """Компактная модель на ~25M параметров."""
    def __init__(self, num_classes=58):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv1d(2, 96, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Dropout1d(0.1),
            
            # Stage 2
            nn.Conv1d(96, 192, 3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout1d(0.15),
            
            # Stage 3
            nn.Conv1d(192, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout1d(0.2),
            
            # Stage 4
            nn.Conv1d(384, 768, 3, padding=1, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Conv1d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout1d(0.25),
            
            # Stage 5
            nn.Conv1d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Conv1d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Функция для проверки параметров
def print_model_summary(model, name="Model"):
    """Печатает детальную информацию о модели."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Детальный подсчет
    conv_params = 0
    linear_params = 0
    bn_params = 0
    
    for name_param, param in model.named_parameters():
        if 'conv' in name_param:
            conv_params += param.numel()
        elif 'fc' in name_param or 'classifier' in name_param:
            linear_params += param.numel()
        elif 'bn' in name_param or 'norm' in name_param:
            bn_params += param.numel()
    
    print(f"\n{'='*60}")
    print(f"{name} Summary:")
    print(f"{'='*60}")
    print(f"Total parameters:      {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable params:  {total_params - trainable_params:,}")
    print(f"\nParameter distribution:")
    print(f"  Convolutional layers: {conv_params:,} ({conv_params/total_params:.1%})")
    print(f"  Linear layers:        {linear_params:,} ({linear_params/total_params:.1%})")
    print(f"  BatchNorm layers:     {bn_params:,} ({bn_params/total_params:.1%})")
    
    return total_params

if __name__ == "__main__":
    print("=== МОДЕЛИ НА 25 МЛН ПАРАМЕТРОВ ДЛЯ КЛАССИФИКАЦИИ МОДУЛЯЦИЙ ===")
    
    # Тестируем обе модели
    models = [
        ("ModulationNet25M", ModulationNet25M),
        ("Compact25MNet", Compact25MNet)
    ]
    
    for name, ModelClass in models:
        print(f"\n\n{'-'*60}")
        print(f"Testing: {name}")
        print(f"{'-'*60}")
        
        model = ModelClass(num_classes=58)
        total_params = print_model_summary(model, name)
        
        # Проверка forward pass
        x = torch.randn(4, 2, 1024)
        y = model(x)
        print(f"\nForward pass test:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Проверка на GPU
        if torch.cuda.is_available():
            try:
                model_gpu = ModelClass(num_classes=58).cuda()
                x_gpu = torch.randn(8, 2, 1024).cuda()
                y_gpu = model_gpu(x_gpu)
                print(f"  GPU test (batch=8): OK")
                
                # Оценка памяти
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"  GPU memory: {gpu_memory:.1f} MB")
            except Exception as e:
                print(f"  GPU test failed: {str(e)[:50]}...")
        
        print(f"{'-'*60}")