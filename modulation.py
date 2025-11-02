import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pickle
import warnings 
import os
import math 

# Фильтр для подавления стандартных предупреждений PyTorch
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`", UserWarning)

# Установка глобального Seed для обеспечения воспроизводимости результатов
torch.manual_seed(42)
np.random.seed(42)

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ И НАСТРОЙКИ ДАТАСЕТА ---
MODULATIONS = ['8PSK', 'AM', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', '16QAM', '64QAM', 'QPSK', 'WBFM']
NUM_MODS = len(MODULATIONS)
SIGNAL_LENGTH = 128
# Диапазон SNR от -20 дБ до 18 дБ с шагом 2 дБ
SNRS = np.arange(-20, 18 + 2, 2) 

# Ограничение на количество сэмплов, используемых для обучения и валидации
MAX_SAMPLES_TO_USE = 200000 
RML_FILEPATH = 'RML2016.10a_dict.pkl' 

# Маппинг имен RML датасета на унифицированные метки классов модуляции
RML_MAPPING = {
    'AM-DSB': 'AM', 'QAM16': '16QAM', 'QAM64': '64QAM',
    '8PSK': '8PSK', 'BPSK': 'BPSK', 'CPFSK': 'CPFSK', 
    'GFSK': 'GFSK', 'PAM4': 'PAM4', 'QPSK': 'QPSK', 'WBFM': 'WBFM'
}
MOD_TO_IDX = {mod: idx for idx, mod in enumerate(MODULATIONS)}

# Глобальная переменная для устройства (CPU/GPU)
DEVICE = None 

# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ И АРХИТЕКТУРА ---

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Функция потерь со сглаживанием меток (Label Smoothing). 
    Применяется для улучшения регуляризации и предотвращения излишней 
    уверенности модели, что способствует лучшей обобщающей способности.
    """
    def __init__(self, eps=0.1, reduction='mean'):
        super().__init__()
        self.eps = eps # Коэффициент сглаживания (10%)
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1] # Количество классов
        log_preds = F.log_softmax(output, dim=-1)
        # NLL Loss: потери при жестких метках
        nll_loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # Smooth Loss: потери при равномерном распределении
        smooth_loss = -log_preds.sum(dim=-1)
        
        if self.reduction == 'mean':
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        
        # LSCE: (1 - epsilon) * NLL + (epsilon / C) * Smooth
        loss = (1. - self.eps) * nll_loss + (self.eps / c) * smooth_loss
        return loss

class ResidualBlock(nn.Module):
    """
    Остаточный Блок (Residual Block) для 1D CNN. 
    Использует skip-connection для обхода части слоя, что помогает 
    обучать очень глубокие сети, предотвращая проблему затухания градиентов.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        # Первая свертка: понижение/повышение размерности, если stride > 1 или каналы меняются
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Вторая свертка: сохраняет размерность
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        # Если входные и выходные каналы/размеры не совпадают, используем 1x1 свертку для выравнивания
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Основной путь (Residual Path)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip Connection: сложение выхода основного пути с выходом из Shortcut
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiHeadAttention(nn.Module):
    """
    Многоголовый Механизм Само-Внимания (Multi-Head Self-Attention, MHA).
    Позволяет модели одновременно фокусироваться на различных частях (головах) 
    сигнала, повышая способность к захвату сложных, нелинейных зависимостей.
    """
    def __init__(self, input_features, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_features // num_heads # Размерность каждой головы
        self.input_features = input_features
        
        # Проекции Q(Query), K(Key), V(Value) через 1x1 свертки (аналог линейных слоев для 1D)
        self.conv_q = nn.Conv1d(input_features, input_features, 1)
        self.conv_k = nn.Conv1d(input_features, input_features, 1)
        self.conv_v = nn.Conv1d(input_features, input_features, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        # Финальная проекция для объединения выходов всех голов
        self.out_conv = nn.Conv1d(input_features, input_features, 1)
    
    def forward(self, x):
        B, C, L = x.shape # (Batch, Channels/Features, Length)
        
        # 1. Проекция Q, K, V и разделение на головы
        Q = self.conv_q(x).view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2) # (B, H, L, D_h)
        K = self.conv_k(x).view(B, self.num_heads, self.head_dim, L)                   # (B, H, D_h, L)
        V = self.conv_v(x).view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2) # (B, H, L, D_h)
        
        # 2. Вычисление карты внимания (Scaled Dot-Product Attention)
        # Score = Q * K / sqrt(D_h)
        energy = torch.matmul(Q, K) / (self.head_dim ** 0.5) # (B, H, L, L) - Матрица сходства
        attention = self.softmax(energy) # Веса внимания
        
        # 3. Применение внимания: Out = Attention * V
        weighted_output = torch.matmul(attention, V) # (B, H, L, D_h)
        
        # 4. Объединение голов (Concatenation) и финальная проекция
        weighted_output = weighted_output.permute(0, 1, 3, 2).contiguous().view(B, C, L) 
        out = self.out_conv(weighted_output)
        
        # Добавление Skip Connection (Residual Connection) для Attention
        out = out + x 
        
        return out

class ResNetAttentionCNN(nn.Module):
    """
    Углубленная ResNet-подобная архитектура (4 слоя) с Multi-Head Attention и 
    Global Average Pooling (GAP). Оптимизирована для максимальной точности 
    классификации радиочастотных модуляций (RMC).
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Начальный сверточный слой: 2 канала (I/Q) -> 64 признаковых карты. Уменьшение длины.
        self.initial_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Углубленные слои ResNet (4 слоя)
        self.layer1 = self._make_layer(64, 128, stride=1, num_blocks=3)  # L=64
        self.layer2 = self._make_layer(128, 256, stride=2, num_blocks=4) # L=32
        self.layer3 = self._make_layer(256, 512, stride=2, num_blocks=6) # L=16
        # Добавлен 4-й слой для повышения глубины и извлечения сложных паттернов
        self.layer4 = self._make_layer(512, 512, stride=1, num_blocks=2) # L=16
        
        # Механизм Multi-Head Attention с 8 головами
        self.attention = MultiHeadAttention(input_features=512, num_heads=8)
        
        # Глобальное Усредняющее Объединение: сжимает временную ось до 1.
        # Повышает робастность к сдвигу и уменьшает количество параметров.
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(), # Из (B, 512, 1) -> (B, 512)
            nn.Dropout(0.6), # Высокий Dropout для сильной регуляризации
            nn.Linear(512, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes) # Финальный выход
        )
        
    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        """Вспомогательная функция для создания последовательности остаточных блоков."""
        layers = []
        # Первый блок может изменять размерность и/или каналы
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Последующие блоки сохраняют размерность
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_conv(x) 
        out = self.layer1(out)     
        out = self.layer2(out)     
        out = self.layer3(out)     
        out = self.layer4(out)     
        
        # Применение механизма MHA
        out = self.attention(out) 
        
        # Применение GAP
        out = self.global_pool(out) 
        
        out = self.classifier(out)
        return out

# --- ЗАГРУЗКА ДАННЫХ RML ---

def load_and_preprocess_rml(filepath):
    """
    Загрузка данных из RML датасета, фильтрация по требуемым модуляциям 
    и нормализация мощности.
    """
    try:
        print(f"Загрузка файла датасета RML: {filepath}...")
        with open(filepath, 'rb') as f:
            Xd = pickle.load(f, encoding='latin1')
    except FileNotFoundError:
        print(f"\n!!! ОШИБКА: Файл RML датасета не найден. Убедитесь, что '{filepath}' находится в каталоге.")
        return None, None
    except Exception as e:
        print(f"\n!!! Ошибка при загрузке RML файла: {e}")
        return None, None
    
    filtered_data = []
    filtered_labels = []
    
    for rml_mod_type, snr in Xd.keys():
        mod_type_map = RML_MAPPING.get(rml_mod_type)

        if mod_type_map in MODULATIONS:
            iq_data = Xd[(rml_mod_type, snr)]
            
            # Нормализация по мощности: деление I/Q данных на среднеквадратичную мощность
            power = np.sqrt(np.mean(iq_data**2, axis=(1, 2), keepdims=True))
            iq_data_normalized = iq_data / (power + 1e-8)
            
            filtered_data.append(iq_data_normalized)
            
            label_idx = MOD_TO_IDX[mod_type_map]
            filtered_labels.extend([label_idx] * iq_data_normalized.shape[0])
            
    if not filtered_data:
        print("!!! ОШИБКА: Не удалось найти требуемые типы модуляции.")
        return None, None

    X = np.concatenate(filtered_data, axis=0).astype(np.float32)
    Y = np.array(filtered_labels, dtype=np.int64) 
    
    print(f"\nRML Загружен: {X.shape[0]} сэмплов")
    
    # Ограничение сэмплов для ускорения обучения, если общее количество слишком велико
    if X.shape[0] > MAX_SAMPLES_TO_USE:
        print(f"Ограничение: Использование {MAX_SAMPLES_TO_USE} случайных сэмплов.")
        indices = np.random.choice(X.shape[0], MAX_SAMPLES_TO_USE, replace=False)
        X = X[indices]
        Y = Y[indices]

    return X, Y

class RmlDataset(Dataset):
    """
    Пользовательский PyTorch Dataset, который предоставляет комплексную
    аугментацию данных для имитации реальных помех в радиоэфире.
    """
    def __init__(self, X, Y, augment=False):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.augment = augment
        
        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Инициализация self.L (длины сигнала) ---
        # Ошибка 'AttributeError: RmlDataset object has no attribute L' устраняется здесь.
        # Длина сигнала берется из третьего измерения входного numpy массива.
        if X.ndim < 3:
            raise ValueError("Входной массив данных (X) должен иметь как минимум 3 измерения (Сэмплы, Каналы, Длина).")
        self.L = X.shape[2] 
        # Дополнительная инициализация устройства для работы аугментации в worker-процессах
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.Y)

    def _augment(self, x):
        """
        Реализация продвинутой аугментации (Data Augmentation) для робастного обучения.
        Каждая помеха применяется с определенной вероятностью.
        """
        
        # Извлечение I и Q компонентов
        I = x[0, :]
        Q = x[1, :]
        
        # Преобразование в комплексный вид S = I + jQ (удобно для манипуляций с фазой/частотой)
        S_complex = I + 1j * Q
        
        # --- 1. Случайный Фазовый Сдвиг (Phase Shift) ---
        if torch.rand(1) < 0.7: 
            # Случайный угол в диапазоне [-pi, pi]
            # Используем device из входного тензора 'x'
            phase_shift = (torch.rand(1) * 2 * math.pi - math.pi).to(x.device)
            S_complex *= torch.exp(1j * phase_shift) # Вращение созвездия
        
        # --- 2. Смещение Несущей Частоты (Carrier Frequency Offset, CFO) ---
        if torch.rand(1) < 0.8: 
            # CFO моделирует расхождение генераторов частоты между TX и RX
            freq_offset_norm = (torch.rand(1) * 0.1 - 0.05).to(x.device) # Диапазон [-0.05, 0.05]
            # Используем self.L, которое теперь гарантированно инициализировано
            t_axis = torch.arange(self.L, dtype=torch.float32, device=x.device) 
            # Применение экспоненциального сдвига exp(j * 2*pi * f_offset * t)
            cfo_factor = torch.exp(1j * 2 * math.pi * freq_offset_norm * t_axis / self.L)
            S_complex *= cfo_factor
            
        # --- 3. Многолучевое Замирание (Rayleigh Fading) ---
        if torch.rand(1) < 0.5: 
            # Моделирует отражения от объектов (здания, горы), искажающие сигнал
            # Генерация Рэлеевского распределения для магнитуды и равномерного для фазы
            # Используем self.L
            rayleigh_magnitude = torch.abs(torch.randn(self.L, device=x.device) + 1j * torch.randn(self.L, device=x.device)) / math.sqrt(2)
            rayleigh_phase = (torch.rand(self.L, device=x.device) * 2 * math.pi - math.pi)
            fading_factor = rayleigh_magnitude * torch.exp(1j * rayleigh_phase)
            
            # Сглаживание фактора замирания для имитации доплеровского эффекта
            window_size = 5
            kernel = torch.ones(window_size, device=x.device) / window_size
            # Для свертки нужно изменить форму
            fading_factor_smooth = F.conv1d(fading_factor.view(1, 1, self.L).real, kernel.view(1, 1, window_size), padding=window_size//2).squeeze() + \
                                   1j * F.conv1d(fading_factor.view(1, 1, self.L).imag, kernel.view(1, 1, window_size), padding=window_size//2).squeeze()
                                   
            S_complex *= fading_factor_smooth

        # --- 4. Добавление Гауссова шума (для дополнительной робастности) ---
        if torch.rand(1) < 0.9: 
            # Дополнительный, небольшой шум к уже существующему в датасете
            noise_level = torch.rand(1, device=x.device) * 0.01 + 0.005 # Уровень шума [0.005, 0.015]
            noise = torch.randn_like(S_complex.real) + 1j * torch.randn_like(S_complex.imag)
            S_complex += noise * noise_level

        # Обратное преобразование в 2-канальный формат (I и Q)
        x[0, :] = S_complex.real
        x[1, :] = S_complex.imag
        
        return x

    def __getitem__(self, idx):
        x = self.X[idx].clone() # Клонирование тензора, чтобы аугментация не меняла исходные данные
        y = self.Y[idx]
        
        if self.augment:
            x = self._augment(x)
            
        return x, y

# --- ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ ---

def train_model(model, train_loader, val_loader, num_epochs=100, device=None, save_best=True):
    """
    Основной цикл обучения PyTorch модели. Включает оптимизатор AdamW, 
    Label Smoothing Loss, планировщик LR и раннюю остановку (Early Stopping).
    """
    if device is None:
        raise ValueError("DEVICE не инициализирован.")
        
    # Использование AdamW с сильной деградацией весов для лучшей регуляризации
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) 
    
    # Функция потерь: Label Smoothing Cross Entropy
    criterion = LabelSmoothingCrossEntropy(eps=0.1) 
    
    # Планировщик LR: уменьшение LR вдвое (factor=0.5), если val_loss не улучшается 5 эпох (patience=5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    # Словари для хранения истории обучения для построения графиков
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'lr': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Цикл обучения
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Обрезка градиента (Gradient Clipping) для предотвращения взрывных градиентов
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
        train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        
        # Цикл валидации
        model.eval()
        val_loss = 0.0
        val_total_samples = 0
        val_correct_predictions = 0
        
        with torch.no_grad(): # Отключение вычисления градиентов для экономии памяти
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_total_samples += labels.size(0)
                val_correct_predictions += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        val_loss /= val_total_samples
        val_accuracy = val_correct_predictions / val_total_samples
        
        scheduler.step(val_loss) # Обновление планировщика LR

        # Сводная строка для эпохи
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:2d}/{num_epochs} | '
              f'Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | '
              f'LR: {current_lr:.6f}')
        
        # Запись истории
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)
        
        # Ранняя остановка (Early Stopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_best:
                # Сохранение весов лучшей модели на основе минимальных валидационных потерь
                torch.save(model.state_dict(), 'best_rmc_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 10: # Остановка, если нет улучшения в течение 10 эпох
                print(f"\n[INFO] Ранняя остановка сработала на эпохе {epoch+1}/{num_epochs}.")
                if save_best:
                    model.load_state_dict(torch.load('best_rmc_model.pth'))
                break
    
    if save_best:
        print("\nОбучение завершено. Загружена лучшая сохраненная модель.")
    
    return history

def evaluate_by_snr(model, rml_filepath, snr_range, device=None):
    """
    Оценка точности модели в зависимости от SNR на полном диапазоне RML.
    Ключевая метрика для оценки робастности модели.
    """
    if device is None:
        raise ValueError("DEVICE не инициализирован.")
        
    print("\n--- 5. ОЦЕНКА ТОЧНОСТИ ПО SNR ---")
    
    try:
        with open(rml_filepath, 'rb') as f:
            Xd = pickle.load(f, encoding='latin1')
    except:
        print(f"[ERROR] Не удалось загрузить файл RML для оценки.")
        return {snr: 0.0 for snr in snr_range}, None, None

    model.eval()
    accuracy_vs_snr = {}
    
    for snr_db in snr_range:
        X_snr = []
        Y_snr_labels = []
        
        # Подготовка данных для текущего SNR
        for rml_mod_type in RML_MAPPING.keys():
            if (rml_mod_type, snr_db) in Xd:
                iq_data = Xd[(rml_mod_type, snr_db)]
                
                # Нормализация данных (также как и при обучении)
                power = np.sqrt(np.mean(iq_data**2, axis=(1, 2), keepdims=True))
                iq_data_normalized = iq_data / (power + 1e-8)
                
                X_snr.append(iq_data_normalized.astype(np.float32))
                
                label_idx = MOD_TO_IDX[RML_MAPPING.get(rml_mod_type)]
                Y_snr_labels.extend([label_idx] * iq_data_normalized.shape[0])

        if not X_snr:
            accuracy_vs_snr[snr_db] = 0.0
            continue
        
        X_snr = np.concatenate(X_snr, axis=0)
        
        # Предсказание на GPU
        X_snr_tensor = torch.from_numpy(X_snr).to(device)
        
        with torch.no_grad():
            outputs = model(X_snr_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        # Вычисление точности
        Y_true_labels = np.array(Y_snr_labels)
        Y_pred_labels = predicted.cpu().numpy()
        
        correct_predictions = np.sum(Y_pred_labels == Y_true_labels)
        accuracy = correct_predictions / len(Y_true_labels)
        accuracy_vs_snr[snr_db] = accuracy
        print(f"SNR: {snr_db:3d} dB | Accuracy: {accuracy:.4f} ({len(Y_true_labels)} сэмплов)")

    return accuracy_vs_snr, None, None # Возвращаем только accuracy_vs_snr для этой функции


def plot_results(accuracy_vs_snr, cm, history):
    """
    Визуализация ключевых результатов обучения: 
    1. Точность vs SNR (робастность).
    2. Нормализованная Матрица Ошибок (Confusion Matrix).
    3. История сходимости (Loss/Accuracy).
    """
    
    print("\n[INFO] Начинается построение графика...")
    
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    
    # --- График 1: Точность против SNR ---
    snrs = sorted(list(accuracy_vs_snr.keys()))
    accuracies = [accuracy_vs_snr[s] for s in snrs]
    
    axs[0].plot(snrs, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=5)
    axs[0].set_title('Точность классификации в зависимости от SNR', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('SNR (дБ)', fontsize=12)
    axs[0].set_ylabel('Точность', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_xticks(snrs[::2]) 
    
    # --- График 2: Нормализованная Матрица Ошибок (Confusion Matrix) ---
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Нормализация по истинным классам
    
    im = axs[1].imshow(cm, interpolation='nearest', cmap='Blues') 
    
    axs[1].set_title('Нормализованная матрица ошибок (Validation Set)', fontsize=14, fontweight='bold')
    
    tick_marks = np.arange(len(MODULATIONS))
    axs[1].set_xticks(tick_marks)
    axs[1].set_yticks(tick_marks)
    axs[1].set_xticklabels(MODULATIONS, rotation=45, ha="right", fontsize=10)
    axs[1].set_yticklabels(MODULATIONS, fontsize=10)
    axs[1].set_ylabel('Истинный класс', fontsize=12)
    axs[1].set_xlabel('Предсказанный класс', fontsize=12)

    # Добавление значений в ячейки
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        axs[1].text(j, i, f'{cm[i, j]:.2f}',
                     horizontalalignment="center",
                     color=color,
                     fontsize=8)

    # --- График 3: История обучения ---
    axs[2].plot(history['accuracy'], label='Точность обучения (Train)', color='#2ca02c', linewidth=2)
    axs[2].plot(history['val_accuracy'], label='Точность валидации (Validation)', color='#d62728', linewidth=2)
    
    # График потерь на второй оси
    ax2 = axs[2].twinx()
    ax2.plot(history['loss'], label='Потери обучения', color='#ff7f0e', linestyle=':', alpha=0.7)
    ax2.plot(history['val_loss'], label='Потери валидации', color='#9467bd', linestyle='--', alpha=0.7)

    axs[2].set_title('История сходимости модели', fontsize=14, fontweight='bold')
    axs[2].set_xlabel('Эпоха', fontsize=12)
    axs[2].set_ylabel('Точность', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('Потери', fontsize=12, color='#ff7f0e')
    
    # Объединение легенд
    lines, labels = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=9)
    
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Результаты классификации модуляции (Углубленный ResNet-MHA с расширенной аугментацией)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    # Сохранение графика
    try:
        plt.savefig('rmc_results.png', dpi=300)
        print("\n[INFO] График сохранен в файл: rmc_results.png")
    except Exception as e:
        print(f"\n[WARNING] Не удалось сохранить файл графика. Ошибка: {e}")
        
    plt.show()


# --- ОСНОВНАЯ ФУНКЦИЯ ПРОГРАММЫ ---
# -------------------------------------------------------------------------------
def main():
    
    print(f"--- 0. НАСТРОЙКА ---")

    # Инициализация устройства (GPU/CPU)
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == 'cuda':
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}. Нагрузка оптимизирована.")
        # Включение cuDNN-бенчмарка для ускорения сверточных операций
        torch.backends.cudnn.benchmark = True 
    else:
        print("ВНИМАНИЕ: Используется CPU. Рекомендуется CUDA для ускорения.")
        
    print(f"Количество используемых сэмплов: {MAX_SAMPLES_TO_USE}")
    
    # --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
    print("\n--- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---")
    
    X, Y = load_and_preprocess_rml(RML_FILEPATH)
    
    if X is None:
        return
    
    # --- 2. ПРОВЕРКА РОБАСТНОСТИ МОДЕЛИ: 5-КРАТНАЯ КРОСС-ВАЛИДАЦИЯ ---
    N_SPLITS = 5
    # Стратифицированный KFold обеспечивает равномерное распределение классов по фолдам
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []
    
    print("\n--- 2. КРОСС-ВАЛИДАЦИЯ (5 ФОЛДОВ) ---")
    
    BATCH_SIZE = 512
    NUM_WORKERS = 6 # Количество рабочих потоков для загрузки данных

    for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
        print(f"\n[CV] Начинается ФОЛД {fold+1}/{N_SPLITS}...")
        
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]
        
        model_fold = ResNetAttentionCNN(num_classes=NUM_MODS).to(DEVICE)

        # Аугментация с реальными помехами применяется только к train set
        train_dataset_fold = RmlDataset(X_train_fold, Y_train_fold, augment=True) 
        val_dataset_fold = RmlDataset(X_val_fold, Y_val_fold, augment=False)
        
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) 
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # Обучение модели. В режиме CV лучшая модель не сохраняется.
        history_fold = train_model(model_fold, train_loader_fold, val_loader_fold, num_epochs=100, device=DEVICE, save_best=False) 
        
        best_val_acc = max(history_fold['val_accuracy'])
        fold_results.append(best_val_acc)
        
        print(f"[CV] ФОЛД {fold+1} завершен. Лучшая Val Accuracy: {best_val_acc:.4f}")

    avg_acc = np.mean(fold_results)
    print(f"\n[CV] СРЕДНЯЯ ВАЛИДАЦИОННАЯ ТОЧНОСТЬ ПО {N_SPLITS} ФОЛДАМ: {avg_acc:.4f}")
    
    # --- 3. ФИНАЛЬНОЕ ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛИ ---
    print("\n--- 3. ФИНАЛЬНОЕ ОБУЧЕНИЕ И ОЦЕНКА ---")
    
    # Используем индексы из последнего фолда для разделения на Train/Validation
    X_train, X_val, Y_train, Y_val = X[train_index], X[val_index], Y[train_index], Y[val_index]
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер валидационной выборки: {X_val.shape[0]}")
    
    # Инициализация финальной модели
    final_model = ResNetAttentionCNN(num_classes=NUM_MODS).to(DEVICE)

    # Создание PyTorch Dataset и DataLoader
    train_dataset = RmlDataset(X_train, Y_train, augment=True) 
    val_dataset = RmlDataset(X_val, Y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Обучение финальной модели и сохранение лучшей (save_best=True)
    history = train_model(final_model, train_loader, val_loader, num_epochs=100, device=DEVICE, save_best=True) 

    # --- 4. ОЦЕНКА И ВИЗУАЛИЗАЦИЯ ---
    
    # Загрузка лучших весов финальной модели для точной оценки
    if os.path.exists('best_rmc_model.pth'):
        final_model.load_state_dict(torch.load('best_rmc_model.pth'))
    else:
        print("[ERROR] Файл 'best_rmc_model.pth' не найден. Используется последняя модель.")
        
    # Оценка точности по SNR на всем датасете RML
    accuracy_vs_snr, _, _ = evaluate_by_snr(final_model, RML_FILEPATH, SNRS, device=DEVICE)

    # Матрица ошибок (на валидационной выборке)
    with torch.no_grad():
        outputs = final_model(torch.from_numpy(X_val).to(DEVICE))
        Y_pred_val = torch.max(outputs.data, 1)[1].cpu().numpy()
    
    cm = confusion_matrix(Y_val, Y_pred_val)
    
    # Визуализация результатов
    plot_results(accuracy_vs_snr, cm, history)
    
    print("\nПрограмма завершена.")

if __name__ == '__main__':
    main()