# inference.py
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class PredictionResult:
    """Результат предсказания."""
    class_idx: int
    class_name: Optional[str]
    confidence: float
    probabilities: np.ndarray
    inference_time: float
    metadata: Dict[str, Any] = None

class ModulationClassifier:
    """Продакшен-классификатор модуляций."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Инициализация классификатора.
        
        Args:
            model_path: Путь к сохраненной модели
            config_path: Путь к конфигурационному файлу
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка конфигурации
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'num_classes': 58,
                'input_channels': 2,
                'num_iq_samples': 1024,
                'confidence_threshold': 0.7
            }
        
        # Загрузка модели
        self.load_model(model_path)
        
        # Загрузка меток классов если есть
        self.class_names = self.load_class_names()
        
        # Статистика
        self.inference_count = 0
        self.total_inference_time = 0.0
        
    def load_model(self, model_path: str):
        """Загрузка модели."""
        try:
            # Попробовать загрузить как TorchScript
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.use_torchscript = True
        except:
            # Загрузить как обычную PyTorch модель
            from model import EfficientModulationNet
            from config import ModelConfig
            
            config = ModelConfig(**self.config)
            self.model = EfficientModulationNet(config).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.use_torchscript = False
        
        self.model.eval()
        print(f"✅ Модель загружена на {self.device}")
        print(f"   TorchScript: {self.use_torchscript}")
        
    def load_class_names(self) -> List[str]:
        """Загрузка имен классов."""
        class_names_path = Path("class_names.json")
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                return json.load(f)
        
        # Стандартные имена для 58 классов torchsig
        default_names = [
            "AM-DSB", "AM-DSB-SC", "AM-SSB", "AM-SSB-SC",
            "ASK", "BPSK", "CPFSK", "DSB-SC", "FM",
            "GFSK", "GMSK", "MSK", "OOK", "PSK",
            "QAM16", "QAM64", "QPSK", "SSB", "SSB-SC",
            "WBFM", "8PSK", "AM", "AM-SSB-WC", "APSK",
            "APSK16", "APSK32", "APSK64", "ASK2", "ASK4",
            "ASK8", "BPSK", "CPM", "DPSK", "DQPSK",
            "DSB", "FSK", "GMSK", "MSK", "OOK",
            "PAM4", "PAM8", "PSK2", "PSK4", "PSK8",
            "QAM128", "QAM256", "QAM32", "QAM512", "QPSK",
            "SQAM", "SQPSK", "V29", "V32", "V34",
            "WBFM", "WBFM-STEREO", "WBFM-Stereo"
        ]
        
        # Дополняем если нужно
        while len(default_names) < self.config['num_classes']:
            default_names.append(f"Class_{len(default_names)}")
        
        return default_names[:self.config['num_classes']]
    
    def preprocess(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Предобработка IQ данных.
        
        Args:
            iq_data: Массив формы [2, N] или комплексный массив [N]
        
        Returns:
            torch.Tensor: Обработанный тензор
        """
        # Конвертация в torch tensor
        if np.iscomplexobj(iq_data):
            # Комплексный массив -> разделяем на I и Q
            i_data = np.real(iq_data).astype(np.float32)
            q_data = np.imag(iq_data).astype(np.float32)
            iq_tensor = np.stack([i_data, q_data], axis=0)
        else:
            # Уже разделенные I и Q
            iq_tensor = iq_data.astype(np.float32)
        
        # Проверка размерности
        if iq_tensor.shape[0] != 2:
            raise ValueError(f"Ожидается 2 канала (I и Q), получено {iq_tensor.shape[0]}")
        
        # Обрезка/дополнение до нужной длины
        target_length = self.config['num_iq_samples']
        current_length = iq_tensor.shape[1]
        
        if current_length < target_length:
            # Дополнение нулями
            pad_length = target_length - current_length
            iq_tensor = np.pad(iq_tensor, ((0, 0), (0, pad_length)), mode='constant')
        elif current_length > target_length:
            # Обрезка
            iq_tensor = iq_tensor[:, :target_length]
        
        # Нормализация
        iq_power = np.sqrt(np.mean(iq_tensor**2, axis=1, keepdims=True))
        iq_tensor = iq_tensor / (iq_power + 1e-8)
        
        # Конвертация в torch tensor
        tensor = torch.from_numpy(iq_tensor).unsqueeze(0)  # Добавляем batch dimension
        
        return tensor
    
    def predict(self, iq_data: np.ndarray, 
                return_confidence: bool = True,
                top_k: int = 3) -> PredictionResult:
        """
        Предсказание типа модуляции.
        
        Args:
            iq_data: Входные IQ данные
            return_confidence: Возвращать уверенность
            top_k: Количество лучших предсказаний
        
        Returns:
            PredictionResult: Результат предсказания
        """
        start_time = time.time()
        
        # Предобработка
        input_tensor = self.preprocess(iq_data).to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.use_torchscript:
                output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        # Post-processing
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        
        inference_time = time.time() - start_time
        
        # Обновление статистики
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Получение top-k предсказаний
        if top_k > 1:
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
            top_predictions = []
            for i in range(top_k):
                class_idx = top_indices[0, i].item()
                top_predictions.append({
                    'class': class_idx,
                    'name': self.class_names[class_idx] if self.class_names else str(class_idx),
                    'probability': top_probs[0, i].item()
                })
        else:
            top_predictions = None
        
        # Создание результата
        result = PredictionResult(
            class_idx=prediction.item(),
            class_name=self.class_names[prediction.item()] if self.class_names else None,
            confidence=confidence.item(),
            probabilities=probabilities.cpu().numpy()[0],
            inference_time=inference_time,
            metadata={
                'top_predictions': top_predictions,
                'input_shape': iq_data.shape
            }
        )
        
        return result
    
    def predict_batch(self, iq_batch: List[np.ndarray]) -> List[PredictionResult]:
        """Пакетное предсказание."""
        results = []
        for iq_data in iq_batch:
            results.append(self.predict(iq_data))
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики inference."""
        avg_time = self.total_inference_time / max(self.inference_count, 1)
        
        return {
            'total_inferences': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_time,
            'inferences_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'device': str(self.device)
        }
    
    def save_statistics(self, path: str = "inference_stats.json"):
        """Сохранение статистики."""
        stats = self.get_statistics()
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

# Пример использования
def example_usage():
    """Пример использования классификатора."""
    
    # Инициализация классификатора
    classifier = ModulationClassifier(
        model_path="production_model_scripted.pt",
        config_path="configs.yaml"
    )
    
    # Генерация тестовых данных
    np.random.seed(42)
    
    # Пример 1: Комплексный сигнал
    t = np.linspace(0, 1, 1024)
    carrier = np.exp(1j * 2 * np.pi * 1e6 * t)
    modulated_signal = carrier * (1 + 0.5 * np.sin(2 * np.pi * 1e4 * t))
    noise = 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    test_signal = modulated_signal + noise
    
    # Предсказание
    result = classifier.predict(test_signal, top_k=3)
    
    print(f"\nРезультат предсказания:")
    print(f"  Класс: {result.class_idx} ({result.class_name})")
    print(f"  Уверенность: {result.confidence:.4f}")
    print(f"  Время inference: {result.inference_time*1000:.2f} мс")
    
    if result.metadata['top_predictions']:
        print(f"  Top-3 предсказания:")
        for pred in result.metadata['top_predictions']:
            print(f"    {pred['name']}: {pred['probability']:.4f}")
    
    # Пример 2: Разделенный I и Q
    i_component = np.real(test_signal)
    q_component = np.imag(test_signal)
    test_signal_iq = np.stack([i_component, q_component], axis=0)
    
    result2 = classifier.predict(test_signal_iq)
    print(f"\nВторой результат: {result2.class_name} ({result2.confidence:.4f})")
    
    # Статистика
    stats = classifier.get_statistics()
    print(f"\nСтатистика:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    example_usage()