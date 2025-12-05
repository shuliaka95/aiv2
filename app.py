# app.py

from flask import Flask, request, jsonify
import torch
import os

from model import MyCustomModel 
from config import CLASSES, NUM_IQ_SAMPLES, MODEL_CONFIG, TRAIN_CONFIG

app = Flask(__name__)

# --- Константы Инференса ---
MODEL_PATH = MODEL_CONFIG['model_path']
DEVICE = TRAIN_CONFIG['device']
model = None

def load_inference_model():
    """
    Загружает веса модели для инференса. 
    """
    global model
    if model is None:
        try:
            # Инициализируем архитектуру на 45 классов
            model_instance = MyCustomModel(num_classes=len(CLASSES))
            
            if os.path.exists(MODEL_PATH):
                model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model_instance.to(DEVICE)
                model_instance.eval()
                model = model_instance
                print(f"--- Модель ({len(CLASSES)} классов) загружена для инференса ---")
            else:
                print("--- [ERROR] Модель не найдена. Требуется обучение. ---")
        except Exception as e:
            print(f"--- [ERROR] Ошибка при загрузке модели: {e} ---")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для классификации сигнала, принимающего сырые IQ данные."""
    if model is None:
        return jsonify({"error": "Model not ready. Training artifact missing or failed to load."}), 503

    data = request.json.get('iq_data')
    
    # Проверка, что входной массив соответствует 2 * N_SAMPLES
    if not data or len(data) != 2 * NUM_IQ_SAMPLES:
        return jsonify({"error": f"Expected {2*NUM_IQ_SAMPLES} float values for I and Q"}), 400

    try:
        # 1. Конвертация данных
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        # Решейп в [1, 2, 1024]: [Batch, Channel, Length]
        input_tensor = tensor_data.view(2, NUM_IQ_SAMPLES).unsqueeze(0).to(DEVICE) 

        # 2. Предсказание
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        # 3. Формирование ответа
        result = {
            "class": CLASSES[pred_idx.item()],
            "confidence": float(confidence.item()),
            # Возвращаем все вероятности
            "all_probs": {k: f"{v:.4f}" for k, v in zip(CLASSES, probs[0].tolist())}
        }
        return jsonify(result)

    except Exception as e:
        print(f"[Inference Error]: {e}")
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500

# Загружаем модель при старте Gunicorn
load_inference_model()