# Modulation Signal Classifier

Нейросетевая модель для классификации 58 типов модуляций сигналов.

## Особенности

- Классификация 58 типов модуляций
- Архитектура ResNet на 15 млн параметров
- Обучение с защитой от переобучения
- Поддержка GPU/CPU
- Готовность к продакшену

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/ваш-username/modulation_classifier.git
cd modulation_classifier

# Установить зависимости
pip install -r requirements.txt
#скачать torchsig
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install -e .

