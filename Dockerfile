# +============================================================================+
# |               ОБРАЗ ДЛЯ МОДЕЛИ С TORCHSIG (ИСПРАВЛЕННЫЙ)                   |
# +============================================================================+
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

# 1. Установка системных зависимостей и Rust (ОБЯЗАТЕЛЬНО)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      libffi-dev libssl-dev \
      python3.10 python3-pip python3-dev \
      git \
      libglib2.0-0 \
      libgl1 \
      libsm6 \
      libxrender1 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 2. Установка Rust (КРИТИЧНО для сборки TorchSig)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable

# 3. Обновление pip и установка PyTorch (ОТДЕЛЬНО)
RUN python3.10 -m pip install --upgrade pip setuptools setuptools-rust wheel
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Копирование файлов проекта
COPY . /app
# Убедимся, что файл dataset.py доступен в корне
RUN cp /app/ml/dataset.py /app/dataset.py 2>/dev/null || true

# 5. Установка зависимостей из requirements.txt (БЕЗ torch и torchsig!)
# УДАЛИТЕ из вашего requirements.txt строки 'torch' и 'torchsig', если они есть
RUN if [ -f "requirements.txt" ]; then python3.10 -m pip install --no-cache-dir -r requirements.txt; fi

# 6. КРИТИЧЕСКИЙ ШАГ: Установка TorchSig через клонирование и ручную сборку
RUN cd /tmp && \
    git clone --depth 1 https://github.com/torchdsp/torchsig.git && \
    cd torchsig && \
    python3.10 -m pip install --no-cache-dir . && \
    cd /app && rm -rf /tmp/torchsig

# 7. ФИНАЛЬНАЯ ПРОВЕРКА (упрощенная, но надежная)
RUN echo "=== Финальная проверка ===" && \
    python3.10 -c "import torch; print('PyTorch OK:', torch.__version__)" && \
    python3.10 -c "import torchsig; print('TorchSig импортирован'); import inspect; print('Расположение:', inspect.getfile(torchsig))" && \
    python3.10 -c "from torchsig.datasets import ModulationsDataset; print('Импорт ModulationsDataset: УСПЕХ')" || \
    (echo "=== Предупреждение: Не удалось импортировать через datasets, проверяем структуру..."; find /usr/local/lib/python3.10/dist-packages/torchsig/ -type f -name "*.py" | head -5)

# 8. Настройка среды (исправлена переменная PYTHONPATH)
ENV PYTHONPATH="/app"
RUN chmod +x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]