// static/js/script.js - ОБНОВЛЕННЫЙ
let currentFile = null;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initializeDragAndDrop();
    setupFileInput();
    setupDataInput();
});

// Переключение вкладок
function switchTab(tabId) {
    // Скрыть все вкладки
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Убрать активный класс со всех кнопок
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Показать выбранную вкладку
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
    
    // Сбросить результаты
    hideResults();
    hideStatus();
}

// Настройка drag and drop для файлов
function initializeDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    // Клик по области загрузки
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

// Настройка input файла
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

// Настройка поля ввода данных
function setupDataInput() {
    const dataInput = document.getElementById('rawDataInput');
    
    dataInput.addEventListener('input', function() {
        const lines = this.value.trim().split('\n').filter(line => line.trim() !== '');
        document.getElementById('dataCount').textContent = `Введено строк: ${lines.length}`;
        
        // Проверяем валидность данных
        let validLines = 0;
        for (let line of lines) {
            const parts = line.replace(',', ' ').split(/\s+/).filter(part => part !== '');
            if (parts.length >= 2 && !isNaN(parseFloat(parts[0])) && !isNaN(parseFloat(parts[1]))) {
                validLines++;
            }
        }
        
        const analyzeBtn = document.getElementById('analyzeDataBtn');
        analyzeBtn.disabled = validLines < 10;
        
        const status = document.getElementById('dataStatus');
        if (validLines >= 10) {
            status.textContent = `✅ Готово к анализу (${validLines} валидных пар)`;
            status.style.color = '#27ae60';
        } else if (lines.length > 0) {
            status.textContent = `❌ Нужно минимум 10 пар I,Q данных (сейчас ${validLines})`;
            status.style.color = '#e74c3c';
        } else {
            status.textContent = '';
        }
    });
}

// Обработка выбранных файлов
function handleFiles(files) {
    if (files.length === 0) return;
    
    currentFile = files[0];
    updateFileInfo(currentFile);
    updateAnalyzeButton(true);
}

// Обновление информации о файле
function updateFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    
    fileInfo.innerHTML = `
        <strong>📄 Выбран файл:</strong> ${file.name}<br>
        <small>💾 Размер: ${sizeMB} MB</small><br>
        <small>📝 Тип: ${file.type || 'Неизвестно'}</small>
    `;
    fileInfo.style.display = 'block';
}

// Обновление кнопки анализа
function updateAnalyzeButton(enabled) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = !enabled;
}

// Анализ файла
async function analyzeFile() {
    if (!currentFile) return;
    
    showStatus('processing', '🔍 Анализируем файл...');
    hideResults();
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', '✅ Анализ завершен!');
            showResults(result);
        } else {
            showStatus('error', '❌ Ошибка: ' + result.error);
        }
        
    } catch (error) {
        showStatus('error', '❌ Ошибка сети: ' + error.message);
    }
}

// Анализ сырых данных
async function analyzeRawData() {
    const rawData = document.getElementById('rawDataInput').value.trim();
    
    if (!rawData) {
        showStatus('error', '❌ Введите данные для анализа');
        return;
    }
    
    showStatus('processing', '🔍 Анализируем данные...');
    hideResults();
    
    try {
        const response = await fetch('/api/analyze_raw', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                raw_data: rawData
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', '✅ Анализ завершен!');
            showResults(result);
        } else {
            showStatus('error', '❌ Ошибка: ' + result.error);
        }
        
    } catch (error) {
        showStatus('error', '❌ Ошибка сети: ' + error.message);
    }
}

// Загрузка тестовых данных
async function loadTestData() {
    try {
        const response = await fetch('/api/generate_test_data');
        const data = await response.json();
        
        document.getElementById('rawDataInput').value = data.test_data;
        document.getElementById('rawDataInput').dispatchEvent(new Event('input'));
        
        showStatus('success', `✅ Загружены тестовые данные (${data.samples} samples, ${data.modulation})`);
    } catch (error) {
        showStatus('error', '❌ Ошибка загрузки тестовых данных');
    }
}

// Очистка данных
function clearData() {
    document.getElementById('rawDataInput').value = '';
    document.getElementById('rawDataInput').dispatchEvent(new Event('input'));
    hideResults();
    hideStatus();
}

// Показать результаты
function showResults(data) {
    const resultsSection = document.getElementById('results');
    const modulationResult = document.getElementById('modulationResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const predictionsList = document.getElementById('predictionsList');
    const fileMeta = document.getElementById('fileMeta');
    const timeMeta = document.getElementById('timeMeta');
    const modelMeta = document.getElementById('modelMeta');
    
    const result = data.result;
    
    // Основной результат
    modulationResult.textContent = `🎯 ${result.modulation}`;
    confidenceResult.textContent = `✅ Уверенность: ${(result.confidence * 100).toFixed(2)}%`;
    
    // Топ предсказания
    predictionsList.innerHTML = result.top_predictions.map(pred => `
        <div class="prediction-item">
            <span class="prediction-name">${pred.modulation}</span>
            <span class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
        </div>
    `).join('');
    
    // Мета-информация
    fileMeta.innerHTML = data.filename ? 
        `<strong>Файл:</strong> ${data.filename}` : 
        `<strong>Данные:</strong> ${data.data_points || data.signal_length} samples`;
    
    timeMeta.innerHTML = `<strong>Время:</strong> ${data.timestamp}`;
    modelMeta.innerHTML = `<strong>Модель:</strong> ${data.model_loaded ? '✅ Обученная' : '⚠️ Демо'}`;
    
    // Показать секцию результатов
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Скрыть результаты
function hideResults() {
    document.getElementById('results').style.display = 'none';
}

// Показать статус
function showStatus(type, message) {
    const status = document.getElementById('status');
    status.className = `status ${type}`;
    status.style.display = 'block';
    status.innerHTML = type === 'processing' 
        ? `<div class="loading"></div>${message}`
        : message;
}

// Скрыть статус
function hideStatus() {
    document.getElementById('status').style.display = 'none';
}