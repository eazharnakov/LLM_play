# LLM Play — Обучение языковых моделей

Проект для fine-tuning LLM моделей с использованием LoRA/QLoRA.

## Структура проекта

```
LLM_play/
├── configs/              # Конфигурации обучения
│   └── train_config.yaml
├── data/                 # Данные (raw/processed)
├── notebooks/            # Jupyter ноутбуки для экспериментов
├── scripts/              # Скрипты запуска
│   ├── train.py          # Запуск обучения
│   └── inference.py      # Генерация текста
├── src/                  # Исходный код
│   ├── models/           # Загрузка моделей и LoRA
│   ├── training/         # Обучение и обработка данных
│   └── utils/            # Утилиты (конфиги, инференс)
├── requirements.txt
└── README.md
```

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Настройка конфигурации

Отредактируйте `configs/train_config.yaml` — укажите модель, датасет и параметры обучения.

### 3. Запуск обучения

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 4. Инференс

```bash
# Одиночный запрос
python scripts/inference.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter-path ./outputs \
  --prompt "Объясни что такое нейронная сеть"

# Интерактивный режим
python scripts/inference.py \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter-path ./outputs \
  --interactive
```

## Возможности

- **QLoRA fine-tuning** — обучение больших моделей на потребительских GPU (4-bit квантизация)
- **Гибкая конфигурация** — все параметры в YAML файле
- **Поддержка HuggingFace** — любая модель и датасет из Hub
- **Пользовательские данные** — загрузка из JSONL файлов
- **Мониторинг** — интеграция с Weights & Biases / TensorBoard

## Требования к железу

| Метод         | VRAM   | Пример GPU          |
|---------------|--------|----------------------|
| QLoRA (4-bit) | 6-12GB | RTX 3060/4060        |
| LoRA (16-bit) | 16-24GB| RTX 3090/4090        |
| Full FT       | 80GB+  | A100/H100            |
