# Проект 22 — Support Ticket Classifier API

FastAPI-сервис для классификации обращений в техподдержку с помощью дообученной модели Qwen2.5-1.5B (Проект 21).

## Структура

```
project_22/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI приложение, endpoints
│   ├── models.py      # Pydantic-схемы запросов и ответов
│   └── inference.py   # Загрузка модели и генерация
├── fine_tuned_model/  # LoRA-адаптеры из Проекта 21
├── .env               # Переменные окружения (не в git)
├── .gitignore
├── requirements.txt
├── test_api.py        # Тесты с httpx
└── README.md
```

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Добавить адаптеры модели

Скопируй файлы из Проекта 21 в папку `fine_tuned_model/`:
- `adapter_model.safetensors`
- `adapter_config.json`

### 3. Настройка .env

```env
BASE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
ADAPTER_PATH=fine_tuned_model
```

### 4. Запуск сервера

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Документация

Открой в браузере: http://localhost:8000/docs

## Endpoints

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Статус сервиса и модели |
| POST | `/generate` | Классификация тикета |

## Примеры запросов

### Health check
```bash
curl http://localhost:8000/health
```

### Классификация тикета
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I was charged twice this month, please refund."}'
```

**Ответ:**
```json
{
  "generated_text": "{\"category\": \"billing\", \"priority\": \"high\"}",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_length": 50,
  "generated_length": 42
}
```

## Тестирование

```bash
# Запусти сервер, потом в другом терминале:
python test_api.py
```

## Обработка ошибок

| Ситуация | HTTP статус |
|---|---|
| Пустой промпт | 422 |
| Промпт > 2048 символов | 422 |
| Модель не загружена | 503 |
| Внутренняя ошибка модели | 500 |
