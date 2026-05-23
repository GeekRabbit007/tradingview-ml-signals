# 🤖 ML Smart TradingView Indicator

Комплексная самообучающаяся ML-система для TradingView на Python + Pine Script v6.

## Архитектура

```
TradingView (график)
    ↓ webhook (OHLCV)
Python ML Server (FastAPI + XGBoost + Isolation Forest)
    ↓ обновление CSV
GitHub Repo (Pine Seeds)
    ↓ request.seed()
Pine Script (индикатор на графике)
```

## Компоненты

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| ML Server | FastAPI + XGBoost + sklearn | Обучение, предсказание, генерация сигналов |
| Pine Seeds | GitHub + CSV | Канал данных в TradingView |
| Pine Script | Pine Script v6 | Визуализация сигналов на графике |
| GitHub Actions | CI/CD | Автообновление данных каждые 15 мин |

## ⚡ Быстрый старт

### 1. ML Server

```bash
# Клонирование
cd ml_tradingview_system

# Настройка окружения
cp .env.example .env
# Отредактируй .env: добавь GITHUB_TOKEN и имя репозитория

# Запуск через Docker
docker-compose up -d

# Или локально
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. GitHub репозиторий (Pine Seeds)

1. Создайте публичный репозиторий на GitHub (например: `username/pine-seeds-signals`)
2. Добавьте структуру:
   ```
   data/
   scripts/
   .github/workflows/
   ```
3. В настройках репозитория: Settings → Secrets → Actions
   - Добавьте `ML_SERVER_URL` (URL вашего сервера)
   - Добавьте `GITHUB_TOKEN` (Personal Access Token с правами `repo`)
4. Скопируйте файлы из этого проекта в репозиторий

### 3. TradingView Pine Script

1. Откройте Pine Editor в TradingView
2. Скопируйте содержимое `ML_Smart_Indicator.pine`
3. Нажмите "Add to chart"
4. В настройках укажите ваш репозиторий Pine Seeds

**Важно:** Pine Seeds данные обновляются TradingView 1 раз в сутки. Для real-time используйте webhook + alerts.

### 4. Webhook из TradingView

Настройте alert в TradingView:
- **Webhook URL**: `http://your-server:8000/webhook/alert`
- **Message**:
```json
{
  "ticker": "{{ticker}}",
  "price": {{close}},
  "time": {{time}},
  "action": "{{strategy.order.action}}"
}
```

## 🔧 ML Модели

### Direction Model (XGBoost)
- **Задача**: Предсказание направления следующей свечи (BUY/SELL/HOLD)
- **Фичи**: SMA/EMA, RSI, MACD, ATR, Bollinger Bands, Volume, лаги
- **Переобучение**: Каждые 100 новых свечей

### Anomaly Detection (Isolation Forest)
- **Задача**: Обнаружение аномальных движений цены
- **Contamination**: 5%
- **Применение**: Маркировка нестандартных свечей

## 📊 Индикатор

### Визуальные элементы:
- **🟢 Стрелка вверх** — ML Buy Signal (confidence ≥ threshold)
- **🔴 Стрелка вниз** — ML Sell Signal (confidence ≥ threshold)
- **🟠 Ромб ⚡** — Аномалия (резкое движение)
- **🟣 Линия** — Предсказанная цена
- **Фон** — Зона тренда (зеленый/красный/серый)
- **Таблица** — Confidence, Strength, Trend, Anomaly

### Алерты:
- `ML Buy Signal`
- `ML Sell Signal`
- `ML Anomaly Detected`

## 🌐 API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/webhook/candles` | POST | Основной endpoint для свечных данных |
| `/webhook/alert` | POST | Простой alert webhook |
| `/signal/{ticker}/{tf}` | GET | Последний сигнал (для polling) |
| `/health` | GET | Статус сервера |

### Пример запроса:
```bash
curl -X POST http://localhost:8000/webhook/candles \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "BTCUSDT",
    "timeframe": "1",
    "candles": [
      {"ticker":"BTCUSDT","timestamp":1700000000,"open":45000,"high":45100,"low":44900,"close":45050,"volume":1000,"timeframe":"1"}
    ]
  }'
```

## 🔒 Ограничения и рекомендации

### Pine Seeds ограничения:
- Обновление 1 раз в сутки (TradingView ограничение)
- Максимум 10 CSV файлов на репозиторий
- Формат строго: `time,open,high,low,close,volume`

### Решение для real-time:
Для реального времени используйте **двухканальную схему**:
1. **Webhook** (real-time) → Python Server → Broker API (исполнение)
2. **Pine Seeds** (1-2 раза в день) → Индикатор (визуализация истории)

### Безопасность:
- Используйте HTTPS + API Key для production
- Не храните токены в коде
- Ограничьте доступ к серверу по IP

## 📁 Структура проекта

```
ml_tradingview_system/
├── app/
│   └── main.py              # FastAPI сервер + ML
├── scripts/
│   ├── fetch_signals.py     # GitHub Actions скрипт
│   └── test_webhook.py      # Тестовый клиент
├── .github/workflows/
│   └── update-pine-seeds.yml # CI/CD
├── data/                    # Локальное хранилище
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── ML_Smart_Indicator.pine  # Pine Script v6
```

## 🚀 Roadmap

- [ ] Добавить LSTM/Transformer модели
- [ ] Интеграция с Binance/Bybit API для исполнения
- [ ] Multi-timeframe ансамбль моделей
- [ ] Автоматический подбор гиперпараметров (Optuna)
- [ ] Telegram-бот для уведомлений

## 📄 Лицензия

MIT License. Используйте на свой страх и риск.
**Не является финансовой рекомендацией.**
