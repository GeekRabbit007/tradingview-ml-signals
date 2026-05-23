"""
Smart Self-Learning ML Server for TradingView
Архитектура: FastAPI + Isolation Forest (аномалии) + XGBoost (направление)
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from github import Github
import joblib
import uvicorn

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TradingView ML Bridge", version="2.0")

# ─── Модели данных ────────────────────────────────────────────────

class CandleData(BaseModel):
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # "1", "5", "15", "60", "240", "D"

class SignalRequest(BaseModel):
    ticker: str
    timeframe: str
    candles: List[CandleData]
    features: Optional[Dict[str, float]] = None

class MLSignal(BaseModel):
    ticker: str
    timestamp: int
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 - 1.0
    strength: float    # -1.0 (strong sell) to 1.0 (strong buy)
    anomaly_score: float
    predicted_price: float
    trend_direction: str  # "UP", "DOWN", "SIDE"
    volatility_regime: str  # "LOW", "NORMAL", "HIGH"
    ml_model_version: str

# ─── Глобальное хранилище ─────────────────────────────────────────

class ModelStore:
    """Хранилище ML-моделей с горячей заменой"""
    def __init__(self):
        self.models: Dict[str, dict] = {}  # ticker -> {model, scaler, last_train}
        self.data_buffer: Dict[str, pd.DataFrame] = {}  # буфер свечей
        self.min_train_samples = 500  # минимум свечей для обучения
        self.retrain_interval = 100   # переобучение каждые N новых свечей

    def get_or_create(self, ticker: str, timeframe: str):
        key = f"{ticker}_{timeframe}"
        if key not in self.models:
            self.models[key] = {
                'direction_model': None,
                'anomaly_model': None,
                'scaler': StandardScaler(),
                'last_train': None,
                'samples_since_train': 0,
                'version': 1
            }
        return self.models[key]

model_store = ModelStore()

# ─── Фиче-инжиниринг ─────────────────────────────────────────────

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет технических индикаторов как фичей для ML"""
    df = df.copy()

    # Базовые ценовые фичи
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Скользящие средние
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'dist_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

    # Волатильность
    df['atr_14'] = ta_atr(df['high'], df['low'], df['close'], 14)
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['bb_width'] = ta_bb_width(df['close'], 20)

    # Импульс
    df['rsi_14'] = ta_rsi(df['close'], 14)
    df['rsi_7'] = ta_rsi(df['close'], 7)
    df['macd'] = ta_macd(df['close'])

    # Объем
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Свечные паттерны
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

    # Лаговые фичи
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume_ratio'].shift(lag)

    # Целевая переменная: направление следующей свечи
    df['target_direction'] = np.where(df['close'].shift(-1) > df['close'], 1, 
                                     np.where(df['close'].shift(-1) < df['close'], -1, 0))

    # Целевая переменная: аномальность (большое движение)
    df['target_anomaly'] = np.where(abs(df['returns']) > df['returns'].rolling(50).std() * 2, -1, 1)

    return df

def ta_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ta_rsi(close, period=14):
    """RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ta_macd(close, fast=12, slow=26, signal=9):
    """MACD line"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    return ema_fast - ema_slow

def ta_bb_width(close, period=20, std_dev=2):
    """Bollinger Bands width"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return (upper - lower) / sma

# ─── ML Обучение ──────────────────────────────────────────────────

def train_models(df: pd.DataFrame, store: dict):
    """Обучение моделей накопленных данных"""
    feature_cols = [c for c in df.columns if c.startswith(('sma_', 'ema_', 'dist_', 'atr_', 'rsi_', 
                                                          'macd', 'volume_', 'body_', 'upper_', 'lower_',
                                                          'return_lag_', 'volatility_', 'bb_'))]

    train_df = df.dropna()
    if len(train_df) < model_store.min_train_samples:
        return False

    X = train_df[feature_cols].values
    y_direction = train_df['target_direction'].values

    # Масштабирование
    X_scaled = store['scaler'].fit_transform(X)

    # 1. Модель направления (XGBoost)
    dtrain = xgb.DMatrix(X_scaled, label=(y_direction + 1))  # сдвиг для 0,1,2
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss'
    }
    store['direction_model'] = xgb.train(params, dtrain, num_boost_round=100)

    # 2. Модель аномалий (Isolation Forest)
    store['anomaly_model'] = IsolationForest(
        n_estimators=100, 
        contamination=0.05,
        random_state=42
    )
    store['anomaly_model'].fit(X_scaled)

    store['last_train'] = datetime.utcnow()
    store['samples_since_train'] = 0
    store['version'] += 1
    store['feature_cols'] = feature_cols

    logger.info(f"Models trained. Version: {store['version']}, Samples: {len(train_df)}")
    return True

def predict(df: pd.DataFrame, store: dict) -> dict:
    """Генерация предсказаний"""
    if store['direction_model'] is None:
        return None

    feature_cols = store.get('feature_cols', [])
    if not feature_cols:
        return None

    latest = df.iloc[-1:][feature_cols].values
    X_scaled = store['scaler'].transform(latest)

    # Предсказание направления
    pred_probs = store['direction_model'].predict(xgb.DMatrix(X_scaled))
    pred_class = np.argmax(pred_probs[0])
    confidence = float(np.max(pred_probs[0]))

    direction_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    strength_map = {0: -1.0, 1: 0.0, 2: 1.0}

    # Аномалия
    anomaly_score = float(store['anomaly_model'].decision_function(X_scaled)[0])
    is_anomaly = store['anomaly_model'].predict(X_scaled)[0] == -1

    # Волатильность
    vol = df['volatility_20'].iloc[-1]
    vol_mean = df['volatility_20'].mean()
    if vol > vol_mean * 1.5:
        vol_regime = "HIGH"
    elif vol < vol_mean * 0.5:
        vol_regime = "LOW"
    else:
        vol_regime = "NORMAL"

    # Прогноз цены (простая экстраполяция)
    current_price = df['close'].iloc[-1]
    predicted_return = (pred_class - 1) * confidence * vol  # смещенный прогноз
    predicted_price = current_price * (1 + predicted_return)

    # Определение тренда
    sma20 = df['sma_20'].iloc[-1]
    sma50 = df['sma_50'].iloc[-1]
    if sma20 > sma50 * 1.01:
        trend = "UP"
    elif sma20 < sma50 * 0.99:
        trend = "DOWN"
    else:
        trend = "SIDE"

    return {
        'signal': direction_map[pred_class],
        'confidence': round(confidence, 4),
        'strength': round(strength_map[pred_class] * confidence, 4),
        'anomaly_score': round(anomaly_score, 4),
        'predicted_price': round(predicted_price, 4),
        'trend_direction': trend,
        'volatility_regime': vol_regime,
        'is_anomaly': bool(is_anomaly),
        'model_version': store['version']
    }

# ─── GitHub Pine Seeds интеграция ───────────────────────────────────

def update_pine_seeds(signal: MLSignal):
    """Обновление CSV в GitHub для request.seed()"""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        repo_name = os.getenv("PINE_SEEDS_REPO", "username/pine-seeds-signals")

        if not github_token:
            logger.warning("GITHUB_TOKEN not set, skipping Pine Seeds update")
            return

        g = Github(github_token)
        repo = g.get_repo(repo_name)

        # Формат CSV для Pine Seeds: time,open,high,low,close,volume
        # Используем close как сигнал, volume как confidence
        timestamp = signal.timestamp
        signal_val = 1 if signal.signal == "BUY" else (-1 if signal.signal == "SELL" else 0)

        csv_line = f"{timestamp},{signal_val},{signal.confidence},{signal.strength},{signal.anomaly_score},{signal.predicted_price}\n"

        file_path = f"data/{signal.ticker}_{signal.timeframe}.csv"

        try:
            contents = repo.get_contents(file_path)
            old_content = contents.decoded_content.decode('utf-8')
            new_content = old_content + csv_line
            repo.update_file(file_path, f"Update {signal.ticker} signal", new_content, contents.sha)
        except:
            header = "time,open,high,low,close,volume\n"
            repo.create_file(file_path, f"Init {signal.ticker}", header + csv_line)

        logger.info(f"Pine Seeds updated for {signal.ticker}")
    except Exception as e:
        logger.error(f"Pine Seeds update failed: {e}")

# ─── API Endpoints ────────────────────────────────────────────────

@app.post("/webhook/candles", response_model=MLSignal)
async def receive_candles(data: SignalRequest, background_tasks: BackgroundTasks):
    """Получение свечных данных от TradingView и генерация сигнала"""
    try:
        # Конвертация в DataFrame
        candles = data.candles
        if len(candles) < 50:
            raise HTTPException(400, "Need at least 50 candles")

        df = pd.DataFrame([c.dict() for c in candles])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime')

        # Расчет фичей
        df = calculate_features(df)

        # Получение/создание модели
        store = model_store.get_or_create(data.ticker, data.timeframe)

        # Добавление в буфер
        key = f"{data.ticker}_{data.timeframe}"
        if key in model_store.data_buffer:
            model_store.data_buffer[key] = pd.concat([model_store.data_buffer[key], df]).tail(2000)
        else:
            model_store.data_buffer[key] = df

        buf_df = model_store.data_buffer[key]
        store['samples_since_train'] += len(df)

        # Обучение при необходимости
        if store['direction_model'] is None or store['samples_since_train'] >= model_store.retrain_interval:
            train_models(buf_df, store)

        # Предсказание
        pred = predict(buf_df, store)
        if pred is None:
            raise HTTPException(503, "Model not ready yet")

        signal = MLSignal(
            ticker=data.ticker,
            timestamp=int(datetime.utcnow().timestamp()),
            signal=pred['signal'],
            confidence=pred['confidence'],
            strength=pred['strength'],
            anomaly_score=pred['anomaly_score'],
            predicted_price=pred['predicted_price'],
            trend_direction=pred['trend_direction'],
            volatility_regime=pred['volatility_regime'],
            ml_model_version=str(pred['model_version'])
        )

        # Асинхронное обновление Pine Seeds
        background_tasks.add_task(update_pine_seeds, signal)

        logger.info(f"Signal generated: {signal.ticker} {signal.signal} conf={signal.confidence}")
        return signal

    except Exception as e:
        logger.error(f"Error processing candles: {e}")
        raise HTTPException(500, str(e))

@app.post("/webhook/alert")
async def receive_alert(alert_data: dict):
    """Получение простого алерта от TradingView (для стратегий)"""
    logger.info(f"Alert received: {alert_data}")
    return {"status": "received"}

@app.get("/signal/{ticker}/{timeframe}")
async def get_latest_signal(ticker: str, timeframe: str):
    """Получение последнего сигнала (для polling)"""
    # Здесь можно добавить Redis/DB для хранения последних сигналов
    return {"ticker": ticker, "timeframe": timeframe, "signal": "HOLD", "timestamp": int(datetime.utcnow().timestamp())}

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(model_store.models)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
