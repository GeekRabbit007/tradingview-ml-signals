#!/usr/bin/env python3
"""
Скрипт для GitHub Actions: забирает последние ML сигналы с сервера
и обновляет CSV файлы в формате Pine Seeds.
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime

SERVER_URL = os.getenv("ML_SERVER_URL", "http://your-server:8000")
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1", "5", "15", "60", "240", "D"]

def fetch_signal(ticker, timeframe):
    try:
        resp = requests.get(f"{SERVER_URL}/signal/{ticker}/{timeframe}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching {ticker}/{timeframe}: {e}")
    return None

def update_csv(ticker, timeframe, signal_data):
    filename = f"data/{ticker}_{timeframe}.csv"

    # Pine Seeds формат: time,open,high,low,close,volume
    # Mapping:
    #   time = timestamp
    #   open = predicted_price
    #   high = strength (0 to 1)
    #   low = anomaly_score
    #   close = signal (-1, 0, 1)
    #   volume = confidence (0 to 1)

    timestamp = signal_data.get("timestamp", int(datetime.utcnow().timestamp()))
    signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    signal_val = signal_map.get(signal_data.get("signal", "HOLD"), 0)
    confidence = signal_data.get("confidence", 0.5)
    strength = signal_data.get("strength", 0.0)
    anomaly = signal_data.get("anomaly_score", 0.0)
    pred_price = signal_data.get("predicted_price", 0.0)

    new_row = {
        "time": timestamp,
        "open": pred_price,
        "high": abs(strength),
        "low": anomaly,
        "close": signal_val,
        "volume": confidence
    }

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        # Храним только последние 5000 записей
        df = df.tail(5000)
    else:
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame([new_row])

    df.to_csv(filename, index=False)
    print(f"Updated {filename}: {signal_data.get('signal')} conf={confidence}")

def main():
    for ticker in TICKERS:
        for tf in TIMEFRAMES:
            signal = fetch_signal(ticker, tf)
            if signal:
                update_csv(ticker, tf, signal)
            else:
                # Если сервер недоступен, создаем HOLD сигнал
                update_csv(ticker, tf, {
                    "timestamp": int(datetime.utcnow().timestamp()),
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "strength": 0.0,
                    "anomaly_score": 0.0,
                    "predicted_price": 0.0
                })

if __name__ == "__main__":
    main()
