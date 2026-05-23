#!/usr/bin/env python3
"""
Тестовый клиент для имитации данных от TradingView.
Отправляет пачку свечей на ML сервер.
"""
import requests
import json
import random
from datetime import datetime, timedelta

SERVER = "http://localhost:8000"

def generate_candles(ticker="BTCUSDT", n=1000, timeframe="1"):
    base_price = 45000.0
    candles = []
    now = datetime.utcnow()

    for i in range(n):
        t = now - timedelta(minutes=n-i)
        noise = random.gauss(0, 0.001)
        trend = 0.0001 * (i / n)  # слабый восходящий тренд

        open_p = base_price * (1 + trend + noise)
        close_p = open_p * (1 + random.gauss(0, 0.002))
        high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.001)))
        low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.001)))
        vol = random.uniform(100, 1000)

        candles.append({
            "ticker": ticker,
            "timestamp": int(t.timestamp()),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": round(vol, 2),
            "timeframe": timeframe
        })
        base_price = close_p

    return candles

def send_candles():
    candles = generate_candles("BTCUSDT", 1000, "1")
    payload = {
        "ticker": "BTCUSDT",
        "timeframe": "1",
        "candles": candles
    }

    resp = requests.post(f"{SERVER}/webhook/candles", json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

if __name__ == "__main__":
    send_candles()
