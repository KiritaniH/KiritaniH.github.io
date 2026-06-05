from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(".0", "", regex=False).str.strip()
    return pd.to_datetime(text, errors="coerce", format="%Y%m%d")


def detect_trade_side(text: str) -> str:
    value = str(text)
    if any(token in value for token in ["买入", "买", "开仓"]):
        return "buy"
    if any(token in value for token in ["卖出", "卖", "平仓"]):
        return "sell"
    return "other"


def compute_trade_regularity(trade_dates: pd.Series) -> float:
    ordered = trade_dates.sort_values()
    intervals = ordered.diff().dt.days.dropna()
    intervals = intervals[intervals > 0]
    if len(intervals) <= 1:
        return 0.0
    mean_interval = float(intervals.mean())
    if mean_interval <= 0:
        return 0.0
    return float(intervals.std(ddof=0) / mean_interval)


def log1p_clip(x: float) -> float:
    return float(np.log1p(max(x, 0.0)))


def max_drawdown_from_curve(curve: Iterable[float]) -> float:
    arr = np.array(list(curve), dtype=float)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    peaks = np.where(peaks == 0, 1e-9, peaks)
    dd = (arr - peaks) / peaks
    return float(dd.min())


def sharpe_like(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    std = returns.std(ddof=0)
    if std <= 1e-9:
        return 0.0
    return float(returns.mean() / std)


def fifo_realized_returns(
    df: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    side_col: str,
    price_col: str,
    volume_col: str,
) -> tuple[list[float], list[float]]:
    inventory: dict[str, deque[tuple[pd.Timestamp, float, float]]] = defaultdict(deque)
    realized_returns: list[float] = []
    holding_days: list[float] = []

    ordered = df.sort_values(date_col).copy()
    for row in ordered.itertuples(index=False):
        symbol = getattr(row, symbol_col)
        side = getattr(row, side_col)
        price = float(getattr(row, price_col))
        volume = float(getattr(row, volume_col))
        trade_date = getattr(row, date_col)

        if pd.isna(symbol) or pd.isna(trade_date) or volume <= 0 or price <= 0:
            continue

        if side == "buy":
            inventory[symbol].append((trade_date, volume, price))
            continue

        if side != "sell":
            continue

        remaining = volume
        while remaining > 0 and inventory[symbol]:
            buy_date, buy_volume, buy_price = inventory[symbol][0]
            matched = min(remaining, buy_volume)
            if buy_price > 0:
                realized_returns.append((price - buy_price) / buy_price)
            holding_days.append(max((trade_date - buy_date).days, 0))
            remaining -= matched
            buy_volume -= matched
            if buy_volume <= 1e-9:
                inventory[symbol].popleft()
            else:
                inventory[symbol][0] = (buy_date, buy_volume, buy_price)

    return realized_returns, holding_days


def strategy_equity_curve(df: pd.DataFrame) -> np.ndarray:
    cash = safe_numeric(df["cash_balance"]).ffill().fillna(0.0).to_numpy(dtype=float)
    posi = safe_numeric(df["posi_balance"]).ffill().fillna(0.0).to_numpy(dtype=float)
    equity = cash + posi
    equity = np.where(np.isfinite(equity), equity, np.nan)
    equity = pd.Series(equity).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    return equity


def client_equity_curve(df: pd.DataFrame, initial_capital: float) -> np.ndarray:
    cash = float(initial_capital)
    position_qty: dict[str, float] = defaultdict(float)
    position_cost: dict[str, float] = defaultdict(float)
    equity = []
    ordered = df.sort_values("trade_date").copy()

    for row in ordered.itertuples(index=False):
        symbol = str(getattr(row, "stock_code"))
        side = getattr(row, "trade_side")
        price = float(getattr(row, "price"))
        volume = float(getattr(row, "volume"))
        amount = float(getattr(row, "amount"))
        if volume <= 0 or price <= 0:
            continue
        if side == "buy":
            cash -= amount
            position_qty[symbol] += volume
            position_cost[symbol] += amount
        elif side == "sell":
            cash += amount
            sell_qty = min(volume, position_qty[symbol])
            if sell_qty > 0 and position_qty[symbol] > 0:
                avg_cost = position_cost[symbol] / position_qty[symbol]
                position_qty[symbol] -= sell_qty
                position_cost[symbol] -= avg_cost * sell_qty
                if position_qty[symbol] <= 1e-9:
                    position_qty[symbol] = 0.0
                    position_cost[symbol] = 0.0

        market_value = 0.0
        for sym, qty in position_qty.items():
            if qty <= 0:
                continue
            if sym == symbol:
                proxy_price = price
            else:
                proxy_price = position_cost[sym] / qty if qty > 0 else 0.0
            market_value += qty * proxy_price
        equity.append(cash + market_value)

    return np.array(equity, dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def min_max_scale(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if math.isclose(min_v, max_v):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)
