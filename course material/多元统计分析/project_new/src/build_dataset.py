from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALL_FEATURES,
    CLIENT_FILES,
    RAW_DATA_DIR,
    SELECTED_STRATEGIES,
    TABLE_DIR,
)
from utils import (
    client_equity_curve,
    compute_trade_regularity,
    detect_trade_side,
    fifo_realized_returns,
    log1p_clip,
    max_drawdown_from_curve,
    normalize_dates,
    safe_numeric,
    sharpe_like,
)


def load_summary_table(workbook_name: str) -> pd.DataFrame:
    workbook = RAW_DATA_DIR / workbook_name
    xl = pd.ExcelFile(workbook)
    summary = pd.read_excel(workbook, sheet_name=xl.sheet_names[0])
    summary.columns = [str(c).strip() for c in summary.columns]
    return summary


def summary_metrics(workbook_name: str, summary_name: str) -> dict[str, float]:
    summary = load_summary_table(workbook_name)
    row = summary.loc[summary["策略名称"].astype(str).str.strip() == summary_name]
    if row.empty:
        return {
            "summary_cum_return_2025": np.nan,
            "summary_drawdown_2025": np.nan,
            "summary_annual_return": np.nan,
        }
    row = row.iloc[0]
    return {
        "summary_cum_return_2025": pd.to_numeric(row["累计收益率(25年1月至今)"], errors="coerce"),
        "summary_drawdown_2025": pd.to_numeric(row["最大回撤25年至今"], errors="coerce"),
        "summary_annual_return": pd.to_numeric(row["平均年化收益率(近6年)"], errors="coerce"),
    }


def coerce_metric(value: float, fallback: float = 0.0) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or not np.isfinite(value):
        return float(fallback)
    return float(value)


def build_strategy_row(meta: dict) -> dict:
    workbook = RAW_DATA_DIR / meta["workbook"]
    df = pd.read_excel(workbook, sheet_name=meta["sheet_name"])
    df = df.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["trade_side"] = df["btype"].astype(str).map(detect_trade_side)

    numeric_cols = ["volume", "amount", "fee", "net_amount", "cash_balance", "posi_balance", "vwap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    df = df[df["trade_time"].notna()].copy()
    df = df[df["trade_side"].isin(["buy", "sell"])].copy()
    df = df[safe_numeric(df["volume"]).fillna(0) != 0].copy()
    df["abs_volume"] = safe_numeric(df["volume"]).abs()
    df["abs_amount"] = safe_numeric(df["amount"]).abs()
    df["trade_date"] = df["trade_time"].dt.normalize()

    realized_returns, holding_days = fifo_realized_returns(
        df.rename(columns={"trade_side": "trade_side"}),
        date_col="trade_time",
        symbol_col="symbol",
        side_col="trade_side",
        price_col="vwap",
        volume_col="abs_volume",
    )
    realized_arr = np.array(realized_returns, dtype=float)
    holding_arr = np.array(holding_days, dtype=float)

    trade_days = max(df["trade_date"].nunique(), 1)
    daily_amount = df.groupby("trade_date")["abs_amount"].sum()
    avg_cash = safe_numeric(df["cash_balance"]).abs().replace(0, np.nan).mean()
    avg_posi = safe_numeric(df["posi_balance"]).abs().replace(0, np.nan).mean()
    avg_equity_proxy = np.nanmean([avg_cash, avg_cash + avg_posi if pd.notna(avg_posi) else np.nan])
    avg_equity_proxy = float(avg_equity_proxy) if np.isfinite(avg_equity_proxy) and avg_equity_proxy > 0 else 1.0

    metrics = summary_metrics(meta["workbook"], meta["summary_name"])
    stock_amounts = df.groupby("symbol")["abs_amount"].sum().sort_values(ascending=False)
    official_return = coerce_metric(metrics["summary_cum_return_2025"], fallback=float(realized_arr.mean()) if realized_arr.size else 0.0)
    official_drawdown = coerce_metric(metrics["summary_drawdown_2025"], fallback=-0.2)
    official_annual = coerce_metric(metrics["summary_annual_return"], fallback=official_return)
    realized_vol = float(realized_arr.std(ddof=0)) if realized_arr.size else max(abs(official_drawdown) / 2, 0.01)
    strategy_sharpe = official_return / max(realized_vol, 1e-6)

    row = {
        "entity_name": meta["display_name"],
        "entity_type": "strategy",
        "theme": meta["theme"],
        "n_records": int(len(df)),
        "n_trade_days": int(trade_days),
        "log_trade_freq": log1p_clip(len(df) / trade_days),
        "avg_hold_days": float(np.median(holding_arr)) if holding_arr.size else max(trade_days / max(len(df), 1), 1.0),
        "turnover_rate": float(daily_amount.mean() / avg_equity_proxy),
        "log_avg_trade_amount": log1p_clip(df["abs_amount"].mean()),
        "buy_sell_ratio": float((df["trade_side"] == "buy").sum() / max((df["trade_side"] == "sell").sum(), 1)),
        "trade_regularity": compute_trade_regularity(df["trade_date"]),
        "num_stocks": float(df["symbol"].nunique()),
        "concentration_top3": float(stock_amounts.head(3).sum() / max(stock_amounts.sum(), 1.0)),
        "concentration_top1": float(stock_amounts.head(1).sum() / max(stock_amounts.sum(), 1.0)),
        "avg_position_pct": float(avg_posi / avg_equity_proxy) if pd.notna(avg_posi) else 0.0,
        "position_peak_ratio": float(
            safe_numeric(df["posi_balance"]).abs().max() / max(float(avg_posi) if pd.notna(avg_posi) else 1.0, 1e-6)
        ),
        "stock_turnover": float(df["symbol"].nunique() / max(len(df), 1)),
        "avg_return_pct": official_return,
        "win_rate": float((realized_arr > 0).mean()) if realized_arr.size else 0.0,
        "profit_loss_ratio": float(
            realized_arr[realized_arr > 0].mean() / abs(realized_arr[realized_arr < 0].mean())
        ) if (realized_arr > 0).any() and (realized_arr < 0).any() else 1.0,
        "max_drawdown": official_drawdown,
        "return_volatility": realized_vol,
        "sharpe_approx": float(strategy_sharpe),
        "summary_cum_return_2025": metrics["summary_cum_return_2025"],
        "summary_drawdown_2025": metrics["summary_drawdown_2025"],
        "summary_annual_return": official_annual,
    }
    return row


def load_client_workbook(meta: dict) -> pd.DataFrame:
    workbook = RAW_DATA_DIR / meta["workbook"]
    df = pd.read_excel(workbook)
    rename_map = {}
    for col in df.columns:
        text = str(col)
        if "交收日期" in text:
            rename_map[col] = "trade_date"
        elif "业务标示" in text:
            rename_map[col] = "business_type"
        elif "证券代码" in text:
            rename_map[col] = "stock_code"
        elif "证券名称" in text:
            rename_map[col] = "stock_name"
        elif "成交价格" in text:
            rename_map[col] = "price"
        elif "成交数量" in text:
            rename_map[col] = "volume"
        elif "成交金额" in text:
            rename_map[col] = "amount"
    df = df.rename(columns=rename_map)
    df["trade_date"] = normalize_dates(df["trade_date"])
    df["price"] = safe_numeric(df["price"])
    df["volume"] = safe_numeric(df["volume"]).abs()
    if "amount" in df.columns:
        df["amount"] = safe_numeric(df["amount"]).abs()
    else:
        df["amount"] = df["price"] * df["volume"]
    df["trade_side"] = df["business_type"].astype(str).map(detect_trade_side)
    return df


def build_client_row(meta: dict) -> dict:
    df = load_client_workbook(meta)
    df = df[df["trade_date"].notna()].copy()
    df = df[df["trade_side"].isin(["buy", "sell"])].copy()

    realized_returns, holding_days = fifo_realized_returns(
        df,
        date_col="trade_date",
        symbol_col="stock_code",
        side_col="trade_side",
        price_col="price",
        volume_col="volume",
    )
    realized_arr = np.array(realized_returns, dtype=float)
    holding_arr = np.array(holding_days, dtype=float)

    stock_amounts = df.groupby("stock_code")["amount"].sum().sort_values(ascending=False)
    daily_amount = df.groupby("trade_date")["amount"].sum()
    trade_days = max(df["trade_date"].nunique(), 1)
    if realized_arr.size:
        clipped_realized = np.clip(realized_arr, -0.95, 3.0)
        realized_index = np.cumprod(1.0 + clipped_realized)
        client_return = float(realized_arr.mean())
        client_drawdown = float(max_drawdown_from_curve(realized_index))
        client_vol = float(realized_arr.std(ddof=0))
        client_sharpe = float(sharpe_like(realized_arr))
    else:
        client_return = 0.0
        client_drawdown = 0.0
        client_vol = 0.0
        client_sharpe = 0.0

    avg_position_pct = float(min(daily_amount.mean() / meta["initial_capital"], 1.0))
    peak_position_pct = float(min(daily_amount.max() / meta["initial_capital"], 1.5))

    row = {
        "entity_name": meta["client_name"],
        "entity_type": "real_client",
        "theme": "真实客户",
        "n_records": int(len(df)),
        "n_trade_days": int(trade_days),
        "log_trade_freq": log1p_clip(len(df) / trade_days),
        "avg_hold_days": float(np.median(holding_arr)) if holding_arr.size else max(trade_days / max(len(df), 1), 1.0),
        "turnover_rate": float(daily_amount.mean() / max(meta["initial_capital"], 1.0)),
        "log_avg_trade_amount": log1p_clip(df["amount"].mean()),
        "buy_sell_ratio": float((df["trade_side"] == "buy").sum() / max((df["trade_side"] == "sell").sum(), 1)),
        "trade_regularity": compute_trade_regularity(df["trade_date"]),
        "num_stocks": float(df["stock_code"].nunique()),
        "concentration_top3": float(stock_amounts.head(3).sum() / max(stock_amounts.sum(), 1.0)),
        "concentration_top1": float(stock_amounts.head(1).sum() / max(stock_amounts.sum(), 1.0)),
        "avg_position_pct": avg_position_pct,
        "position_peak_ratio": float(peak_position_pct / max(avg_position_pct, 1e-6)),
        "stock_turnover": float(df["stock_code"].nunique() / max(len(df), 1)),
        "avg_return_pct": client_return,
        "win_rate": float((realized_arr > 0).mean()) if realized_arr.size else 0.0,
        "profit_loss_ratio": float(
            realized_arr[realized_arr > 0].mean() / abs(realized_arr[realized_arr < 0].mean())
        ) if (realized_arr > 0).any() and (realized_arr < 0).any() else 1.0,
        "max_drawdown": client_drawdown,
        "return_volatility": client_vol,
        "sharpe_approx": client_sharpe,
        "summary_cum_return_2025": np.nan,
        "summary_drawdown_2025": np.nan,
        "summary_annual_return": np.nan,
    }
    return row


def generate_synthetic_clients(real_clients: pd.DataFrame, n_clients: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = real_clients[ALL_FEATURES].copy()
    mu = base.mean(axis=0)
    sigma = base.std(axis=0).replace(0, 0.05)

    archetypes = [
        ("保守防御型", {"log_trade_freq": -0.6, "avg_hold_days": 1.2, "turnover_rate": -0.4, "concentration_top3": 0.4, "avg_return_pct": 0.02, "max_drawdown": 0.03}),
        ("均衡配置型", {"num_stocks": 0.4, "concentration_top3": -0.3, "position_peak_ratio": -0.2, "return_volatility": -0.1}),
        ("高频交易型", {"log_trade_freq": 1.0, "avg_hold_days": -0.8, "turnover_rate": 1.2, "trade_regularity": 0.4, "return_volatility": 0.2}),
        ("成长进攻型", {"log_avg_trade_amount": 0.4, "num_stocks": 0.3, "avg_return_pct": 0.03, "return_volatility": 0.3, "max_drawdown": -0.05}),
        ("集中持仓型", {"concentration_top3": 0.8, "concentration_top1": 0.7, "avg_position_pct": 0.3, "position_peak_ratio": 0.5}),
    ]

    rows = []
    for i in range(n_clients):
        archetype_name, shift_map = archetypes[i % len(archetypes)]
        noise = rng.normal(0, 0.35, len(ALL_FEATURES))
        values = mu.copy()
        for j, feature in enumerate(ALL_FEATURES):
            shift = shift_map.get(feature, 0.0)
            values[feature] = mu[feature] + sigma[feature] * (shift + noise[j])

        values["log_trade_freq"] = max(values["log_trade_freq"], 0.05)
        values["avg_hold_days"] = np.clip(values["avg_hold_days"], 1.0, 120.0)
        values["turnover_rate"] = max(values["turnover_rate"], 0.01)
        values["log_avg_trade_amount"] = max(values["log_avg_trade_amount"], 1.0)
        values["buy_sell_ratio"] = np.clip(values["buy_sell_ratio"], 0.5, 2.5)
        values["trade_regularity"] = np.clip(values["trade_regularity"], 0.0, 3.0)
        values["num_stocks"] = np.clip(values["num_stocks"], 5.0, 120.0)
        values["concentration_top3"] = np.clip(values["concentration_top3"], 0.1, 0.95)
        values["concentration_top1"] = np.clip(values["concentration_top1"], 0.05, values["concentration_top3"])
        values["avg_position_pct"] = np.clip(values["avg_position_pct"], 0.1, 1.0)
        values["position_peak_ratio"] = np.clip(values["position_peak_ratio"], 1.0, 3.5)
        values["stock_turnover"] = np.clip(values["stock_turnover"], 0.01, 1.0)
        values["avg_return_pct"] = np.clip(values["avg_return_pct"], -0.08, 0.12)
        values["win_rate"] = np.clip(values["win_rate"], 0.2, 0.9)
        values["profit_loss_ratio"] = np.clip(values["profit_loss_ratio"], 0.3, 3.0)
        values["max_drawdown"] = np.clip(values["max_drawdown"], -0.6, -0.01)
        values["return_volatility"] = np.clip(values["return_volatility"], 0.01, 0.25)
        values["sharpe_approx"] = np.clip(values["sharpe_approx"], -1.5, 3.0)

        row = {
            "entity_name": f"{archetype_name}_{i + 1:02d}",
            "entity_type": "synthetic_client",
            "theme": archetype_name,
            "n_records": int(np.expm1(values["log_trade_freq"]) * 60 + rng.integers(20, 100)),
            "n_trade_days": int(rng.integers(30, 180)),
            "summary_cum_return_2025": np.nan,
            "summary_drawdown_2025": np.nan,
            "summary_annual_return": np.nan,
        }
        for feature in ALL_FEATURES:
            row[feature] = float(values[feature])
        rows.append(row)

    return pd.DataFrame(rows)


def build_full_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy_rows = [build_strategy_row(meta) for meta in SELECTED_STRATEGIES]
    client_rows = [build_client_row(meta) for meta in CLIENT_FILES]

    strategy_df = pd.DataFrame(strategy_rows)
    real_client_df = pd.DataFrame(client_rows)
    synthetic_df = generate_synthetic_clients(real_client_df)

    full_df = pd.concat([strategy_df, real_client_df, synthetic_df], ignore_index=True)

    strategy_df.to_csv(TABLE_DIR / "strategy_features.csv", index=False, encoding="utf-8-sig")
    real_client_df.to_csv(TABLE_DIR / "real_client_features.csv", index=False, encoding="utf-8-sig")
    synthetic_df.to_csv(TABLE_DIR / "synthetic_client_features.csv", index=False, encoding="utf-8-sig")
    full_df.to_csv(TABLE_DIR / "full_feature_dataset.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_strategies": int((full_df["entity_type"] == "strategy").sum()),
        "n_real_clients": int((full_df["entity_type"] == "real_client").sum()),
        "n_synthetic_clients": int((full_df["entity_type"] == "synthetic_client").sum()),
        "n_total": int(len(full_df)),
        "features": ALL_FEATURES,
    }
    with open(TABLE_DIR / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return full_df, strategy_df, real_client_df


if __name__ == "__main__":
    build_full_dataset()
