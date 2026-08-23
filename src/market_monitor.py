"""Fetch A-share daily bars and calculate technical indicators.

The script intentionally stores only market data and indicators. Personal
portfolio quantities, costs and account balances should stay outside a public
GitHub repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_PATH = ROOT / "config" / "watchlist.csv"
DATA_PATH = ROOT / "data" / "ohlcv.csv"
REPORT_CSV_PATH = ROOT / "output" / "technical_report.csv"
REPORT_MD_PATH = ROOT / "output" / "technical_report.md"

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
NAV_URL = "https://fund.eastmoney.com/pingzhongdata/{code}.js"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def secid(row: pd.Series) -> str:
    market_id = "1" if str(row["market"]).upper() == "SH" else "0"
    return f"{market_id}.{str(row['code']).zfill(6)}"


def fetch_kline(row: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV, preferring Tencent and falling back to Eastmoney.

    Tencent's public endpoint is used first because it was more reliable in
    repeated GitHub Actions-style requests during testing. Eastmoney remains
    a fallback and is also used for future extensions that need more fields.
    """

    try:
        return fetch_tencent_kline(row, start_date, end_date)
    except Exception as tencent_error:
        try:
            return fetch_eastmoney_kline(row, start_date, end_date)
        except Exception as eastmoney_error:
            raise RuntimeError(
                f"腾讯和东方财富接口均失败；腾讯={tencent_error}；东方财富={eastmoney_error}"
            ) from eastmoney_error


def standardize_frame(
    records: list[list[str]], row: pd.Series, include_amount: bool = False
) -> pd.DataFrame:
    """Convert source-specific rows into the common OHLCV schema."""

    if not records:
        raise RuntimeError(f"没有返回行情数据：{row['code']} {row['name']}")
    # Tencent occasionally appends an extra field to a few historical rows;
    # the first six fields are the stable date/open/close/high/low/volume set.
    records = [record[:6] for record in records if len(record) >= 6]
    columns = ["date", "open", "close", "high", "low", "volume"]
    frame = pd.DataFrame(records, columns=columns)
    frame["code"] = str(row["code"]).zfill(6)
    frame["name"] = row["name"]
    frame["market"] = row["market"]
    frame["kind"] = row["kind"]

    numeric_columns = ["open", "close", "high", "low", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date", "close"])
    frame["amount"] = np.nan
    frame["pct_change"] = frame["close"].pct_change() * 100
    frame["turnover"] = np.nan
    return frame[
        [
            "date",
            "code",
            "name",
            "market",
            "kind",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pct_change",
            "turnover",
        ]
    ]


def fetch_tencent_kline(row: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars from Tencent's public quotation endpoint."""

    market = "sh" if str(row["market"]).upper() == "SH" else "sz"
    adjustment = "qfq" if int(row.get("fqt", 1)) == 1 else "bfq"
    symbol = f"{market}{str(row['code']).zfill(6)}"
    params = {"param": f"{symbol},day,,,500,{adjustment}"}
    response = requests.get(TENCENT_KLINE_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    source = (payload.get("data") or {}).get(symbol) or {}
    key = "qfqday" if adjustment == "qfq" and source.get("qfqday") else "day"
    records = source.get(key) or []
    frame = standardize_frame(records, row)
    return frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()


def fetch_eastmoney_kline(row: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars from Eastmoney's public historical endpoint."""

    params = {
        "secid": secid(row),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",  # daily
        "fqt": str(int(row.get("fqt", 1))),
        "beg": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    payload = request_json(KLINE_URL, params)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"没有返回行情数据：{row['code']} {row['name']}")

    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    records = [item.split(",") for item in klines]
    frame = pd.DataFrame(records, columns=columns[: len(records[0])])
    frame["code"] = str(row["code"]).zfill(6)
    frame["name"] = row["name"]
    frame["market"] = row["market"]
    frame["kind"] = row["kind"]

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame[
        [
            "date",
            "code",
            "name",
            "market",
            "kind",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pct_change",
            "turnover",
        ]
    ]


def fetch_latest_nav(code: str) -> tuple[str | None, float | None]:
    """Best-effort ETF NAV lookup; failure must not stop the whole update."""

    try:
        response = requests.get(NAV_URL.format(code=code), headers=HEADERS, timeout=30)
        response.raise_for_status()
        match = re.search(r"Data_netWorthTrend\s*=\s*(\[.*?\]);", response.text, re.S)
        if not match:
            return None, None
        trend = json.loads(match.group(1))
        valid = [item for item in trend if item.get("y") is not None]
        if not valid:
            return None, None
        latest = valid[-1]
        nav_date = datetime.fromtimestamp(latest["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        return nav_date, float(latest["y"])
    except (ValueError, KeyError, TypeError, requests.RequestException, json.JSONDecodeError):
        return None, None


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    for period in (5, 10, 20, 60):
        ma = close.rolling(period, min_periods=period).mean()
        frame[f"ma{period}"] = ma
        frame[f"bias{period}"] = (close / ma - 1) * 100

    lowest = low.rolling(9, min_periods=9).min()
    highest = high.rolling(9, min_periods=9).max()
    denominator = (highest - lowest).replace(0, np.nan)
    rsv = ((close - lowest) / denominator * 100).fillna(50)
    frame["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    frame["d"] = frame["k"].ewm(alpha=1 / 3, adjust=False).mean()
    frame["j"] = 3 * frame["k"] - 2 * frame["d"]
    frame["kdj_cross"] = np.select(
        [
            (frame["k"] > frame["d"]) & (frame["k"].shift(1) <= frame["d"].shift(1)),
            (frame["k"] < frame["d"]) & (frame["k"].shift(1) >= frame["d"].shift(1)),
        ],
        ["金叉", "死叉"],
        default="",
    )

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["dif"] = ema12 - ema26
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = 2 * (frame["dif"] - frame["dea"])

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    frame["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    frame["atr14"] = true_range.rolling(14, min_periods=14).mean()
    frame["volume_ratio20"] = volume / volume.rolling(20, min_periods=20).mean()

    frame["trend"] = np.select(
        [
            (close > frame["ma20"]) & (frame["ma20"] > frame["ma60"]),
            (close < frame["ma20"]) & (frame["ma20"] < frame["ma60"]),
        ],
        ["多头趋势", "空头趋势"],
        default="震荡/过渡",
    )
    frame["signal"] = frame.apply(signal_for_row, axis=1)
    return frame


def signal_for_row(row: pd.Series) -> str:
    required = ["ma20", "ma60", "bias20", "k", "d", "j", "volume_ratio20"]
    if any(pd.isna(row.get(column)) for column in required):
        return "数据不足"
    if row["bias5"] > 5 or row["j"] > 100:
        return "短线偏热，不追高"
    if row["k"] < 25 and row["d"] < 25 and row["k"] > row["d"] and row["close"] > row["ma5"]:
        return "低位反弹观察"
    if row["trend"] == "多头趋势" and row["k"] > row["d"] and -5 <= row["bias20"] <= 5:
        return "趋势偏强，回撤观察"
    if row["close"] < row["ma20"] and row["bias20"] < -5:
        return "弱势，暂不加仓"
    return "中性观察"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(latest: pd.DataFrame, updated_at: str) -> str:
    headers = [
        "代码",
        "名称",
        "日期",
        "收盘",
        "K",
        "D",
        "J",
        "BIAS5",
        "BIAS20",
        "MACD柱",
        "RSI14",
        "量比20",
        "趋势",
        "技术状态",
        "溢价率",
    ]
    lines = [
        "# A股持仓技术监控",
        "",
        f"更新时间：{updated_at}",
        "",
        "说明：数据来自公开行情接口；技术状态仅用于研究，不构成自动交易指令。513310 的溢价率使用最近可取得的基金净值估算，净值日期可能早于行情日期。",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for _, row in latest.sort_values("code").iterrows():
        values = [
            row["code"],
            row["name"],
            row["date"],
            fmt(row["close"]),
            fmt(row["k"]),
            fmt(row["d"]),
            fmt(row["j"]),
            fmt(row["bias5"]),
            fmt(row["bias20"]),
            fmt(row["macd_hist"], 3),
            fmt(row["rsi14"]),
            fmt(row["volume_ratio20"]),
            row["trend"],
            row["signal"],
            fmt(row.get("premium_rate")),
        ]
        lines.append("|" + "|".join(str(value).replace("|", "\\|") for value in values) + "|")
    lines.extend(
        [
            "",
            "指标口径：KDJ(9,3,3)、BIAS5/10/20、MACD(12,26,9)、RSI14、MA5/10/20/60、ATR14、20日量比。",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=420)
    args = parser.parse_args()

    watchlist = pd.read_csv(WATCHLIST_PATH, dtype={"code": str})
    end = date.today()
    start = end - timedelta(days=args.lookback_days)
    all_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for _, row in watchlist.iterrows():
        try:
            all_frames.append(fetch_kline(row, start.isoformat(), end.isoformat()))
        except Exception as exc:  # keep other symbols updating if one endpoint fails
            errors.append(f"{row['code']} {row['name']}: {exc}")

    if not all_frames:
        raise RuntimeError("所有标的均未取得行情数据。\n" + "\n".join(errors))

    fresh = pd.concat(all_frames, ignore_index=True)
    if DATA_PATH.exists():
        old = pd.read_csv(DATA_PATH, dtype={"code": str})
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh
    combined["code"] = combined["code"].astype(str).str.zfill(6)
    combined = combined.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    calculated: list[pd.DataFrame] = []
    for code, group in combined.groupby("code", sort=False):
        calculated.append(add_indicators(group))
    indicators = pd.concat(calculated, ignore_index=True)
    latest = indicators.sort_values("date").groupby("code", as_index=False).tail(1).copy()
    latest["nav_date"] = None
    latest["nav"] = np.nan
    latest["premium_rate"] = np.nan

    nav_required = set(watchlist.loc[watchlist["nav_required"].astype(str) == "1", "code"].astype(str))
    for index, row in latest.iterrows():
        if str(row["code"]) not in nav_required:
            continue
        nav_date, nav = fetch_latest_nav(str(row["code"]))
        latest.at[index, "nav_date"] = nav_date
        latest.at[index, "nav"] = nav
        if nav and row["close"]:
            latest.at[index, "premium_rate"] = (row["close"] / nav - 1) * 100

    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(REPORT_CSV_PATH, index=False, encoding="utf-8-sig")
    updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    REPORT_MD_PATH.write_text(build_markdown(latest, updated_at), encoding="utf-8")

    print(f"已更新 {len(latest)} 个标的，行情记录 {len(combined)} 行。")
    if errors:
        print("部分标的更新失败：", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)


if __name__ == "__main__":
    run()
