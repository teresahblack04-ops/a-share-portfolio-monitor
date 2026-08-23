"""Build an automatic market-proxy risk dashboard from the public price history.

This intentionally does not contain personal quantities, costs, or account
balances. The group returns are equal-weight proxies, not the user's account
return. They are useful for detecting regime changes in the monitored themes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ohlcv.csv"
FACTOR_PATH = ROOT / "config" / "monitoring_factors.csv"
OUTPUT_CSV = ROOT / "output" / "risk_dashboard.csv"
OUTPUT_MD = ROOT / "output" / "risk_dashboard.md"

GROUPS = {
    "半导体代理组合": ["513310", "588710"],
    "A500代理": ["563360"],
    "黄金代理": ["518880"],
}


def fmt(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def group_series(data: pd.DataFrame, codes: list[str]) -> pd.Series:
    selected = data[data["code"].isin(codes)].copy()
    selected["close"] = pd.to_numeric(selected["close"], errors="coerce")
    wide = selected.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    return wide.pct_change().mean(axis=1).add(1).cumprod()


def make_dashboard() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(DATA_PATH, dtype={"code": str})
    data["code"] = data["code"].str.zfill(6)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["code", "date"])
    base = group_series(data, GROUPS["A500代理"])
    rows: list[dict[str, object]] = []
    alerts: list[str] = []

    for group, codes in GROUPS.items():
        series = group_series(data, codes).dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        row: dict[str, object] = {
            "group": group,
            "codes": ",".join(codes),
            "latest_date": series.index[-1].strftime("%Y-%m-%d"),
            "return_5d_pct": np.nan,
            "return_20d_pct": np.nan,
            "return_60d_pct": np.nan,
            "volatility20_pct": np.nan,
            "drawdown60_pct": np.nan,
            "relative_20d_vs_a500_pct": np.nan,
            "status": "数据不足",
        }
        for period, key in ((5, "return_5d_pct"), (20, "return_20d_pct"), (60, "return_60d_pct")):
            if len(series) > period:
                row[key] = (latest / float(series.iloc[-period - 1]) - 1) * 100
        if len(series) >= 20:
            returns = series.pct_change().tail(20)
            row["volatility20_pct"] = returns.std() * np.sqrt(252) * 100
        tail = series.tail(60)
        if not tail.empty:
            row["drawdown60_pct"] = (latest / float(tail.max()) - 1) * 100
        if group != "A500代理" and len(series) > 20 and len(base) > 20:
            common = pd.concat([series, base], axis=1, join="inner").dropna()
            if len(common) > 20:
                row["relative_20d_vs_a500_pct"] = (
                    (common.iloc[-1, 0] / common.iloc[-21, 0])
                    / (common.iloc[-1, 1] / common.iloc[-21, 1])
                    - 1
                ) * 100

        if group == "半导体代理组合":
            relative = row["relative_20d_vs_a500_pct"]
            ret20 = row["return_20d_pct"]
            if pd.notna(relative) and relative < -5 and pd.notna(ret20) and ret20 < 0:
                row["status"] = "相对弱势，暂停追涨"
                alerts.append("半导体代理组合相对A500走弱，且20日收益为负")
            elif pd.notna(relative) and relative > 8 and pd.notna(ret20) and ret20 > 10:
                row["status"] = "相对强势，防止短线过热"
                alerts.append("半导体代理组合短期相对强势，注意追高风险")
            else:
                row["status"] = "中性观察"
        elif group == "黄金代理":
            row["status"] = "防守资产观察"
        else:
            row["status"] = "基准代理"
        rows.append(row)
    return pd.DataFrame(rows), alerts


def build_markdown(table: pd.DataFrame, alerts: list[str], factors: pd.DataFrame) -> str:
    lines = [
        "# 行业与组合风险面板",
        "",
        f"更新时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "说明：收益、波动和回撤为等权市场代理，不代表个人账户收益；基本面因子按披露频率更新，不因单日价格波动直接改变投资计划。",
        "",
        "## 自动价格面板",
        "",
        "|代理组合|代码|行情日期|5日收益|20日收益|60日收益|20日波动|60日回撤|相对A500（20日）|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "|{}|{}|{}|{}%|{}%|{}%|{}%|{}%|{}%|{}|".format(
                row["group"], row["codes"], row["latest_date"],
                fmt(row["return_5d_pct"]), fmt(row["return_20d_pct"]),
                fmt(row["return_60d_pct"]), fmt(row["volatility20_pct"]),
                fmt(row["drawdown60_pct"]), fmt(row["relative_20d_vs_a500_pct"]),
                row["status"],
            )
        )
    lines.extend(["", "## 当前预警", ""])
    lines.extend([f"- {item}" for item in alerts] or ["- 暂无自动价格预警。"])
    lines.extend(["", "## 基本面因子清单", "", "|因子|类别|频率|自动化程度|触发原则|影响资产|", "|---|---|---|---|---|---|"])
    for _, row in factors.iterrows():
        lines.append("|{}|{}|{}|{}|{}|{}|".format(row["factor"], row["category"], row["cadence"], row["automation"], row["decision_rule"], row["affected_assets"]))
    lines.extend(["", "分级原则：单一信号=黄色观察；两个独立信号连续恶化=橙色，暂停相关加仓；需求、价格、库存/现金流三类信号同时恶化=红色，复核并降低相关风险暴露。", ""])
    return "\n".join(lines)


def main() -> None:
    table, alerts = make_dashboard()
    factors = pd.read_csv(FACTOR_PATH)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_MD.write_text(build_markdown(table, alerts, factors), encoding="utf-8")
    print(f"已生成风险面板：{len(table)} 个代理组合、{len(factors)} 个监测因子。")


if __name__ == "__main__":
    main()
