"""Fetch a small, reproducible set of SEC financial facts for risk monitoring.

SEC company facts are used only as a public quarterly proxy. The report does
not pretend to measure AI revenue when a company does not disclose it in a
standardized tag; those factors remain explicitly marked for manual review.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPANY_PATH = ROOT / "config" / "fundamental_companies.csv"
OUTPUT_CSV = ROOT / "output" / "fundamental_report.csv"
OUTPUT_MD = ROOT / "output" / "fundamental_report.md"
SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_HEADERS = {"User-Agent": "a-share-portfolio-monitor/1.0 (public research tool)"}


CONCEPTS = {
    # Modern filers generally use the contract-revenue tag. Older tags remain
    # as fallbacks, but the newest valid reporting period wins.
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenue",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "inventory": ["InventoryNet", "InventoryGross"],
}


def fetch_facts(cik: int) -> dict[str, Any]:
    request = Request(SEC_URL.format(cik=cik), headers=SEC_HEADERS)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fact_entries(facts: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in names:
        units = (us_gaap.get(name) or {}).get("units", {})
        for values in units.values():
            result.extend(values)
    return result


def latest_quarter(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in entries:
        if not item.get("start") or not item.get("end") or item.get("val") is None:
            continue
        start = pd.to_datetime(item["start"], errors="coerce")
        end = pd.to_datetime(item["end"], errors="coerce")
        days = (end - start).days
        if pd.isna(start) or pd.isna(end) or not 60 <= days <= 130:
            continue
        if item.get("form") not in {"10-Q", "10-K", "20-F", "40-F"}:
            continue
        candidates.append((end, item))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def latest_balance(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in entries:
        if not item.get("end") or item.get("val") is None or item.get("start"):
            continue
        if item.get("form") not in {"10-Q", "10-K", "20-F", "40-F"}:
            continue
        candidates.append((pd.to_datetime(item["end"], errors="coerce"), item))
    candidates = [item for item in candidates if not pd.isna(item[0])]
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def pick_value(facts: dict[str, Any], key: str, balance: bool = False) -> tuple[str | None, float | None]:
    entries = fact_entries(facts, CONCEPTS[key])
    item = latest_balance(entries) if balance else latest_quarter(entries)
    if not item:
        return None, None
    return item.get("end"), float(item["val"])


def collect() -> pd.DataFrame:
    companies = pd.read_csv(COMPANY_PATH, dtype={"cik": int})
    rows: list[dict[str, Any]] = []
    for _, company in companies.iterrows():
        row: dict[str, Any] = {
            "company": company["company"], "group": company["group"], "cik": int(company["cik"]),
            "report_end": None, "revenue_usd_b": np.nan, "operating_cash_flow_usd_b": np.nan,
            "capex_usd_b": np.nan, "capex_to_operating_cash_flow_pct": np.nan,
            "inventory_usd_b": np.nan, "status": "数据缺失/待人工核验", "source": "SEC Company Facts",
        }
        try:
            facts = fetch_facts(int(company["cik"]))
            dates = []
            for key in ("revenue", "operating_cash_flow", "capex"):
                end, value = pick_value(facts, key)
                if end:
                    dates.append(end)
                if value is not None:
                    row[f"{key}_usd_b"] = value / 1_000_000_000
            end, value = pick_value(facts, "inventory", balance=True)
            if end:
                dates.append(end)
            if value is not None:
                row["inventory_usd_b"] = value / 1_000_000_000
            row["report_end"] = max(dates) if dates else None
            ocf = row["operating_cash_flow_usd_b"]
            capex = row["capex_usd_b"]
            if pd.notna(ocf) and pd.notna(capex) and ocf > 0:
                row["capex_to_operating_cash_flow_pct"] = abs(capex) / ocf * 100
            if company["group"] == "memory":
                row["status"] = "观察库存、现金流与价格是否背离"
            else:
                row["status"] = "观察资本开支、现金流与AI变现是否匹配"
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"{company['company']} SEC数据获取失败：{exc}", file=sys.stderr)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def build_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# 基本面与行业因子面板", "",
        f"更新时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", "",
        "说明：自动部分仅使用SEC公开公司事实数据；不同字段可能来自同一公司最近不同披露期。AI收入、HBM价格、服务器DRAM合约价、先进封装产能等未标准化披露指标仍需结合业绩会和产业链资料人工核验。",
        "", "|公司|类别|报告期|收入（十亿美元）|经营现金流（十亿美元）|资本开支（十亿美元）|资本开支/经营现金流|库存（十亿美元）|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        lines.append("|{}|{}|{}|{}|{}|{}|{}%|{}|{}|".format(row["company"], row["group"], row["report_end"] or "—", fmt(row["revenue_usd_b"]), fmt(row["operating_cash_flow_usd_b"]), fmt(row["capex_usd_b"]), fmt(row["capex_to_operating_cash_flow_pct"]), fmt(row["inventory_usd_b"]), row["status"]))
    lines.extend(["", "解读规则：资本开支单季变动不能单独构成卖出理由；需要与AI收入/使用量、存储价格、库存和现金流至少两项独立信号交叉验证。", ""])
    return "\n".join(lines)


def main() -> None:
    table = collect()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_MD.write_text(build_markdown(table), encoding="utf-8")
    print(f"已生成基本面面板：{len(table)} 家公司。")


if __name__ == "__main__":
    main()
