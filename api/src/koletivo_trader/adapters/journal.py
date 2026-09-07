from __future__ import annotations

import csv
import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from koletivo_trader.domain.models import SessionContext, Signal, Trade
from koletivo_trader.paths import JOURNAL_DIR

_LOCK = threading.Lock()

ORDER_FIELDS = [
    "ts",
    "mode",
    "symbol",
    "bar_id",
    "side",
    "chart_type",
    "predicted_chart_type",
    "hit_pct",
    "phrase",
    "entry",
    "stop",
    "take",
    "contracts",
    "sent",
    "reason",
]
TRADE_FIELDS = [
    "ticket",
    "open_ts",
    "close_ts",
    "side",
    "entry",
    "exit",
    "stop",
    "take",
    "points",
    "pnl",
    "result",
    "contracts",
    "reason",
]
LEDGER_FIELDS = [
    "ts",
    "event",
    "balance_open",
    "equity",
    "balance",
    "margin_free",
    "open_pnl",
    "closed_pnl_day",
    "n_trades",
    "n_wins",
]


class CsvJournal:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or JOURNAL_DIR

    def _dir(self, day: date) -> Path:
        path = self.root / day.isoformat()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _append(self, path: Path, fields: list[str], row: dict[str, Any]) -> None:
        with _LOCK:
            new_file = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if new_file:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in fields})
                handle.flush()

    def save_context(self, day: date, context: SessionContext) -> None:
        path = self._dir(day) / "context.json"
        path.write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_context(self, day: date) -> SessionContext | None:
        path = self._dir(day) / "context.json"
        if not path.exists():
            return None
        return SessionContext.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def append_order(
        self,
        day: date,
        signal: Signal,
        *,
        mode: str,
        symbol: str,
        bar_id: str,
        contracts: float,
        ts: datetime | None = None,
    ) -> None:
        stamp = ts or datetime.now()
        self._append(
            self._dir(day) / "orders.csv",
            ORDER_FIELDS,
            {
                "ts": stamp.isoformat(timespec="seconds"),
                "mode": mode,
                "symbol": symbol,
                "bar_id": bar_id,
                "side": signal.side.value,
                "chart_type": signal.chart_type.value,
                "predicted_chart_type": ""
                if signal.predicted_chart_type is None
                else signal.predicted_chart_type.value,
                "hit_pct": f"{signal.hit_pct:.4f}",
                "phrase": signal.phrase,
                "entry": signal.entry,
                "stop": signal.stop,
                "take": signal.take,
                "contracts": contracts,
                "sent": str(signal.sent).lower(),
                "reason": signal.reason,
            },
        )

    def append_trade(self, day: date, trade: Trade) -> None:
        self._append(
            self._dir(day) / "trades.csv",
            TRADE_FIELDS,
            {
                "ticket": trade.ticket or "",
                "open_ts": trade.entry_time.isoformat(timespec="seconds"),
                "close_ts": "" if trade.exit_time is None else trade.exit_time.isoformat(timespec="seconds"),
                "side": trade.side.value,
                "entry": trade.entry,
                "exit": trade.exit,
                "stop": trade.stop,
                "take": trade.take,
                "points": trade.points,
                "pnl": trade.pnl,
                "result": trade.result.value,
                "contracts": trade.contracts,
                "reason": trade.reason,
            },
        )

    def append_ledger(self, day: date, row: dict[str, Any]) -> None:
        self._append(self._dir(day) / "ledger.csv", LEDGER_FIELDS, row)

    def _read(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def day_payload(self, day: date) -> dict[str, Any]:
        folder = self.root / day.isoformat()
        orders = self._read(folder / "orders.csv")
        trades = self._read(folder / "trades.csv")
        ledger = self._read(folder / "ledger.csv")
        context = None
        ctx_path = folder / "context.json"
        if ctx_path.exists():
            context = json.loads(ctx_path.read_text(encoding="utf-8"))
        first = ledger[0] if ledger else {}
        last = ledger[-1] if ledger else {}
        return {
            "date": day.isoformat(),
            "orders": orders,
            "trades": trades,
            "ledger": ledger,
            "context": context,
            "balance_open": _num(first.get("balance_open") or first.get("balance")),
            "balance": _num(last.get("balance")),
            "equity": _num(last.get("equity")),
            "closed_pnl_day": _num(last.get("closed_pnl_day")),
            "n_trades": len(trades),
        }

    def real_days(self) -> list[str]:
        if not self.root.exists():
            return []
        days: list[str] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            trades = folder / "trades.csv"
            ledger = folder / "ledger.csv"
            if trades.exists() or ledger.exists():
                days.append(folder.name)
        return days

    def first_real_day(self) -> str | None:
        days = self.real_days()
        return days[0] if days else None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
