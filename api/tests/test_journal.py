from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from koletivo_trader.adapters.journal import CsvJournal
from koletivo_trader.domain.enums import ChartType, Side, TradeResult
from koletivo_trader.domain.models import Signal, Trade


def test_journal_roundtrip(tmp_path: Path) -> None:
    store = CsvJournal(tmp_path)
    day = date(2026, 9, 7)
    sig = Signal(
        side=Side.HOLD,
        entry=1000,
        stop=0,
        take=0,
        chart_type=ChartType.CONSOLIDATION,
        hit_pct=0.4,
        phrase="melhor não comprar nem vender",
        reason="low_hit",
    )
    store.append_order(day, sig, mode="paper", symbol="WIN$", bar_id="x", contracts=1, ts=datetime(2026, 9, 7, 10, 20))
    trade = Trade(
        side=Side.BUY,
        entry_time=datetime(2026, 9, 7, 10, 20),
        exit_time=datetime(2026, 9, 7, 10, 25),
        entry=1000,
        exit=1100,
        stop=900,
        take=1200,
        points=100,
        pnl=20,
        result=TradeResult.GAIN,
        reason="daytrade",
        contracts=1,
        ticket=1,
    )
    store.append_trade(day, trade)
    store.append_ledger(
        day,
        {
            "ts": "2026-09-07T10:25:00",
            "event": "fill",
            "balance_open": 1000,
            "equity": 1020,
            "balance": 1020,
            "margin_free": 1020,
            "open_pnl": 0,
            "closed_pnl_day": 20,
            "n_trades": 1,
            "n_wins": 1,
        },
    )
    payload = store.day_payload(day)
    assert payload["n_trades"] == 1
    assert payload["orders"][0]["side"] == "HOLD"
    assert payload["balance_open"] == 1000
    assert store.first_real_day() == "2026-09-07"
