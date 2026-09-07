from __future__ import annotations

from datetime import datetime, timedelta

from koletivo_trader.domain.enums import ChartType, DayType, Side
from koletivo_trader.domain.fusion import fuse_signals
from koletivo_trader.domain.market import classify_chart, classify_day
from koletivo_trader.domain.models import Candle, SessionContext, Signal
from koletivo_trader.domain.risk import contracts_for_bank, protect_levels


def _bar(i: int, open_: float, high: float, low: float, close: float, start: datetime | None = None) -> Candle:
    ts = (start or datetime(2026, 3, 13, 9, 15)) + timedelta(minutes=i)
    return Candle("WIN$", ts, open_, high, low, close, 1000)


def test_contracts_for_bank() -> None:
    assert contracts_for_bank(500) == 1
    assert contracts_for_bank(1000) == 1
    assert contracts_for_bank(2000) == 2
    assert contracts_for_bank(5000) == 5
    assert contracts_for_bank(10000) == 10


def test_trend_day_up() -> None:
    candles = [_bar(i * 5, 1000 + i * 20, 1005 + i * 20, 998 + i * 20, 1004 + i * 20) for i in range(20)]
    assert classify_day(candles) is DayType.TREND_UP


def test_impulse_chart() -> None:
    candles = [_bar(i, 1000 + i * 8, 1004 + i * 8, 999 + i * 8, 1003 + i * 8) for i in range(15)]
    assert classify_chart(candles) in {ChartType.IMPULSE_UP, ChartType.PULLBACK_UP, ChartType.BREAKOUT}


def test_hold_phrase_mentions_not_trading() -> None:
    from koletivo_trader.domain.copy import phrase_for

    text = phrase_for(Side.HOLD, ChartType.CONSOLIDATION, 0.41, reason="low_hit")
    assert "não comprar nem vender" in text
    assert "41%" in text


def test_first_block_requires_swing_agreement() -> None:
    day = Signal(
        side=Side.BUY,
        entry=1000,
        stop=900,
        take=1200,
        chart_type=ChartType.IMPULSE_UP,
        hit_pct=0.8,
        phrase="x",
        reason="daytrade",
    )
    ctx = SessionContext(
        as_of=datetime(2026, 3, 12, 17, 0),
        previous_date="2026-03-12",
        swing_signal=Side.SELL,
        day_type=DayType.TREND_DOWN,
        predicted_day_type=DayType.TREND_DOWN,
        swing_hit_pct=0.7,
        phrase="queda",
    )
    fused = fuse_signals(day, ctx, swing_weight=0.15, min_hit_pct=0.62, minutes_from_open=5)
    assert fused.side is Side.HOLD
    assert fused.reason == "swing_discord"
    assert fused.chart_type is ChartType.IMPULSE_UP
    assert "nenhuma ordem" in fused.phrase.lower() or "não" in fused.phrase.lower()


def test_near_gain_pulls_stop_into_profit() -> None:
    stop, take, _ = protect_levels(
        buy=True,
        entry=1000,
        stop=900,
        take=1200,
        orig_stop=900,
        mark=1180,
        extreme=1180,
        tick=5,
        be_trigger=25,
        be_lock=10,
        trail_enabled=True,
        trail_trigger=60,
        trail_distance=50,
        orig_take=1200,
        near_gain_points=30,
    )
    assert stop > 1000
    assert stop < 1180
