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


def test_gold_hours_run_until_17() -> None:
    from koletivo_trader.domain.session import SessionFilter

    flt = SessionFilter()
    day = datetime(2026, 3, 13)
    assert flt.allows(day.replace(hour=16, minute=5))
    assert flt.allows(day.replace(hour=17, minute=0))
    assert not flt.allows(day.replace(hour=17, minute=5))


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


def test_swing_discord_scales_by_weight_not_veto() -> None:
    day = Signal(
        side=Side.BUY,
        entry=1000,
        stop=900,
        take=1200,
        chart_type=ChartType.IMPULSE_UP,
        hit_pct=0.96,
        phrase="x",
        reason="daytrade",
    )
    ctx = SessionContext(
        as_of=datetime(2026, 3, 12, 17, 0),
        previous_date="2026-03-12",
        swing_signal=Side.SELL,
        day_type=DayType.TREND_DOWN,
        predicted_day_type=DayType.TREND_DOWN,
        swing_hit_pct=0.80,
        phrase="queda",
    )
    fused = fuse_signals(day, ctx, swing_weight=0.20, min_hit_pct=0.62, minutes_from_open=5)
    assert fused.side is Side.BUY
    assert abs(fused.hit_pct - 0.80) < 1e-9


def test_first_block_requires_swing_agreement() -> None:
    """Legacy name: disagreement no longer vetoes; it only drags hit by weight."""
    test_swing_discord_scales_by_weight_not_veto()


def test_swing_weight_zero_leaves_daytrade_hit() -> None:
    day = Signal(
        side=Side.BUY,
        entry=1000,
        stop=900,
        take=1200,
        chart_type=ChartType.IMPULSE_UP,
        hit_pct=0.8,
        phrase="x",
        reason="daytrade",
        predicted_chart_type=ChartType.BREAKOUT,
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
    fused = fuse_signals(day, ctx, swing_weight=0.0, min_hit_pct=0.62, minutes_from_open=5)
    assert fused.side is Side.BUY
    assert abs(fused.hit_pct - 0.8) < 1e-9
    assert fused.predicted_chart_type is ChartType.BREAKOUT


def test_clamp_decider_weights_caps_aux_at_40pct() -> None:
    from koletivo_trader.domain.fibonacci import clamp_decider_weights

    day, swing, fib = clamp_decider_weights(0.3, 0.3)
    assert abs(day + swing + fib - 1.0) < 1e-9
    assert swing + fib <= 0.4 + 1e-9
    assert day >= 0.6 - 1e-9
    day0, swing0, fib0 = clamp_decider_weights(0.0, 0.0)
    assert swing0 == 0.0 and fib0 == 0.0 and abs(day0 - 1.0) < 1e-9


def test_fib_boost_higher_near_support_in_uptrend() -> None:
    from koletivo_trader.domain.fibonacci import fib_boost

    start = datetime(2026, 3, 13, 9, 15)
    candles = [_bar(i, 1000 + i * 10, 1008 + i * 10, 998 + i * 10, 1006 + i * 10, start) for i in range(12)]
    lo = min(c.low for c in candles)
    hi = max(c.high for c in candles)
    mid = hi - 0.5 * (hi - lo)
    near = fib_boost(Side.BUY, mid, candles, tick=5)
    far = fib_boost(Side.BUY, hi + 200, candles, tick=5)
    assert near > far
    assert fib_boost(Side.HOLD, mid, candles) == 0.0


def test_fib_discord_scales_by_weight() -> None:
    day = Signal(
        side=Side.BUY,
        entry=1000,
        stop=900,
        take=1200,
        chart_type=ChartType.IMPULSE_UP,
        hit_pct=0.96,
        phrase="x",
        reason="daytrade",
    )
    fused = fuse_signals(
        day,
        None,
        swing_weight=0.0,
        min_hit_pct=0.62,
        minutes_from_open=30,
        fib_weight=0.2,
        fib_boost=-0.80,
    )
    assert fused.side is Side.BUY
    assert abs(fused.hit_pct - 0.80) < 1e-9


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


def test_genetic_repair_keeps_daytrade_majority_and_rr() -> None:
    from koletivo_trader.ml.genetics import HI, LO, repair_genome, unpack

    assert LO[2] == 0.17
    assert HI[2] == 0.83
    g = repair_genome([100.0, 120.0, 0.5, 0.3, 0.3, 7.0])
    p = unpack(g)
    assert p["swing_weight"] + p["fib_weight"] <= 0.4 + 1e-9
    assert p["gain"] >= p["stop"] * 1.5 - 1e-9
    assert p["offset_points"] % 5 == 0
    assert unpack(repair_genome([100.0, 200.0, 0.05, 0.0, 0.0, 0.0]))["min_hit"] == 0.17
    assert unpack(repair_genome([100.0, 200.0, 0.99, 0.0, 0.0, 0.0]))["min_hit"] == 0.83
