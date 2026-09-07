from __future__ import annotations

from datetime import datetime, timedelta

from koletivo_trader.domain.enums import Side, TradeResult
from koletivo_trader.domain.models import Candle


def simulate_touch(
    side: Side,
    entry: float,
    stop_points: float,
    gain_points: float,
    future: list[Candle],
) -> TradeResult:
    """Walk future bars. If stop and gain share a bar, stop wins (conservative)."""
    result, _ = simulate_touch_at(side, entry, stop_points, gain_points, future)
    return result


def simulate_touch_at(
    side: Side,
    entry: float,
    stop_points: float,
    gain_points: float,
    future: list[Candle],
) -> tuple[TradeResult, datetime | None]:
    if side is Side.HOLD or not future:
        return TradeResult.NONE, None
    if side is Side.BUY:
        stop = entry - stop_points
        take = entry + gain_points
        for bar in future:
            hit_stop = bar.low <= stop
            hit_gain = bar.high >= take
            if hit_stop:
                return TradeResult.STOP, bar.timestamp
            if hit_gain:
                return TradeResult.GAIN, bar.timestamp
        return TradeResult.NONE, None
    stop = entry + stop_points
    take = entry - gain_points
    for bar in future:
        hit_stop = bar.high >= stop
        hit_gain = bar.low <= take
        if hit_stop:
            return TradeResult.STOP, bar.timestamp
        if hit_gain:
            return TradeResult.GAIN, bar.timestamp
    return TradeResult.NONE, None


def label_side(
    entry: float,
    stop_points: float,
    gain_points: float,
    future: list[Candle],
) -> Side:
    buy = simulate_touch(Side.BUY, entry, stop_points, gain_points, future)
    sell = simulate_touch(Side.SELL, entry, stop_points, gain_points, future)
    if buy is TradeResult.GAIN and sell is not TradeResult.GAIN:
        return Side.BUY
    if sell is TradeResult.GAIN and buy is not TradeResult.GAIN:
        return Side.SELL
    return Side.HOLD


def first_touch_side(entry: float, delta: float, future: list[Candle]) -> Side:
    """Which way price travels `delta` points first. Same-bar both sides → HOLD."""
    if delta <= 0 or not future:
        return Side.HOLD
    for bar in future:
        up = bar.high >= entry + delta
        down = bar.low <= entry - delta
        if up and down:
            return Side.HOLD
        if up:
            return Side.BUY
        if down:
            return Side.SELL
    return Side.HOLD


def atr_points(window: list[Candle]) -> float:
    if not window:
        return 50.0
    ranges = [max(c.range, 1.0) for c in window]
    return float(sum(ranges) / len(ranges))


def adaptive_barriers(
    window: list[Candle],
    *,
    stop_mult: float = 1.2,
    gain_mult: float = 2.0,
    stop_floor: float = 40.0,
    stop_cap: float = 120.0,
    gain_cap: float = 240.0,
) -> tuple[float, float]:
    """ATR of 1-min window, snapped to 5-point ticks, floor for WIN mini ops."""
    atr = atr_points(window)
    stop = max(stop_floor, round(atr * stop_mult / 5.0) * 5.0)
    gain = max(stop * 1.5, round(atr * gain_mult / 5.0) * 5.0)
    stop = min(stop, stop_cap)
    gain = min(gain, gain_cap)
    return stop, gain


def direction_delta(window: list[Candle], *, mult: float = 0.7, floor: float = 25.0, cap: float = 80.0) -> float:
    atr = atr_points(window)
    return float(min(cap, max(floor, round(atr * mult / 5.0) * 5.0)))


def same_day_m1(
    m1: list[Candle],
    start: datetime,
    *,
    index: dict | None = None,
    n: int = 180,
) -> list[Candle]:
    """M1 path from `start` inclusive, same civil day, capped at n bars. Train/backtest Y only."""
    idx_map = index if index is not None else {c.timestamp: i for i, c in enumerate(m1)}
    i = idx_map.get(start)
    if i is None:
        return []
    day = start.date()
    out: list[Candle] = []
    for candle in m1[i : i + n]:
        if candle.timestamp.date() != day:
            break
        out.append(candle)
    return out


def leak_free_windows(
    m1: list[Candle],
    m5: list[Candle],
    *,
    lookback: int = 15,
    horizon: int = 3,
) -> list[tuple[list[Candle], list[Candle], Candle]]:
    """Each sample: 15 closed M1 ending at T, 3 future M5 starting at T. No overlap into X."""
    if not m1 or not m5:
        return []
    m1_index = {c.timestamp: i for i, c in enumerate(m1)}
    out: list[tuple[list[Candle], list[Candle], Candle]] = []
    for i, bar in enumerate(m5[:-horizon]):
        last_m1_ts = bar.timestamp + timedelta(minutes=4)
        idx = m1_index.get(last_m1_ts)
        if idx is None or idx + 1 < lookback:
            continue
        future = m5[i + 1 : i + 1 + horizon]
        if len(future) < horizon or future[0].timestamp.date() != bar.timestamp.date():
            continue
        window = m1[idx + 1 - lookback : idx + 1]
        if len(window) != lookback:
            continue
        if window[-1].timestamp >= future[0].timestamp:
            continue
        out.append((window, future, future[0]))
    return out


def prior_m5_bars(
    m5: list[Candle],
    entry: Candle,
    n: int = 6,
    *,
    index: dict | None = None,
) -> list[Candle]:
    """Closed M5 bars strictly before the entry bar (no leakage)."""
    idx_map = index if index is not None else {c.timestamp: i for i, c in enumerate(m5)}
    i = idx_map.get(entry.timestamp)
    if i is None:
        return []
    out: list[Candle] = []
    j = i - 1
    day = entry.timestamp.date()
    while j >= 0 and len(out) < n:
        if m5[j].timestamp.date() != day:
            break
        out.append(m5[j])
        j -= 1
    out.reverse()
    return out
