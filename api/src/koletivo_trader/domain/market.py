from __future__ import annotations

import numpy as np

from koletivo_trader.domain.enums import ChartType, DayType, Side
from koletivo_trader.domain.models import Candle

_BULLISH_CHARTS = {ChartType.IMPULSE_UP, ChartType.PULLBACK_UP}
_BEARISH_CHARTS = {ChartType.IMPULSE_DOWN, ChartType.PULLBACK_DOWN}
_BULLISH_DAYS = {DayType.TREND_UP, DayType.NORMAL_VARIATION}
_BEARISH_DAYS = {DayType.TREND_DOWN}


def _closes(candles: list[Candle]) -> np.ndarray:
    return np.array([c.close for c in candles], dtype=float)


def _linreg(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(slope), float(r2)


def classify_day(candles: list[Candle]) -> DayType:
    """Auction-market style day type from a full session of OHLC bars."""
    if len(candles) < 6:
        return DayType.NON_TREND
    opens = np.array([c.open for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    closes = _closes(candles)
    session_high = float(highs.max())
    session_low = float(lows.min())
    rng = max(session_high - session_low, 1e-9)
    open_px = float(opens[0])
    close_px = float(closes[-1])
    close_loc = (close_px - session_low) / rng
    body = abs(close_px - open_px)
    bar_ranges = highs - lows
    mean_bar = float(np.mean(bar_ranges)) or 1.0
    expansion = rng / max(mean_bar, 1e-9)
    first_third = candles[: max(len(candles) // 3, 1)]
    ib_high = max(c.high for c in first_third)
    ib_low = min(c.low for c in first_third)
    ib_range = max(ib_high - ib_low, 1e-9)
    extension = rng / ib_range
    slope, r2 = _linreg(closes)
    slope_norm = slope * len(closes) / rng

    if expansion >= 8.0 and r2 < 0.25:
        return DayType.VOLATILE
    if rng / mean_bar < 3.2 and abs(slope_norm) < 0.35:
        return DayType.NON_TREND
    if r2 >= 0.55 and slope_norm > 0.55 and close_loc >= 0.72:
        return DayType.TREND_UP
    if r2 >= 0.55 and slope_norm < -0.55 and close_loc <= 0.28:
        return DayType.TREND_DOWN
    if close_loc > 0.38 and close_loc < 0.62 and body < 0.35 * rng:
        return DayType.NEUTRAL
    if extension >= 2.2 and (close_loc >= 0.7 or close_loc <= 0.3):
        return DayType.NORMAL_VARIATION
    return DayType.NORMAL


def classify_chart(candles: list[Candle]) -> ChartType:
    """Micro structure of a short window (3 M5)."""
    if len(candles) < 3:
        return ChartType.INDECISION
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    closes = _closes(candles)
    rng = max(float(highs.max() - lows.min()), 1e-9)
    slope, r2 = _linreg(closes)
    slope_norm = slope * len(closes) / rng
    first = candles[: len(candles) // 2]
    last = candles[len(candles) // 2 :]
    first_slope, _ = _linreg(_closes(first))
    last_slope, last_r2 = _linreg(_closes(last))
    last_high = max(c.high for c in last)
    last_low = min(c.low for c in last)
    prior_high = max(c.high for c in first)
    prior_low = min(c.low for c in first)
    bodies = np.abs(np.array([c.body for c in candles], dtype=float))
    mean_range = float(np.mean(highs - lows)) or 1.0
    compact = rng < 1.6 * mean_range * 3

    if last_high > prior_high * 1.0000 and last_high >= prior_high + 0.15 * rng and slope_norm > 0.25:
        if r2 < 0.35:
            return ChartType.BREAKOUT
    if last_low < prior_low and slope_norm < -0.25 and r2 < 0.35:
        return ChartType.BREAKOUT
    if compact and abs(slope_norm) < 0.28:
        return ChartType.CONSOLIDATION
    if first_slope > 0 and last_slope < 0 and r2 < 0.45:
        return ChartType.REVERSAL
    if first_slope < 0 and last_slope > 0 and r2 < 0.45:
        return ChartType.REVERSAL
    if r2 >= 0.55 and slope_norm >= 0.45:
        return ChartType.IMPULSE_UP
    if r2 >= 0.55 and slope_norm <= -0.45:
        return ChartType.IMPULSE_DOWN
    if slope_norm > 0.2 and last_r2 < 0.4 and last_slope < first_slope:
        return ChartType.PULLBACK_UP
    if slope_norm < -0.2 and last_r2 < 0.4 and last_slope > first_slope:
        return ChartType.PULLBACK_DOWN
    if float(np.mean(bodies)) < 0.25 * mean_range:
        return ChartType.INDECISION
    if slope_norm > 0.15:
        return ChartType.PULLBACK_UP
    if slope_norm < -0.15:
        return ChartType.PULLBACK_DOWN
    return ChartType.INDECISION


def chart_matches_day(chart: ChartType, day: DayType) -> bool | None:
    """True = confirma, False = discorda, None = neutro."""
    if day is DayType.TREND_UP and chart in _BULLISH_CHARTS | {ChartType.BREAKOUT}:
        return True
    if day is DayType.TREND_DOWN and chart in _BEARISH_CHARTS | {ChartType.BREAKOUT}:
        return True
    if day in {DayType.NON_TREND, DayType.NEUTRAL} and chart in {
        ChartType.CONSOLIDATION,
        ChartType.INDECISION,
    }:
        return True
    if day is DayType.VOLATILE and chart in {ChartType.REVERSAL, ChartType.INDECISION, ChartType.BREAKOUT}:
        return True
    if day is DayType.TREND_UP and chart in _BEARISH_CHARTS:
        return False
    if day is DayType.TREND_DOWN and chart in _BULLISH_CHARTS:
        return False
    if day in {DayType.NON_TREND, DayType.NEUTRAL} and chart in {
        ChartType.IMPULSE_UP,
        ChartType.IMPULSE_DOWN,
        ChartType.BREAKOUT,
    }:
        return False
    return None


def heuristic_side(chart: ChartType, slope_norm: float) -> Side:
    if chart in {ChartType.CONSOLIDATION, ChartType.INDECISION}:
        return Side.HOLD
    if chart is ChartType.IMPULSE_UP or (chart is ChartType.PULLBACK_UP and slope_norm > 0):
        return Side.BUY
    if chart is ChartType.IMPULSE_DOWN or (chart is ChartType.PULLBACK_DOWN and slope_norm < 0):
        return Side.SELL
    if chart is ChartType.BREAKOUT:
        return Side.BUY if slope_norm >= 0 else Side.SELL
    if chart is ChartType.REVERSAL:
        return Side.SELL if slope_norm > 0 else Side.BUY
    return Side.HOLD
