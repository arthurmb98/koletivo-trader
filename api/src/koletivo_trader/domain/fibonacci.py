from __future__ import annotations

from koletivo_trader.domain.enums import Side
from koletivo_trader.domain.models import Candle

FIB_RATIOS = (0.382, 0.5, 0.618)
AUX_WEIGHT_CAP = 0.4


def clamp_decider_weights(swing_weight: float, fib_weight: float) -> tuple[float, float, float]:
    """Daytrade stays >= 0.6. Swing and Fib may be 0. Sum of the two <= 0.4."""
    swing = min(max(float(swing_weight), 0.0), AUX_WEIGHT_CAP)
    fib = min(max(float(fib_weight), 0.0), AUX_WEIGHT_CAP)
    extra = swing + fib
    if extra > AUX_WEIGHT_CAP:
        scale = AUX_WEIGHT_CAP / extra
        swing *= scale
        fib *= scale
    day = 1.0 - swing - fib
    return day, swing, fib


def impulse_bounds(candles: list[Candle]) -> tuple[float, float] | None:
    if len(candles) < 4:
        return None
    lo = min(c.low for c in candles)
    hi = max(c.high for c in candles)
    if hi - lo < 1e-9:
        return None
    return lo, hi


def fib_retracement_levels(lo: float, hi: float, *, uptrend: bool) -> list[float]:
    rng = hi - lo
    if uptrend:
        return [hi - ratio * rng for ratio in FIB_RATIOS]
    return [lo + ratio * rng for ratio in FIB_RATIOS]


def fib_boost(
    side: Side,
    price: float,
    candles: list[Candle],
    *,
    tick: float = 5.0,
) -> float:
    """Signed confluence in [-1, 1]: +near Fib with the trade, −near Fib against it."""
    if side is Side.HOLD or not candles:
        return 0.0
    bounds = impulse_bounds(candles)
    if bounds is None:
        return 0.0
    lo, hi = bounds
    uptrend = candles[-1].close >= (lo + hi) / 2.0
    levels = fib_retracement_levels(lo, hi, uptrend=uptrend)
    band = max(tick * 2.0, (hi - lo) * 0.04)
    near = min(abs(price - lvl) for lvl in levels)
    closeness = max(0.0, 1.0 - near / (band * 3))
    aligned = (side is Side.BUY and uptrend) or (side is Side.SELL and not uptrend)
    return float(closeness if aligned else -closeness)
