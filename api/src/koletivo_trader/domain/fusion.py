from __future__ import annotations

from koletivo_trader.domain.copy import phrase_for
from koletivo_trader.domain.enums import Side
from koletivo_trader.domain.fibonacci import clamp_decider_weights
from koletivo_trader.domain.models import SessionContext, Signal


def swing_signed(side: Side, context: SessionContext | None) -> float:
    """+conf if same side, −conf if opposite, 0 if unused / HOLD."""
    if context is None or side is Side.HOLD:
        return 0.0
    if context.swing_signal is Side.HOLD:
        return 0.0
    conf = min(max(float(context.swing_hit_pct), 0.0), 1.0)
    if context.swing_signal is side:
        return conf
    return -conf


def fuse_hit(
    hit_day: float,
    *,
    swing_weight: float,
    swing_signed_score: float,
    fib_weight: float = 0.0,
    fib_signed_score: float = 0.0,
) -> float:
    """Daytrade keeps its full hit. Auxiliaries add/subtract weight × confidence.

    Example: hit 0.96, swing disagrees at 0.80 with weight 0.2
    → 0.96 + 0.2 × (−0.80) = 0.80. Signal stays if still above min_hit.
    """
    _, w_swing, w_fib = clamp_decider_weights(swing_weight, fib_weight)
    fused = float(hit_day) + w_swing * float(swing_signed_score) + w_fib * float(fib_signed_score)
    return min(0.99, max(0.01, fused))


def fuse_signals(
    daytrade: Signal,
    context: SessionContext | None,
    *,
    swing_weight: float,
    min_hit_pct: float,
    minutes_from_open: float = 0.0,
    first_block_minutes: float = 15.0,
    fib_weight: float = 0.0,
    fib_boost: float = 0.0,
) -> Signal:
    del minutes_from_open, first_block_minutes
    hit_day = float(daytrade.hit_pct)
    side = daytrade.side
    reason = daytrade.reason
    predicted = daytrade.predicted_chart_type
    signed_swing = swing_signed(side, context) if swing_weight > 0 else 0.0
    signed_fib = float(fib_boost) if fib_weight > 0 and side in {Side.BUY, Side.SELL} else 0.0
    fused = fuse_hit(
        hit_day,
        swing_weight=swing_weight,
        swing_signed_score=signed_swing,
        fib_weight=fib_weight,
        fib_signed_score=signed_fib,
    )
    if side in {Side.BUY, Side.SELL} and fused < min_hit_pct:
        side = Side.HOLD
        reason = "low_hit"
    elif (
        side in {Side.BUY, Side.SELL}
        and predicted is not None
        and predicted != daytrade.chart_type
    ):
        side = Side.HOLD
        reason = "chart_mismatch"
    elif signed_swing < 0 and side in {Side.BUY, Side.SELL}:
        reason = "swing_drag" if reason == "daytrade" else reason
    phrase = phrase_for(
        side,
        daytrade.chart_type,
        fused,
        reason=reason,
        day_type=None if context is None else context.predicted_day_type,
        swing_signal=None if context is None else context.swing_signal,
        predicted_chart=predicted,
    )
    return Signal(
        side=side,
        entry=daytrade.entry,
        stop=daytrade.stop,
        take=daytrade.take,
        chart_type=daytrade.chart_type,
        hit_pct=fused,
        phrase=phrase,
        reason=reason,
        day_type=None if context is None else context.predicted_day_type,
        swing_signal=None if context is None else context.swing_signal,
        predicted_chart_type=predicted,
    )
