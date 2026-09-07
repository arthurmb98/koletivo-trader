from __future__ import annotations

from koletivo_trader.domain.copy import phrase_for
from koletivo_trader.domain.enums import DayType, Side
from koletivo_trader.domain.market import chart_matches_day
from koletivo_trader.domain.models import SessionContext, Signal


def fuse_signals(
    daytrade: Signal,
    context: SessionContext | None,
    *,
    swing_weight: float,
    min_hit_pct: float,
    minutes_from_open: float,
    first_block_minutes: float = 15.0,
) -> Signal:
    weight = min(0.999, max(0.001, float(swing_weight)))
    hit = float(daytrade.hit_pct)
    reason = daytrade.reason
    side = daytrade.side
    if context is None:
        if hit < min_hit_pct:
            side = Side.HOLD
            reason = "low_hit"
        phrase = phrase_for(side, daytrade.chart_type, hit, reason=reason)
        return Signal(
            side=side,
            entry=daytrade.entry,
            stop=daytrade.stop,
            take=daytrade.take,
            chart_type=daytrade.chart_type,
            hit_pct=hit,
            phrase=phrase,
            reason=reason,
        )

    in_first_block = minutes_from_open < first_block_minutes
    if in_first_block:
        if side in {Side.BUY, Side.SELL} and context.swing_signal in {Side.BUY, Side.SELL}:
            if side is not context.swing_signal:
                side = Side.HOLD
                reason = "swing_discord"
                hit = min(hit, min_hit_pct * 0.9)
        elif context.swing_signal is Side.HOLD and side in {Side.BUY, Side.SELL}:
            hit *= 1.0 - weight
            if hit < min_hit_pct:
                side = Side.HOLD
                reason = "swing_discord"

    match = chart_matches_day(daytrade.chart_type, context.predicted_day_type)
    boost = 0.0
    if match is True:
        boost = 0.5 + 0.5 * context.swing_hit_pct
    elif match is False:
        boost = 0.15
        if not in_first_block:
            reason = "day_mismatch"
    else:
        boost = 0.35
    fused = (1.0 - weight) * hit + weight * boost
    fused = min(0.99, max(0.01, fused))
    if fused < min_hit_pct:
        side = Side.HOLD
        if reason not in {"swing_discord", "day_mismatch"}:
            reason = "low_hit"
    phrase = phrase_for(
        side,
        daytrade.chart_type,
        fused,
        reason=reason,
        day_type=context.predicted_day_type,
        swing_signal=context.swing_signal,
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
        day_type=context.predicted_day_type,
        swing_signal=context.swing_signal,
    )
