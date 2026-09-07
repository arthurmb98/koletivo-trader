from __future__ import annotations

from koletivo_trader.domain.enums import Side


def round_to_tick(points: float, tick: float) -> float:
    if tick <= 0:
        return points
    return round(points / tick) * tick


def contracts_for_bank(bank: float, *, step: float = 1000.0, cap: int = 10) -> int:
    """500→1, 1000→1, +1 mini a cada R$ 1000, teto 10 (banca 10k)."""
    if bank < step:
        return 1
    return min(cap, max(1, int(bank // step)))


class RiskCalculator:
    def __init__(self, stop_points: float, gain_points: float, tick_size: float) -> None:
        self.stop_points = float(stop_points)
        self.gain_points = float(gain_points)
        self.tick = float(tick_size)

    def distances(self) -> tuple[float, float]:
        return round_to_tick(self.stop_points, self.tick), round_to_tick(self.gain_points, self.tick)

    def levels(self, side: Side, entry: float) -> tuple[float, float]:
        stop_pts, gain_pts = self.distances()
        if side is Side.BUY:
            return entry - stop_pts, entry + gain_pts
        if side is Side.SELL:
            return entry + stop_pts, entry - gain_pts
        return 0.0, 0.0


def protect_levels(
    *,
    buy: bool,
    entry: float,
    stop: float,
    take: float,
    orig_stop: float,
    mark: float | None,
    extreme: float,
    tick: float,
    be_trigger: float,
    be_lock: float,
    trail_enabled: bool,
    trail_trigger: float,
    trail_distance: float,
    orig_take: float,
    near_gain_points: float = 30.0,
) -> tuple[float, float, float]:
    new_stop = stop
    new_take = take
    new_extreme = extreme
    if mark is None:
        return new_stop, new_take, new_extreme
    if buy:
        new_extreme = max(extreme, mark)
        fav = mark - entry
    else:
        new_extreme = min(extreme, mark)
        fav = entry - mark
    if be_trigger > 0 and fav >= be_trigger:
        lock = round_to_tick(entry + be_lock if buy else entry - be_lock, tick)
        new_stop = max(new_stop, lock) if buy else min(new_stop, lock)
    if trail_enabled:
        if buy and new_extreme - entry >= trail_trigger:
            new_stop = max(new_stop, round_to_tick(new_extreme - trail_distance, tick))
        elif (not buy) and entry - new_extreme >= trail_trigger:
            new_stop = min(new_stop, round_to_tick(new_extreme + trail_distance, tick))
    # Perto do gain: trava o stop no lado positivo para um pullback ainda ser lucro.
    if near_gain_points > 0:
        to_take = (orig_take - mark) if buy else (mark - orig_take)
        if 0 < to_take <= near_gain_points:
            lock = round_to_tick(mark - tick * 2 if buy else mark + tick * 2, tick)
            if buy:
                lock = max(lock, round_to_tick(entry + tick, tick))
                new_stop = max(new_stop, lock)
            else:
                lock = min(lock, round_to_tick(entry - tick, tick))
                new_stop = min(new_stop, lock)
    if buy:
        new_stop = max(new_stop, orig_stop)
        new_take = min(new_take, orig_take)
        new_stop = min(new_stop, round_to_tick(mark - tick, tick))
        new_stop = max(new_stop, orig_stop)
        if new_take <= new_stop:
            new_take = round_to_tick(new_stop + tick, tick)
    else:
        new_stop = min(new_stop, orig_stop)
        new_take = max(new_take, orig_take)
        new_stop = max(new_stop, round_to_tick(mark + tick, tick))
        new_stop = min(new_stop, orig_stop)
        if new_take >= new_stop:
            new_take = round_to_tick(new_stop - tick, tick)
    return new_stop, new_take, new_extreme
