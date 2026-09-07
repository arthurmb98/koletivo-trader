from __future__ import annotations

from datetime import datetime, timedelta

from koletivo_trader.domain.enums import Side, TradeResult
from koletivo_trader.domain.models import Candle
from koletivo_trader.ml.labels import label_side, leak_free_windows, simulate_touch


def _c(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle("WIN$", ts, o, h, l, c, 1)


def test_stop_wins_if_same_bar_touches_both() -> None:
    start = datetime(2026, 3, 13, 10, 20)
    future = [_c(start, 1000, 1300, 800, 1100)]
    assert simulate_touch(Side.BUY, 1000, 100, 200, future) is TradeResult.STOP


def test_first_touch_picks_the_first_barrier() -> None:
    from koletivo_trader.ml.labels import first_touch_side

    start = datetime(2026, 3, 13, 10, 20)
    future = [
        _c(start, 1000, 1020, 990, 1010),
        _c(start + timedelta(minutes=5), 1010, 1040, 1005, 1035),
    ]
    assert first_touch_side(1000, 30, future) is Side.BUY


def test_label_hold_when_neither_side_gains() -> None:
    start = datetime(2026, 3, 13, 10, 20)
    future = [
        _c(start, 1000, 1010, 990, 1005),
        _c(start + timedelta(minutes=5), 1005, 1012, 995, 1000),
        _c(start + timedelta(minutes=10), 1000, 1008, 992, 997),
    ]
    assert label_side(1000, 100, 200, future) is Side.HOLD


def test_windows_do_not_put_future_m5_into_x() -> None:
    t0 = datetime(2026, 3, 13, 9, 0)
    m1 = [_c(t0 + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i) for i in range(40)]
    m5 = [_c(t0 + timedelta(minutes=i * 5), 100, 110, 90, 105) for i in range(8)]
    windows = leak_free_windows(m1, m5, lookback=15, horizon=3)
    assert windows
    window, future, entry = windows[0]
    assert window[-1].timestamp < future[0].timestamp
    assert entry.timestamp == future[0].timestamp
    assert all(c.timestamp not in {b.timestamp for b in future} for c in window)
