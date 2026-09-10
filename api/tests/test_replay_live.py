from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from koletivo_trader.adapters.config import load_bank_config, load_named_config
from koletivo_trader.application import replay as replay_mod
from koletivo_trader.application.replay import replay_meta
from koletivo_trader.domain.models import Candle
from koletivo_trader.ml.labels import leak_free_windows, live_features_for_closed_m5, prior_m5_bars


def _c(ts: datetime, o: float, h: float, l: float, c: float) -> Candle:
    return Candle("WIN$", ts, o, h, l, c, 1)


def test_bank_config_uses_study_yaml_not_seed() -> None:
    seed = load_named_config("best_candles_m5_1000_a")
    bank = load_bank_config(1000)
    assert bank.name == "best_bank_1000"
    assert bank.risk.gain_points > bank.risk.stop_points
    assert seed.risk.stop_points == 100.0
    assert (
        bank.risk.stop_points != seed.risk.stop_points
        or bank.execution.offset_points != seed.execution.offset_points
        or bank.filters.min_hit_pct != seed.filters.min_hit_pct
    )
    assert load_bank_config(1200).name == "best_bank_1000"


def test_replay_meta_defaults_to_study_contracts() -> None:
    meta = replay_meta()
    assert meta["lot"] == "scaled"
    assert [item["key"] for item in meta["lots"]] == ["fixed", "scaled"]


def test_replay_module_uses_study_backtest() -> None:
    text = Path(replay_mod.__file__).read_text(encoding="utf-8")
    assert "score_bank_window" in text
    assert "load_bank_config" in text
    assert "simulate_touch(" not in text.replace("simulate_touch_at", "")


def test_live_m5_window_matches_leak_free_windows() -> None:
    t0 = datetime(2026, 3, 13, 9, 0)
    m1 = [_c(t0 + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i) for i in range(80)]
    m5 = [_c(t0 + timedelta(minutes=i * 5), 100, 110, 90, 105) for i in range(12)]
    windows = leak_free_windows(m1, m5, lookback=3, horizon=3)
    assert windows
    window, _future, entry = windows[3]
    closed = window[-1]
    live_window, live_prior = live_features_for_closed_m5(m5, closed, lookback=3)
    assert [c.timestamp for c in live_window] == [c.timestamp for c in window]
    prior = prior_m5_bars(m5, window[0])
    assert [c.timestamp for c in live_prior] == [c.timestamp for c in prior]
    assert live_prior[-1].timestamp < window[0].timestamp
    assert entry.timestamp == _future[0].timestamp
