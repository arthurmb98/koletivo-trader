from __future__ import annotations

from koletivo_trader.application.replay import replay_meta
from koletivo_trader.domain.product import BANKS, CASE, HORIZON_M5, LOOKBACK_M1, TIMEFRAME
from koletivo_trader.domain.risk import contracts_for_bank
from koletivo_trader.ml.parameters import search_parameters


def test_product_locks_m5_last_candles_and_four_banks() -> None:
    assert CASE == "last_candles"
    assert TIMEFRAME == "m5"
    assert BANKS == (500.0, 1000.0, 5000.0, 10000.0)
    assert LOOKBACK_M1 == 15
    assert HORIZON_M5 == 3
    assert contracts_for_bank(10000) == 10


def test_replay_meta_has_no_m1_or_last_candle() -> None:
    meta = replay_meta("m1")
    assert meta["timeframe"] == "m5"
    assert meta["timeframes"] == [{"key": "m5", "label": "5 min"}]
    assert [item["key"] for item in meta["cases"]] == ["last_candles"]
    assert meta["banks"] == [500, 1000, 5000, 10000]


def test_search_parameters_default_includes_10000() -> None:
    assert search_parameters.__defaults__ is not None
    banks = search_parameters.__defaults__[0]
    assert banks == BANKS
    assert 10000.0 in banks
