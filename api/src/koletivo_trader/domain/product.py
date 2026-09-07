from __future__ import annotations

CASE = "last_candles"
TIMEFRAME = "m5"
BANKS = (500.0, 1000.0, 5000.0, 10000.0)
LOOKBACK_M1 = 15
HORIZON_M5 = 3


def bank_keys() -> tuple[str, ...]:
    return tuple(str(int(bank)) for bank in BANKS)
