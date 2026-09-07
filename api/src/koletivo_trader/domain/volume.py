from __future__ import annotations

import numpy as np

from koletivo_trader.domain.models import Candle


def volume_pattern_features(candles: list[Candle]) -> np.ndarray:
    """Bar-volume signatures: confirmation, exhaustion, breakout, fakeout, absorption, A/D, liquidity."""
    n = 10
    out = np.zeros(n, dtype=float)
    if len(candles) < 5:
        return out
    vols = np.array([max(c.volume, 0.0) for c in candles], dtype=float)
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    rng = np.maximum(highs - lows, 1e-9)
    mean20 = float(vols[-20:].mean()) if len(vols) >= 5 else float(vols.mean())
    last_v = float(vols[-1])
    rvol = last_v / max(mean20, 1e-9)
    out[0] = min(rvol / 3.0, 2.0)
    px_up = closes[-1] >= closes[-5]
    vol_up = last_v >= mean20
    out[1] = 1.0 if px_up and vol_up else 0.0
    out[2] = 1.0 if px_up and not vol_up else 0.0
    broke_high = highs[-1] > highs[-6:-1].max() if len(highs) >= 6 else False
    broke_low = lows[-1] < lows[-6:-1].min() if len(lows) >= 6 else False
    broke = broke_high or broke_low
    out[3] = 1.0 if broke and rvol >= 1.5 else 0.0
    out[4] = 1.0 if broke and rvol < 1.0 else 0.0
    out[5] = 1.0 if rvol >= 1.5 and rng[-1] <= np.median(rng) * 0.7 else 0.0
    up_vol = float(vols[1:][closes[1:] >= closes[:-1]].sum()) if len(closes) > 1 else 0.0
    down_vol = float(vols[1:][closes[1:] < closes[:-1]].sum()) if len(closes) > 1 else 0.0
    tot = up_vol + down_vol
    out[6] = (up_vol - down_vol) / tot if tot else 0.0
    std = float(vols.std())
    out[7] = 0.0 if std < 1e-9 else float((last_v - vols.mean()) / std)
    out[8] = float((closes[-1] - closes[0]) / max(float(highs.max() - lows.min()), 1e-9))
    out[9] = min(last_v / max(float(np.median(vols)), 1e-9) / 4.0, 2.0)
    return out
