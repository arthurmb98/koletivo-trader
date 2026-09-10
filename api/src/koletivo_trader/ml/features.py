from __future__ import annotations

import numpy as np

from koletivo_trader.domain.enums import ChartType, DayType
from koletivo_trader.domain.market import classify_chart, classify_day
from koletivo_trader.domain.models import Candle
from koletivo_trader.domain.volume import volume_pattern_features

DAY_INDEX = {item: i for i, item in enumerate(DayType)}
CHART_INDEX = {item: i for i, item in enumerate(ChartType)}
PRIOR_M5_DIM = 12
STRUCT_DIM = 27
N_BLOCKS = 3
BLOCK_M1 = 5
BLOCK_OHLC_VOL_DIM = N_BLOCKS * (BLOCK_M1 * 4 + BLOCK_M1)
M5_BARS = 3
M5_OHLC_VOL_DIM = M5_BARS * 5
VOL_SIG_DIM = 10


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


def atr_proxy(candles: list[Candle]) -> float:
    if not candles:
        return 1.0
    ranges = np.array([max(c.range, 1e-9) for c in candles], dtype=float)
    return float(np.mean(ranges))


def window_vector(candles: list[Candle], expected: int = 15) -> np.ndarray:
    if not candles:
        return np.zeros(expected * 4, dtype=float)
    bars = list(candles[-expected:])
    while len(bars) < expected:
        bars.insert(0, bars[0])
    last = bars[-1].close
    scale = max(atr_proxy(bars), 1e-9)
    vals: list[float] = []
    for bar in bars:
        vals.extend(
            [
                (bar.open - last) / scale,
                (bar.high - last) / scale,
                (bar.low - last) / scale,
                (bar.close - last) / scale,
            ]
        )
    return np.asarray(vals, dtype=float)


def m5_bar_vectors(candles: list[Candle], n: int = M5_BARS) -> np.ndarray:
    """3 M5 bars: ATR-normalized OHLC + volume ratio per bar."""
    dim = n * 5
    if not candles:
        return np.zeros(dim, dtype=float)
    bars = list(candles[-n:])
    while len(bars) < n:
        bars.insert(0, bars[0])
    last = bars[-1].close
    scale = max(atr_proxy(bars), 1e-9)
    vol_scale = max(float(np.mean([max(c.volume, 0.0) for c in bars])), 1e-9)
    vals: list[float] = []
    for bar in bars:
        vals.extend(
            [
                (bar.open - last) / scale,
                (bar.high - last) / scale,
                (bar.low - last) / scale,
                (bar.close - last) / scale,
                max(bar.volume, 0.0) / vol_scale,
            ]
        )
    return np.asarray(vals, dtype=float)


def m1_block_vectors(candles: list[Candle], n_blocks: int = N_BLOCKS, block: int = BLOCK_M1) -> np.ndarray:
    """3 blocks of 5 M1: 20 ATR-normalized OHLC + 5 volume ratios per block."""
    expected = n_blocks * block
    dim = n_blocks * (block * 4 + block)
    if not candles:
        return np.zeros(dim, dtype=float)
    bars = list(candles[-expected:])
    while len(bars) < expected:
        bars.insert(0, bars[0])
    last = bars[-1].close
    scale = max(atr_proxy(bars), 1e-9)
    vol_scale = max(float(np.mean([max(c.volume, 0.0) for c in bars])), 1e-9)
    vals: list[float] = []
    for b in range(n_blocks):
        chunk = bars[b * block : (b + 1) * block]
        for bar in chunk:
            vals.extend(
                [
                    (bar.open - last) / scale,
                    (bar.high - last) / scale,
                    (bar.low - last) / scale,
                    (bar.close - last) / scale,
                ]
            )
        for bar in chunk:
            vals.append(max(bar.volume, 0.0) / vol_scale)
    return np.asarray(vals, dtype=float)


def _rsi_like(closes: np.ndarray, n: int) -> float:
    if len(closes) < n + 1:
        n = max(len(closes) - 1, 1)
    d = np.diff(closes[-(n + 1) :])
    gain = float(np.clip(d, 0, None).mean()) if len(d) else 0.0
    loss = float(np.clip(-d, 0, None).mean()) if len(d) else 0.0
    return gain / (gain + loss + 1e-9)


def _streak(closes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.0
    s = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if s < 0:
                break
            s += 1
        elif closes[i] < closes[i - 1]:
            if s > 0:
                break
            s -= 1
        else:
            break
    return s / 10.0


def _structure_features(candles: list[Candle]) -> np.ndarray:
    closes = np.array([c.close for c in candles], dtype=float)
    opens = np.array([c.open for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    vols = np.array([c.volume for c in candles], dtype=float)
    scale = max(atr_proxy(candles), 1e-9)
    rets = np.diff(closes, prepend=closes[0]) / scale
    rng = np.maximum(highs - lows, 1e-9)
    body = closes - opens
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows
    typical = (highs + lows + closes) / 3.0
    vol_safe = np.maximum(vols, 1.0)
    vwap = float(np.sum(typical * vol_safe) / np.sum(vol_safe))

    def _ret(n: int) -> float:
        if len(closes) <= n:
            return 0.0
        return float((closes[-1] - closes[-1 - n]) / scale)

    last5 = rng[-5:].mean() if len(rng) >= 5 else rng.mean()
    first5 = rng[:5].mean() if len(rng) >= 10 else rng.mean()
    vol_z = 0.0 if vols.std() < 1e-9 else float((vols[-1] - vols.mean()) / vols.std())
    up_frac = float((body > 0).mean())
    clv = float((closes[-1] - lows[-1]) / rng[-1] * 2 - 1)
    ts = candles[-1].timestamp
    hour = ts.hour + ts.minute / 60.0
    morning = 1.0 if 9.25 <= hour <= 11.0 else 0.0
    afternoon = 1.0 if 14.5 <= hour <= 17.0 else 0.0
    slope, r2 = _linreg(closes)
    break_high = 1.0 if len(highs) >= 6 and highs[-1] > highs[-6:-1].max() else 0.0
    break_low = 1.0 if len(lows) >= 6 and lows[-1] < lows[-6:-1].min() else 0.0
    return np.array(
        [
            _ret(1),
            _ret(3),
            _ret(5),
            _ret(10) if len(closes) > 10 else _ret(5),
            float(rets[-3:].std() if len(rets) >= 3 else 0.0),
            float(last5 / scale),
            float(rng.mean() / scale),
            float(body[-1] / scale),
            float(upper[-3:].mean() / scale),
            float(lower[-3:].mean() / scale),
            clv,
            up_frac,
            vol_z,
            slope * len(closes) / scale,
            r2,
            hour / 24.0,
            float(ts.weekday()) / 6.0,
            morning,
            afternoon,
            _rsi_like(closes, 5),
            _rsi_like(closes, 10),
            _streak(closes),
            float((closes[-1] - vwap) / scale),
            float(last5 / max(first5, 1e-9)),
            break_high,
            break_low,
            float(np.sum(np.abs(body)) / max(float(np.sum(rng)), 1e-9)),
        ],
        dtype=float,
    )


def prior_m5_features(prior: list[Candle] | None) -> np.ndarray:
    out = np.zeros(PRIOR_M5_DIM, dtype=float)
    if not prior:
        return out
    closes = np.array([c.close for c in prior], dtype=float)
    highs = np.array([c.high for c in prior], dtype=float)
    lows = np.array([c.low for c in prior], dtype=float)
    vols = np.array([c.volume for c in prior], dtype=float)
    scale = max(atr_proxy(prior), 1e-9)
    slope, r2 = _linreg(closes)
    rng = max(float(highs.max() - lows.min()), 1e-9)
    out[0] = slope * len(closes) / scale
    out[1] = r2
    out[2] = float((closes[-1] - closes[0]) / scale)
    out[3] = float((closes[-1] - lows.min()) / rng * 2 - 1)
    out[4] = float(np.mean(highs - lows) / scale)
    out[5] = float((closes[-1] - closes[-2]) / scale) if len(closes) > 1 else 0.0
    out[6] = 0.0 if vols.std() < 1e-9 else float((vols[-1] - vols.mean()) / vols.std())
    out[7] = len(prior) / 6.0
    if len(closes) >= 3:
        out[8] = float((closes[-1] - closes[-3]) / scale)
    out[9] = 1.0 if len(highs) > 1 and closes[-1] >= highs[:-1].max() else 0.0
    out[10] = 1.0 if len(lows) > 1 and closes[-1] <= lows[:-1].min() else 0.0
    out[11] = float((prior[-1].body) / scale)
    return out


def daytrade_features(candles: list[Candle], prior_m5: list[Candle] | None = None) -> np.ndarray:
    bars = m5_bar_vectors(candles)
    struct = _structure_features(candles)
    vol_sig = volume_pattern_features(candles)
    chart = classify_chart(candles)
    onehot = np.zeros(len(ChartType), dtype=float)
    onehot[CHART_INDEX[chart]] = 1.0
    return np.concatenate([bars, struct, vol_sig, onehot, prior_m5_features(prior_m5)])


def slope_norm(candles: list[Candle]) -> float:
    closes = np.array([c.close for c in candles], dtype=float)
    scale = max(atr_proxy(candles), 1e-9)
    slope, _ = _linreg(closes)
    return slope * max(len(closes), 1) / scale


def swing_features(day_m5: list[Candle]) -> np.ndarray:
    if not day_m5:
        return np.zeros(16 + len(DayType), dtype=float)
    closes = np.array([c.close for c in day_m5], dtype=float)
    highs = np.array([c.high for c in day_m5], dtype=float)
    lows = np.array([c.low for c in day_m5], dtype=float)
    rng = max(float(highs.max() - lows.min()), 1e-9)
    slope, r2 = _linreg(closes)
    slope_n = slope * len(closes) / rng
    close_loc = (closes[-1] - lows.min()) / rng
    open_loc = (day_m5[0].open - lows.min()) / rng
    body = abs(closes[-1] - day_m5[0].open) / rng
    vol = np.array([c.volume for c in day_m5], dtype=float)
    vol_z = 0.0 if vol.std() < 1e-9 else float((vol[-1] - vol.mean()) / vol.std())
    idx = np.linspace(0, len(closes) - 1, 8).astype(int)
    sampled = (closes[idx] - closes[-1]) / rng
    day = classify_day(day_m5)
    onehot = np.zeros(len(DayType), dtype=float)
    onehot[DAY_INDEX[day]] = 1.0
    extra = np.array(
        [slope_n, r2, close_loc, open_loc, body, vol_z, rng / max(atr_proxy(day_m5), 1e-9), len(day_m5) / 90.0],
        dtype=float,
    )
    return np.concatenate([sampled, extra, onehot])
