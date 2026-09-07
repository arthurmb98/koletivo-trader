from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from koletivo_trader.domain.models import Candle

OHLC = ["Abertura", "Máximo", "Mínimo", "Fechamento"]
WRITE_COLS = ["Ativo", "Data", "Hora", *OHLC, "Volume"]


def load_candles(path: Path, symbol: str = "WIN$") -> list[Candle]:
    frame = read_frame(path)
    return frame_to_candles(frame, symbol=symbol)


def read_frame(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for sep in (",", ";", "\t"):
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding)
                if len(df.columns) == 1:
                    continue
                return _normalize(df)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    raise ValueError(f"Não foi possível ler {path}: {last_error}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip("<>").strip() for c in df.columns]
    if {"Ativo", "Data", "Hora", *OHLC} <= set(df.columns):
        ts = pd.to_datetime(
            df["Data"].astype(str).str.strip() + " " + df["Hora"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
        out = df.copy()
        out["timestamp"] = ts
        return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    lower = {c.lower(): c for c in df.columns}
    if {"date", "time", "open", "high", "low", "close"} <= set(lower):
        ts = pd.to_datetime(
            df[lower["date"]].astype(str).str.strip() + " " + df[lower["time"]].astype(str).str.strip(),
            errors="coerce",
        )
        out = pd.DataFrame(
            {
                "Ativo": "WIN$",
                "timestamp": ts,
                "Abertura": pd.to_numeric(df[lower["open"]], errors="coerce"),
                "Máximo": pd.to_numeric(df[lower["high"]], errors="coerce"),
                "Mínimo": pd.to_numeric(df[lower["low"]], errors="coerce"),
                "Fechamento": pd.to_numeric(df[lower["close"]], errors="coerce"),
                "Volume": pd.to_numeric(df[lower["vol"]], errors="coerce") if "vol" in lower else 0.0,
            }
        )
        out["Data"] = out["timestamp"].dt.strftime("%d/%m/%Y")
        out["Hora"] = out["timestamp"].dt.strftime("%H:%M:%S")
        return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    raise ValueError(f"CSV sem colunas reconhecidas: {list(df.columns)}")


def frame_to_candles(frame: pd.DataFrame, symbol: str = "WIN$") -> list[Candle]:
    candles: list[Candle] = []
    for row in frame.itertuples(index=False):
        ts = getattr(row, "timestamp")
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        candles.append(
            Candle(
                symbol=str(getattr(row, "Ativo", symbol) or symbol),
                timestamp=ts,
                open=float(row.Abertura),
                high=float(row.Máximo),
                low=float(row.Mínimo),
                close=float(row.Fechamento),
                volume=float(getattr(row, "Volume", 0) or 0),
            )
        )
    return candles


def candles_on_day(candles: list[Candle], day) -> list[Candle]:
    return [c for c in candles if c.timestamp.date() == day]


def unique_days(candles: list[Candle]) -> list:
    seen: list = []
    for candle in candles:
        day = candle.timestamp.date()
        if not seen or seen[-1] != day:
            seen.append(day)
    return seen
