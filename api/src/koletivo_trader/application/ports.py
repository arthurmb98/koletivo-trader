from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Protocol

from koletivo_trader.domain.enums import Side
from koletivo_trader.domain.models import Candle, SessionContext, Signal


class MarketFeed(ABC):
    @abstractmethod
    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        ...

    def candles_for_day(self, symbol: str, timeframe: str, day: date) -> list[Candle]:
        return []

    def status(self) -> dict[str, Any]:
        return {"ready": False}


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def last_closed_candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        ...

    @abstractmethod
    def send(self, signal: Signal, volume: float) -> dict[str, Any]:
        ...

    @abstractmethod
    def close_position(self, ticket: int, side: Side, volume: float) -> dict[str, Any]:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...


class IntradayPredictor(Protocol):
    def predict(self, m1_window: list[Candle], entry: float, stop: float, take: float) -> Signal:
        ...


class ContextPredictor(Protocol):
    def predict(self, previous_m5: list[Candle]) -> SessionContext:
        ...
