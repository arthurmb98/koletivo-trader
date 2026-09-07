from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from koletivo_trader.domain.enums import ChartType, DayType, Side, TradeResult


@dataclass(frozen=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    def to_ohlc(self) -> tuple[float, float, float, float]:
        return self.open, self.high, self.low, self.close


@dataclass
class Signal:
    side: Side
    entry: float
    stop: float
    take: float
    chart_type: ChartType
    hit_pct: float
    phrase: str
    reason: str = ""
    day_type: DayType | None = None
    swing_signal: Side | None = None
    sent: bool = False
    predicted_chart_type: ChartType | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "entry": self.entry,
            "stop": self.stop,
            "take": self.take,
            "chart_type": self.chart_type.value,
            "predicted_chart_type": None
            if self.predicted_chart_type is None
            else self.predicted_chart_type.value,
            "hit_pct": self.hit_pct,
            "phrase": self.phrase,
            "reason": self.reason,
            "day_type": None if self.day_type is None else self.day_type.value,
            "swing_signal": None if self.swing_signal is None else self.swing_signal.value,
            "sent": self.sent,
        }


@dataclass
class SessionContext:
    as_of: datetime
    previous_date: str
    swing_signal: Side
    day_type: DayType
    predicted_day_type: DayType
    swing_hit_pct: float
    phrase: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        data["swing_signal"] = self.swing_signal.value
        data["day_type"] = self.day_type.value
        data["predicted_day_type"] = self.predicted_day_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionContext":
        return cls(
            as_of=datetime.fromisoformat(str(data["as_of"])),
            previous_date=str(data["previous_date"]),
            swing_signal=Side(data["swing_signal"]),
            day_type=DayType(data["day_type"]),
            predicted_day_type=DayType(data["predicted_day_type"]),
            swing_hit_pct=float(data["swing_hit_pct"]),
            phrase=str(data.get("phrase") or ""),
        )


@dataclass
class Trade:
    side: Side
    entry_time: datetime
    exit_time: datetime | None
    entry: float
    exit: float
    stop: float
    take: float
    points: float
    pnl: float
    result: TradeResult
    reason: str
    contracts: float = 1
    ticket: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": None if self.exit_time is None else self.exit_time.isoformat(),
            "entry": self.entry,
            "exit": self.exit,
            "stop": self.stop,
            "take": self.take,
            "points": self.points,
            "pnl": self.pnl,
            "result": self.result.value,
            "reason": self.reason,
            "contracts": self.contracts,
            "ticket": self.ticket,
        }


@dataclass
class Position:
    side: Side
    entry: float
    stop: float
    take: float
    time: datetime
    contracts: float
    ticket: int | None = None
    reason: str = ""
    extreme: float = 0.0
    orig_stop: float = 0.0
    orig_take: float = 0.0
