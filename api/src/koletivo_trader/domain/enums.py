from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def opposite(self) -> "Side":
        if self is Side.BUY:
            return Side.SELL
        if self is Side.SELL:
            return Side.BUY
        return Side.HOLD


class DayType(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    NORMAL = "normal"
    NORMAL_VARIATION = "normal_variation"
    NEUTRAL = "neutral"
    NON_TREND = "non_trend"
    VOLATILE = "volatile"


class ChartType(str, Enum):
    IMPULSE_UP = "impulse_up"
    IMPULSE_DOWN = "impulse_down"
    PULLBACK_UP = "pullback_up"
    PULLBACK_DOWN = "pullback_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    INDECISION = "indecision"


class OrderMode(str, Enum):
    PAPER = "paper"
    MT5 = "mt5"
    PRD = "prd"


class TradeResult(str, Enum):
    GAIN = "gain"
    STOP = "stop"
    BE = "be"
    MANUAL = "manual"
    NONE = "none"


DAY_TYPE_PT = {
    DayType.TREND_UP: "dia de tendência de alta",
    DayType.TREND_DOWN: "dia de tendência de queda",
    DayType.NORMAL: "dia normal",
    DayType.NORMAL_VARIATION: "dia de variação normal expandida",
    DayType.NEUTRAL: "dia neutro",
    DayType.NON_TREND: "dia sem tendência (faixa estreita)",
    DayType.VOLATILE: "dia volátil / instável",
}

CHART_TYPE_PT = {
    ChartType.IMPULSE_UP: "impulso altista",
    ChartType.IMPULSE_DOWN: "impulso baixista",
    ChartType.PULLBACK_UP: "pullback de alta",
    ChartType.PULLBACK_DOWN: "pullback de baixa",
    ChartType.CONSOLIDATION: "consolidação estreita",
    ChartType.BREAKOUT: "rompimento",
    ChartType.REVERSAL: "reversão",
    ChartType.INDECISION: "indecisão",
}

SIDE_PT = {
    Side.BUY: "Compra",
    Side.SELL: "Venda",
    Side.HOLD: "Não fazer nada",
}
