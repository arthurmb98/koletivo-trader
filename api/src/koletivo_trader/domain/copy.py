from __future__ import annotations

from koletivo_trader.domain.enums import (
    CHART_TYPE_PT,
    DAY_TYPE_PT,
    SIDE_PT,
    ChartType,
    DayType,
    Side,
)


def phrase_for(
    side: Side,
    chart: ChartType,
    hit_pct: float,
    *,
    reason: str = "",
    day_type: DayType | None = None,
    swing_signal: Side | None = None,
) -> str:
    chart_pt = CHART_TYPE_PT[chart]
    pct = f"{hit_pct * 100:.0f}%"
    if reason == "swing_discord":
        swing_pt = SIDE_PT.get(swing_signal or Side.HOLD, "o contexto de D-1")
        day_pt = DAY_TYPE_PT.get(day_type or DayType.NORMAL, "o tipo de dia previsto")
        wanted = SIDE_PT.get(Side.BUY if swing_signal is Side.SELL else Side.SELL, "o daytrade")
        if swing_signal in {Side.BUY, Side.SELL}:
            other = "compra" if swing_signal is Side.SELL else "venda"
            return (
                f"O recorte de 15 minutos parece {chart_pt}, mas o swing de D-1 aponta {swing_pt.lower()} "
                f"({day_pt}). Sem alinhamento nos primeiros 15 min, nenhuma ordem."
            )
        return (
            f"O gráfico curto é {chart_pt} e pediria {wanted.lower()}, mas o contexto de D-1 "
            f"({day_pt}) não confirma. Melhor não comprar nem vender."
        )
    if reason == "day_mismatch":
        day_pt = DAY_TYPE_PT.get(day_type or DayType.NORMAL, "o tipo de dia")
        return (
            f"{chart_pt.capitalize()} em desacordo com {day_pt}. "
            f"A chance de gain cai para {pct} — melhor não comprar nem vender."
        )
    if reason == "low_hit":
        return (
            f"{chart_pt.capitalize()}; a probabilidade de atingir o gain antes do stop é {pct}, "
            "baixa nos dois lados — melhor não comprar nem vender."
        )
    if side is Side.HOLD:
        return (
            f"{chart_pt.capitalize()}; a chance de gain é baixa nos dois lados ({pct}) — "
            "melhor não comprar nem vender."
        )
    verb = "compra" if side is Side.BUY else "venda"
    return (
        f"{chart_pt.capitalize()}; probabilidade de atingir o gain de {verb} antes do stop: {pct}."
    )
