from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from koletivo_trader.adapters.config import AppConfig
from koletivo_trader.domain.copy import phrase_for
from koletivo_trader.domain.enums import Side, TradeResult
from koletivo_trader.domain.fibonacci import fib_boost
from koletivo_trader.domain.fusion import fuse_signals
from koletivo_trader.domain.market import classify_chart
from koletivo_trader.domain.models import Candle, Signal
from koletivo_trader.domain.product import BANKS
from koletivo_trader.domain.risk import contracts_for_bank
from koletivo_trader.domain.session import SessionFilter
from koletivo_trader.ml.features import daytrade_features, slope_norm
from koletivo_trader.ml.labels import leak_free_windows, prior_m5_bars, same_day_m1, simulate_touch_at
from koletivo_trader.ml.genetics import run_genetic_search
from koletivo_trader.ml.models import DaytradeModel, SwingModel, group_days


@dataclass
class ParamResult:
    stop_points: float
    gain_points: float
    min_hit_pct: float
    swing_weight: float
    fib_weight: float
    bank: float
    contracts: int
    n_trades: int
    n_wins: int
    win_rate: float
    net_pnl: float
    max_dd: float
    profit_factor: float = 1.0
    expectancy: float = 0.0
    offset_points: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class _Prepared:
    ts: object
    minutes: float
    entry: float
    future: list[Candle]
    raw: Signal
    ctx: object
    bar_id: str
    fib_boost: float
    session_m1: list[Candle]


def _prepare(
    windows: list,
    swing_by_day: dict,
    cfg: AppConfig,
    model: DaytradeModel,
    m5: list[Candle],
    m1: list[Candle],
) -> list[_Prepared]:
    session = SessionFilter.from_config(cfg)
    m5_index = {c.timestamp: i for i, c in enumerate(m5)}
    m1_index = {c.timestamp: i for i, c in enumerate(m1)}
    tick = float(cfg.instrument.tick_size)
    filtered: list[tuple] = []
    for window, future, entry_bar in windows:
        ts = entry_bar.timestamp
        if not session.allows(ts):
            continue
        if getattr(model.recipe, "morning_only", False) and ts.hour >= 11:
            continue
        filtered.append((window, future, entry_bar, prior_m5_bars(m5, entry_bar, index=m5_index)))
    if not filtered:
        return []
    print(f"  montando features de {len(filtered)} barras…", flush=True)
    X = np.vstack([daytrade_features(w, p) for w, _, _, p in filtered])
    p_buy, p_sell, q_buy, q_sell = model.score_matrix(X)
    from koletivo_trader.domain.enums import ChartType

    predicted_all: list = [None] * len(filtered)
    if model._chart is not None:
        names = model._chart_enc.inverse_transform(model._chart.predict(X))
        predicted_all = [ChartType(name) for name in names]
    out: list[_Prepared] = []
    last_bar_id = None
    for i, (window, future, entry_bar, _prior) in enumerate(filtered):
        ts = entry_bar.timestamp
        bar_id = ts.isoformat()
        if bar_id == last_bar_id:
            continue
        last_bar_id = bar_id
        pb, ps, qb, qs = float(p_buy[i]), float(p_sell[i]), float(q_buy[i]), float(q_sell[i])
        sl = slope_norm(window)
        if model.recipe.mode == "momentum_meta":
            if sl >= model.recipe.momentum_cut:
                ps = 0.0
                pb = max(pb, 0.65)
            elif sl <= -model.recipe.momentum_cut:
                pb = 0.0
                ps = max(ps, 0.65)
            else:
                pb = ps = 0.0
        elif model.recipe.mode == "agree":
            if sl >= model.recipe.momentum_cut:
                ps *= 0.25
            elif sl <= -model.recipe.momentum_cut:
                pb *= 0.25
            else:
                pb *= 0.4
                ps *= 0.4
        if pb >= ps:
            side, p_dir, q = Side.BUY, pb, qb
        else:
            side, p_dir, q = Side.SELL, ps, qs
        hit = float(q)
        if p_dir < 0.52 or q < model.min_score:
            side = Side.HOLD
        chart = classify_chart(window)
        predicted = predicted_all[i]
        raw = Signal(
            side=side,
            entry=entry_bar.open,
            stop=0.0,
            take=0.0,
            chart_type=chart,
            hit_pct=hit,
            phrase=phrase_for(side, chart, hit, predicted_chart=predicted),
            reason="daytrade",
            predicted_chart_type=predicted,
        )
        start = m1_index.get(entry_bar.timestamp)
        if start is None:
            path = future
        else:
            day = entry_bar.timestamp.date()
            path = []
            for candle in m1[start : start + 180]:
                if candle.timestamp.date() != day:
                    break
                path.append(candle)
            path = path or future
        boost = fib_boost(side, entry_bar.open, window, tick=tick)
        out.append(
            _Prepared(
                ts=ts,
                minutes=session.minutes_from_open(ts),
                entry=entry_bar.open,
                future=path,
                raw=raw,
                ctx=swing_by_day.get(window[-1].timestamp.date()),
                bar_id=bar_id,
                fib_boost=boost,
                session_m1=window,
            )
        )
    return out


def _score_prepared(
    rows: list[_Prepared],
    cfg: AppConfig,
    stop: float,
    gain: float,
    min_hit: float,
    weight: float,
    bank: float,
    fib_weight: float = 0.0,
    offset_points: float = 0.0,
) -> ParamResult:
    contracts = contracts_for_bank(bank)
    pv = cfg.account.point_value
    tick = float(cfg.instrument.tick_size)
    from koletivo_trader.domain.risk import round_to_tick

    equity = bank
    peak = bank
    max_dd = 0.0
    n_trades = 0
    n_wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    trades_today = 0
    last_day = None
    cooldown = None
    daily_loss_cap = float(cfg.risk.daily_loss_points or 0.0)
    day_points = 0.0
    for row in rows:
        day = row.ts.date() if hasattr(row.ts, "date") else None
        if day != last_day:
            trades_today = 0
            day_points = 0.0
            last_day = day
            cooldown = None
        if cooldown is not None and row.ts <= cooldown:
            continue
        fused = fuse_signals(
            row.raw,
            row.ctx,
            swing_weight=weight,
            min_hit_pct=min_hit,
            minutes_from_open=row.minutes,
            first_block_minutes=cfg.filters.first_block_minutes,
            fib_weight=fib_weight,
            fib_boost=row.fib_boost,
        )
        if fused.side is Side.HOLD:
            continue
        if trades_today >= int(cfg.risk.max_trades_per_day):
            continue
        if daily_loss_cap > 0 and day_points <= -daily_loss_cap:
            continue
        entry = round_to_tick(row.entry + offset_points, tick)
        result, hit_ts = simulate_touch_at(fused.side, entry, stop, gain, row.future)
        if result is TradeResult.NONE:
            continue
        n_trades += 1
        trades_today += 1
        cooldown = hit_ts
        points = gain if result is TradeResult.GAIN else -stop
        day_points += points
        pnl = points * pv * contracts - cfg.account.contract_cost * contracts
        if pnl > 0:
            n_wins += 1
            gross_win += pnl
        else:
            gross_loss += abs(pnl)
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    win_rate = (n_wins / n_trades * 100.0) if n_trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss else (2.0 if gross_win else 0.0)
    exp = ((equity - bank) / n_trades) if n_trades else 0.0
    return ParamResult(
        stop,
        gain,
        min_hit,
        weight,
        fib_weight,
        bank,
        contracts,
        n_trades,
        n_wins,
        win_rate,
        equity - bank,
        max_dd,
        pf,
        exp,
        offset_points,
    )


def _swing_map(m5_eval: list[Candle], m5_prev_days: list[list[Candle]], swing: SwingModel) -> dict:
    swing_by_day = {}
    test_days = group_days(m5_eval)
    prev = m5_prev_days[-1] if m5_prev_days else None
    for i, day in enumerate(test_days):
        source = test_days[i - 1] if i else prev
        if source:
            swing_by_day[day[0].timestamp.date()] = swing.predict(source)
    return swing_by_day


def prepare_eval(
    m1: list[Candle],
    m5: list[Candle],
    m5_prev_days: list[list[Candle]],
    cfg: AppConfig,
    model: DaytradeModel,
    swing: SwingModel,
) -> list[_Prepared]:
    windows = leak_free_windows(m1, m5)
    print(f"  {len(windows)} janelas brutas, swing + features…", flush=True)
    swing_by_day = _swing_map(m5, m5_prev_days, swing)
    prepared = _prepare(windows, swing_by_day, cfg, model, m5, m1)
    n_dir = sum(1 for row in prepared if row.raw.side in {Side.BUY, Side.SELL})
    print(f"  janelas={len(windows)} preparadas={len(prepared)} direcionais={n_dir}")
    return prepared


def _row_date(row: _Prepared):
    return row.ts.date() if hasattr(row.ts, "date") else None


def _tune_confirm_split(rows: list[_Prepared], confirm_frac: float = 0.25) -> tuple[list[_Prepared], list[_Prepared]]:
    dates = sorted({d for row in rows if (d := _row_date(row)) is not None})
    if len(dates) < 10:
        return rows, rows
    cut = dates[max(1, int(len(dates) * (1.0 - confirm_frac)))]
    tune = [row for row in rows if _row_date(row) is not None and _row_date(row) < cut]
    confirm = [row for row in rows if _row_date(row) is not None and _row_date(row) >= cut]
    if not tune or not confirm:
        return rows, rows
    return tune, confirm


def _purged_folds(rows: list[_Prepared], n_folds: int = 3, embargo_days: int = 1) -> list[list[_Prepared]]:
    dates = sorted({d for row in rows if (d := _row_date(row)) is not None})
    if len(dates) < n_folds * 4:
        return [rows]
    size = max(len(dates) // n_folds, 1)
    folds: list[list[_Prepared]] = []
    for i in range(n_folds):
        start = i * size
        end = (i + 1) * size if i < n_folds - 1 else len(dates)
        chunk = dates[start:end]
        if i and embargo_days:
            chunk = chunk[embargo_days:]
        allowed = set(chunk)
        part = [row for row in rows if _row_date(row) in allowed]
        if part:
            folds.append(part)
    return folds or [rows]


def _breakeven(stop: float, gain: float, point_value: float, cost: float) -> float:
    win_cash = gain * point_value - cost
    loss_cash = stop * point_value + cost
    return loss_cash / max(win_cash + loss_cash, 1e-9)


def search_parameters(
    m1_eval: list[Candle],
    m5_eval: list[Candle],
    m5_train_days: list[list[Candle]],
    cfg: AppConfig,
    model: DaytradeModel,
    swing: SwingModel,
    banks: tuple[float, ...] = BANKS,
    n_trials: int = 36,
    prepared: list[_Prepared] | None = None,
) -> dict[str, ParamResult]:
    del n_trials
    if prepared is None:
        prepared = prepare_eval(m1_eval, m5_eval, m5_train_days, cfg, model, swing)
    tune, confirm = _tune_confirm_split(prepared)
    folds = _purged_folds(tune, n_folds=3, embargo_days=1)
    print(f"  AG folds={len(folds)} tune={len(tune)} confirm={len(confirm)}", flush=True)

    winners: dict[str, ParamResult] = {}
    for bank in banks:
        def evaluate(params: dict, bank_value: float = bank) -> float:
            return _fold_score(
                folds,
                cfg,
                params["stop"],
                params["gain"],
                params["min_hit"],
                params["swing_weight"],
                params["fib_weight"],
                bank_value,
                params["offset_points"],
            )

        elite = run_genetic_search(evaluate, population=24, generations=16, elite=4, seed=7)
        picked = _score_prepared(
            prepared,
            cfg,
            elite.stop,
            elite.gain,
            elite.min_hit,
            elite.swing_weight,
            bank,
            elite.fib_weight,
            elite.offset_points,
        )
        if confirm and confirm is not tune:
            conf = _score_prepared(
                confirm,
                cfg,
                elite.stop,
                elite.gain,
                elite.min_hit,
                elite.swing_weight,
                bank,
                elite.fib_weight,
                elite.offset_points,
            )
            print(
                f"  confirm banca {int(bank)}: n={conf.n_trades} wr={conf.win_rate:.1f}% pnl={conf.net_pnl:.0f}",
                flush=True,
            )
        winners[str(int(bank))] = picked
        w = picked
        print(
            f"  banca {int(bank)}: trades={w.n_trades} wr={w.win_rate:.1f}% "
            f"pnl={w.net_pnl:.0f} dd={w.max_dd:.0f} stop={w.stop_points:.0f} gain={w.gain_points:.0f} "
            f"min_hit={w.min_hit_pct:.3f} swing_w={w.swing_weight:.3f} fib_w={w.fib_weight:.3f} "
            f"offset={w.offset_points:.0f}",
            flush=True,
        )
    return winners


def _fold_score(
    folds: list[list[_Prepared]],
    cfg: AppConfig,
    stop: float,
    gain: float,
    min_hit: float,
    swing_w: float,
    fib_w: float,
    bank: float,
    offset_points: float = 0.0,
) -> float:
    be = _breakeven(stop, gain, cfg.account.point_value, cfg.account.contract_cost)
    pnls: list[float] = []
    wrs: list[float] = []
    dds: list[float] = []
    n_trades: list[int] = []
    for fold in folds:
        result = _score_prepared(
            fold, cfg, stop, gain, min_hit, swing_w, bank, fib_w, offset_points
        )
        pnls.append(result.net_pnl)
        wrs.append(result.win_rate / 100.0)
        dds.append(result.max_dd / max(bank, 1.0))
        n_trades.append(result.n_trades)
    if min(n_trades) < 6 or sum(n_trades) < 24:
        return -1e6
    median_pnl = float(np.median(pnls))
    mean_wr = float(np.mean(wrs))
    worst_dd = float(np.max(dds))
    if worst_dd > 0.28:
        return -1e5
    if mean_wr < be + 0.015:
        return median_pnl / bank * 5.0 - 20.0
    if median_pnl <= 0:
        return median_pnl / bank * 8.0 - 8.0
    mean_pnl = float(np.mean(pnls))
    stab = 1.0 - float(np.std(pnls)) / (abs(mean_pnl) + bank * 0.05)
    return (
        median_pnl / bank * 28.0
        + (mean_wr - be) * 55.0
        + max(stab, 0.0) * 12.0
        + gain / max(stop, 1.0) * 1.2
        - worst_dd * 45.0
    )


def score_fixed(
    prepared: list[_Prepared],
    cfg: AppConfig,
    result: ParamResult,
    bank: float | None = None,
) -> ParamResult:
    return _score_prepared(
        prepared,
        cfg,
        result.stop_points,
        result.gain_points,
        result.min_hit_pct,
        result.swing_weight,
        bank if bank is not None else result.bank,
        result.fib_weight,
        result.offset_points,
    )
