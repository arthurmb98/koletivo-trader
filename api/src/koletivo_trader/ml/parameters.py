from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna

from koletivo_trader.adapters.config import AppConfig
from koletivo_trader.domain.enums import Side, TradeResult
from koletivo_trader.domain.fusion import fuse_signals
from koletivo_trader.domain.models import Candle, Signal
from koletivo_trader.domain.risk import contracts_for_bank
from koletivo_trader.domain.session import SessionFilter
from koletivo_trader.ml.features import daytrade_features, slope_norm
from koletivo_trader.ml.labels import leak_free_windows, prior_m5_bars, simulate_touch_at
from koletivo_trader.ml.models import DaytradeModel, SwingModel, group_days
from koletivo_trader.domain.market import classify_chart
from koletivo_trader.domain.copy import phrase_for


@dataclass
class ParamResult:
    stop_points: float
    gain_points: float
    min_hit_pct: float
    swing_weight: float
    bank: float
    contracts: int
    n_trades: int
    n_wins: int
    win_rate: float
    net_pnl: float
    max_dd: float
    profit_factor: float = 1.0
    expectancy: float = 0.0

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


def _prepare(
    windows: list,
    swing_by_day: dict,
    cfg: AppConfig,
    model: DaytradeModel,
    m5: list[Candle],
) -> list[_Prepared]:
    session = SessionFilter.from_config(cfg)
    m5_index = {c.timestamp: i for i, c in enumerate(m5)}
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
    X = np.vstack([daytrade_features(w, p) for w, _, _, p in filtered])
    p_buy, p_sell, q_buy, q_sell = model.score_matrix(X)
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
        raw = Signal(
            side=side,
            entry=entry_bar.open,
            stop=0.0,
            take=0.0,
            chart_type=chart,
            hit_pct=hit,
            phrase=phrase_for(side, chart, hit),
            reason="daytrade",
        )
        out.append(
            _Prepared(
                ts=ts,
                minutes=session.minutes_from_open(ts),
                entry=entry_bar.open,
                future=future,
                raw=raw,
                ctx=swing_by_day.get(window[-1].timestamp.date()),
                bar_id=bar_id,
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
) -> ParamResult:
    contracts = contracts_for_bank(bank)
    pv = cfg.account.point_value
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
        )
        if fused.side is Side.HOLD:
            continue
        if trades_today >= int(cfg.risk.max_trades_per_day):
            continue
        if daily_loss_cap > 0 and day_points <= -daily_loss_cap:
            continue
        result, hit_ts = simulate_touch_at(fused.side, row.entry, stop, gain, row.future)
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
        stop, gain, min_hit, weight, bank, contracts, n_trades, n_wins, win_rate, equity - bank, max_dd, pf, exp
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
    swing_by_day = _swing_map(m5, m5_prev_days, swing)
    prepared = _prepare(windows, swing_by_day, cfg, model, m5)
    n_dir = sum(1 for row in prepared if row.raw.side in {Side.BUY, Side.SELL})
    print(f"  janelas={len(windows)} preparadas={len(prepared)} direcionais={n_dir}")
    return prepared


def search_parameters(
    m1_eval: list[Candle],
    m5_eval: list[Candle],
    m5_train_days: list[list[Candle]],
    cfg: AppConfig,
    model: DaytradeModel,
    swing: SwingModel,
    banks: tuple[float, ...] = (500.0, 1000.0, 5000.0),
    n_trials: int = 24,
    prepared: list[_Prepared] | None = None,
) -> dict[str, ParamResult]:
    if prepared is None:
        prepared = prepare_eval(m1_eval, m5_eval, m5_train_days, cfg, model, swing)

    winners: dict[str, ParamResult] = {}
    for bank in banks:
        def objective(trial: optuna.Trial, bank_value: float = bank) -> float:
            stop = trial.suggest_categorical("stop", [40.0, 50.0, 60.0, 80.0, 100.0])
            gain = trial.suggest_categorical("gain", [80.0, 100.0, 120.0, 150.0, 160.0, 200.0])
            if gain < stop * 1.5:
                return -1e6
            min_hit = trial.suggest_float("min_hit", 0.30, 0.62)
            weight = trial.suggest_float("swing_weight", 0.05, 0.22)
            result = _score_prepared(prepared, cfg, stop, gain, min_hit, weight, bank_value)
            if result.n_trades < 15:
                return -1e6
            pv = cfg.account.point_value
            cost = cfg.account.contract_cost
            win_cash = gain * pv - cost
            loss_cash = stop * pv + cost
            be = loss_cash / max(win_cash + loss_cash, 1e-9)
            wr = result.win_rate / 100.0
            if wr < be + 0.02:
                return result.net_pnl / max(bank_value, 1.0) * 8.0 - 25.0
            if result.max_dd > bank_value * 0.35:
                return -1e5
            if result.net_pnl <= 0:
                return result.net_pnl / max(bank_value, 1.0) * 10.0 - 5.0
            dd_pen = result.max_dd / max(bank_value, 1.0)
            rr = gain / max(stop, 1.0)
            return (
                result.net_pnl / bank_value * 30.0
                + (wr - be) * 50.0
                + result.profit_factor * 5.0
                + rr * 1.5
                - dd_pen * 40.0
            )

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        winners[str(int(bank))] = _score_prepared(
            prepared,
            cfg,
            float(best["stop"]),
            float(best["gain"]),
            float(best["min_hit"]),
            float(best["swing_weight"]),
            bank,
        )
        w = winners[str(int(bank))]
        print(
            f"  banca {int(bank)}: trades={w.n_trades} wr={w.win_rate:.1f}% "
            f"pnl={w.net_pnl:.0f} dd={w.max_dd:.0f} stop={w.stop_points:.0f} gain={w.gain_points:.0f} "
            f"min_hit={w.min_hit_pct:.3f}"
        )
    return winners


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
    )
