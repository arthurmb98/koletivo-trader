from __future__ import annotations

import threading
from datetime import date, datetime
from typing import Any

from koletivo_trader.adapters.candles import load_candles
from koletivo_trader.adapters.config import load_named_config
from koletivo_trader.domain.enums import Side, TradeResult
from koletivo_trader.domain.fibonacci import fib_boost
from koletivo_trader.domain.fusion import fuse_signals
from koletivo_trader.domain.models import Trade
from koletivo_trader.domain.risk import RiskCalculator, contracts_for_bank, round_to_tick
from koletivo_trader.domain.session import SessionFilter
from koletivo_trader.ml.labels import simulate_touch
from koletivo_trader.ml.models import DaytradeModel, SwingModel, group_days
from koletivo_trader.paths import RESULTS_DIR

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class ReplayEngine:
    """Paper walk of CSV windows. Does not write journal or send MT5 orders."""

    def __init__(self) -> None:
        self.cfg = load_named_config("best_candles_m5_1000_a")
        self.running = False
        self.done = False
        self.error: str | None = None
        self.snap: dict[str, Any] = self._empty()
        self._thread: threading.Thread | None = None

    def _empty(self) -> dict[str, Any]:
        bank = self.cfg.account.initial_bank
        return {
            "running": False,
            "done": False,
            "error": None,
            "config": self.cfg.name,
            "case": "last_candles",
            "timeframe": "m5",
            "source": "paper",
            "order_mode": "paper",
            "interval_sec": 1,
            "last_tick": None,
            "last_bar_time": None,
            "cursor": 0,
            "n_bars": 0,
            "initial_bank": bank,
            "lot": "fixed",
            "bank": bank,
            "net_pnl": 0.0,
            "today_pnl": 0.0,
            "avg_daily": 0.0,
            "n_days": 0,
            "n_trades": 0,
            "n_wins": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "contracts": 1,
            "max_contracts": 1,
            "signal": None,
            "position": None,
            "pending": None,
            "trades": [],
            "equity": [],
            "daily": [],
            "signals": [],
            "candles": [],
            "quote": None,
            "open_pnl": 0.0,
            "skip_reason": None,
            "wait_reason": "aguardando_candle",
            "periods": {"window_days": 0, "levels": ["daily"], "series": {"daily": []}, "avg": {}},
        }

    def snapshot(self) -> dict[str, Any]:
        self.snap["running"] = self.running
        self.snap["done"] = self.done
        self.snap["error"] = self.error
        return self.snap

    def stop(self) -> dict[str, Any]:
        self.running = False
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        self.running = False
        self.done = False
        self.snap = self._empty()
        return self.snapshot()

    def start(
        self,
        *,
        start: str,
        end: str,
        initial_bank: float = 1000.0,
        timeframe: str = "m5",
        case: str = "last_candles",
        lot: str = "fixed",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del timeframe, case
        self.running = True
        self.done = False
        self.error = None
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"start": start, "end": end, "initial_bank": initial_bank, "lot": lot},
            daemon=True,
        )
        self._thread.start()
        return self.snapshot()

    def _run(self, start: str, end: str, initial_bank: float, lot: str) -> None:
        try:
            self.snap = self._simulate(date.fromisoformat(start[:10]), date.fromisoformat(end[:10]), initial_bank, lot)
            self.done = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.snap["error"] = str(exc)
        finally:
            self.running = False

    def _simulate(self, start: date, end: date, initial_bank: float, lot: str) -> dict[str, Any]:
        cfg = self.cfg
        m1 = [
            c
            for c in load_candles(cfg.resolve_csv(cfg.data.test_m1))
            if start <= c.timestamp.date() <= end
        ]
        m5 = [
            c
            for c in load_candles(cfg.resolve_csv(cfg.data.test_m5))
            if start <= c.timestamp.date() <= end
        ]
        daytrade = DaytradeModel()
        swing = SwingModel()
        if joblib is not None:
            day_path = RESULTS_DIR / "model_daytrade.joblib"
            swing_path = RESULTS_DIR / "model_swing.joblib"
            if day_path.exists():
                daytrade = joblib.load(day_path)
            if swing_path.exists():
                swing = joblib.load(swing_path)
        session = SessionFilter.from_config(cfg)
        risk = RiskCalculator(cfg.risk.stop_points, cfg.risk.gain_points, cfg.instrument.tick_size)
        days_m5 = group_days(m5)
        ctx = None
        bank = initial_bank
        peak = bank
        max_dd = 0.0
        trades: list[Trade] = []
        signals: list[dict[str, Any]] = []
        equity = [{"t": datetime.combine(start, datetime.min.time()).isoformat(), "bank": bank}]
        m1_by_day: dict[date, list] = {}
        for candle in m1:
            m1_by_day.setdefault(candle.timestamp.date(), []).append(candle)
        for i, day_bars in enumerate(days_m5):
            if i:
                ctx = swing.predict(days_m5[i - 1])
            day = day_bars[0].timestamp.date()
            day_m1 = m1_by_day.get(day, [])
            for j, bar in enumerate(day_bars[:-3]):
                if not session.allows(bar.timestamp):
                    continue
                contracts = 1 if lot != "scaled" else contracts_for_bank(bank)
                last_m1 = [c for c in day_m1 if c.timestamp <= bar.timestamp + __import__("datetime").timedelta(minutes=4)]
                window = last_m1[-15:]
                if len(window) < 15:
                    continue
                entry = day_bars[j + 1].open
                stop, take = risk.levels(Side.BUY, entry)
                prior_m5 = day_bars[: j + 1][-6:]
                raw = daytrade.predict(window, entry, stop, take, prior_m5)
                fused = fuse_signals(
                    raw,
                    ctx,
                    swing_weight=cfg.filters.swing_weight,
                    min_hit_pct=cfg.filters.min_hit_pct,
                    minutes_from_open=session.minutes_from_open(bar.timestamp),
                    first_block_minutes=cfg.filters.first_block_minutes,
                    fib_weight=cfg.filters.fib_weight,
                    fib_boost=fib_boost(raw.side, entry, window, tick=float(cfg.instrument.tick_size)),
                )
                payload = fused.to_dict()
                payload["t"] = bar.timestamp.isoformat()
                signals.append(payload)
                if fused.side is Side.HOLD:
                    continue
                offset = float(getattr(cfg.execution, "offset_points", 0.0) or 0.0)
                entry = round_to_tick(entry + offset, float(cfg.instrument.tick_size))
                fused.entry = entry
                fused.stop, fused.take = risk.levels(fused.side, entry)
                future = day_bars[j + 1 : j + 4]
                result = simulate_touch(fused.side, entry, cfg.risk.stop_points, cfg.risk.gain_points, future)
                if result is TradeResult.NONE:
                    continue
                exit_px = fused.take if result is TradeResult.GAIN else fused.stop
                points = cfg.risk.gain_points if result is TradeResult.GAIN else -cfg.risk.stop_points
                pnl = points * cfg.account.point_value * contracts - cfg.account.contract_cost * contracts
                trades.append(
                    Trade(
                        side=fused.side,
                        entry_time=future[0].timestamp,
                        exit_time=future[-1].timestamp,
                        entry=entry,
                        exit=exit_px,
                        stop=fused.stop,
                        take=fused.take,
                        points=points,
                        pnl=pnl,
                        result=result,
                        reason=fused.reason,
                        contracts=contracts,
                    )
                )
                bank += pnl
                peak = max(peak, bank)
                max_dd = max(max_dd, peak - bank)
                equity.append({"t": future[-1].timestamp.isoformat(), "bank": bank})
        wins = sum(1 for t in trades if t.pnl > 0)
        daily: dict[str, float] = {}
        for trade in trades:
            key = trade.entry_time.date().isoformat()
            daily[key] = daily.get(key, 0.0) + trade.pnl
        snap = self._empty()
        snap.update(
            {
                "running": False,
                "done": True,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "n_bars": len(m5),
                "cursor": len(m5),
                "initial_bank": initial_bank,
                "bank": bank,
                "net_pnl": bank - initial_bank,
                "today_pnl": list(daily.values())[-1] if daily else 0.0,
                "n_days": len(daily),
                "n_trades": len(trades),
                "n_wins": wins,
                "win_rate": (wins / len(trades) * 100.0) if trades else 0.0,
                "max_drawdown": max_dd,
                "max_drawdown_pct": (max_dd / initial_bank * 100.0) if initial_bank else 0.0,
                "contracts": contracts_for_bank(bank) if lot == "scaled" else 1,
                "trades": [t.to_dict() for t in trades[-80:]],
                "signals": signals[-40:][::-1],
                "equity": equity[-200:],
                "daily": [{"t": k, "pnl": v} for k, v in daily.items()],
                "candles": [
                    {"t": c.timestamp.isoformat(), "open": c.open, "high": c.high, "low": c.low, "close": c.close}
                    for c in m5[-80:]
                ],
                "signal": signals[-1] if signals else None,
                "periods": {
                    "window_days": len(daily),
                    "levels": ["daily"],
                    "series": {"daily": [{"t": k, "pnl": v} for k, v in daily.items()]},
                    "avg": {
                        "daily": {
                            "avg": (sum(daily.values()) / len(daily)) if daily else 0.0,
                            "avg_gain": 0.0,
                            "avg_loss": 0.0,
                            "n": len(daily),
                            "n_gain": sum(1 for v in daily.values() if v > 0),
                            "n_loss": sum(1 for v in daily.values() if v < 0),
                        }
                    },
                },
            }
        )
        return snap


_REPLAY: ReplayEngine | None = None


def get_replay_engine() -> ReplayEngine:
    global _REPLAY
    if _REPLAY is None:
        _REPLAY = ReplayEngine()
    return _REPLAY


def replay_meta(timeframe: str = "m5") -> dict[str, Any]:
    del timeframe
    return {
        "banks": [500, 1000, 2000, 3000, 5000, 10000],
        "cases": [{"key": "last_candles", "label": "15 x M1 → 3 x M5"}],
        "timeframes": [{"key": "m5", "label": "5 min"}],
        "timeframe": "m5",
        "min_date": "2025-01-02",
        "max_date": "2026-08-26",
        "default_start": "2026-08-17",
        "default_end": "2026-08-21",
        "max_span_months": 3,
        "lots": [{"key": "fixed", "label": "1 contrato"}, {"key": "scaled", "label": "Crescente / R$ 1.000"}],
        "lot": "fixed",
    }
