from __future__ import annotations

import threading
from datetime import date, timedelta
from typing import Any

from koletivo_trader.adapters.candles import load_candles
from koletivo_trader.adapters.config import load_bank_config
from koletivo_trader.domain.product import BANKS, CASE, TIMEFRAME
from koletivo_trader.domain.risk import contracts_for_bank
from koletivo_trader.ml.models import DaytradeModel, SwingModel
from koletivo_trader.ml.parameters import prepare_eval, score_bank_window
from koletivo_trader.paths import RESULTS_DIR

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

PAD_DAYS = 14


class ReplayEngine:
    """Paper walk of CSV windows using the same strategy path as the study."""

    def __init__(self) -> None:
        self.cfg = load_bank_config(1000)
        self.running = False
        self.done = False
        self.error: str | None = None
        self.snap: dict[str, Any] = self._empty()
        self._thread: threading.Thread | None = None
        self._daytrade: DaytradeModel | None = None
        self._swing: SwingModel | None = None

    def _empty(self) -> dict[str, Any]:
        bank = self.cfg.account.initial_bank
        return {
            "running": False,
            "done": False,
            "error": None,
            "config": self.cfg.name,
            "case": CASE,
            "timeframe": TIMEFRAME,
            "source": "paper",
            "order_mode": "paper",
            "interval_sec": 1,
            "last_tick": None,
            "last_bar_time": None,
            "cursor": 0,
            "n_bars": 0,
            "initial_bank": bank,
            "lot": "scaled",
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
            "contracts": contracts_for_bank(bank),
            "max_contracts": 10,
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
        timeframe: str = TIMEFRAME,
        case: str = CASE,
        lot: str = "scaled",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del timeframe, case
        self.cfg = load_bank_config(initial_bank)
        self.running = True
        self.done = False
        self.error = None
        self.snap = self._empty()
        self.snap["running"] = True
        self.snap["config"] = self.cfg.name
        self.snap["initial_bank"] = float(initial_bank)
        self.snap["lot"] = lot
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"start": start, "end": end, "initial_bank": initial_bank, "lot": lot},
            daemon=True,
        )
        self._thread.start()
        return self.snapshot()

    def _models(self) -> tuple[DaytradeModel, SwingModel]:
        if self._daytrade is None:
            self._daytrade = DaytradeModel()
            self._swing = SwingModel()
            if joblib is not None:
                day_path = RESULTS_DIR / "model_daytrade.joblib"
                swing_path = RESULTS_DIR / "model_swing.joblib"
                if day_path.exists():
                    self._daytrade = joblib.load(day_path)
                if swing_path.exists():
                    self._swing = joblib.load(swing_path)
        assert self._daytrade is not None and self._swing is not None
        return self._daytrade, self._swing

    def _run(self, start: str, end: str, initial_bank: float, lot: str) -> None:
        try:
            self.snap = self._simulate(
                date.fromisoformat(start[:10]),
                date.fromisoformat(end[:10]),
                initial_bank,
                lot,
            )
            self.done = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.snap["error"] = str(exc)
        finally:
            self.running = False

    def _simulate(self, start: date, end: date, initial_bank: float, lot: str) -> dict[str, Any]:
        cfg = load_bank_config(initial_bank)
        self.cfg = cfg
        pad_start = start - timedelta(days=PAD_DAYS)
        m1_all = load_candles(cfg.resolve_csv(cfg.data.test_m1))
        m5_all = load_candles(cfg.resolve_csv(cfg.data.test_m5))
        m1 = [c for c in m1_all if pad_start <= c.timestamp.date() <= end]
        m5 = [c for c in m5_all if pad_start <= c.timestamp.date() <= end]
        m5_window = [c for c in m5 if start <= c.timestamp.date() <= end]
        daytrade, swing = self._models()
        prepared = prepare_eval(m1, m5, [], cfg, daytrade, swing, verbose=False)
        rows = [
            row
            for row in prepared
            if hasattr(row.ts, "date") and start <= row.ts.date() <= end
        ]
        scored = score_bank_window(rows, cfg, initial_bank, lot=lot, collect_signals=True)
        contracts = 1 if lot == "fixed" else contracts_for_bank(initial_bank)
        trades = [t.to_dict() if hasattr(t, "to_dict") else t for t in scored.trades]
        signals = list(scored.signals)
        daily_rows = list(scored.daily)
        daily_map = {row["t"]: float(row.get("pnl") or 0.0) for row in daily_rows}
        wins = scored.n_wins
        snap = self._empty()
        snap.update(
            {
                "running": False,
                "done": True,
                "config": cfg.name,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "n_bars": len(m5_window),
                "cursor": len(m5_window),
                "initial_bank": initial_bank,
                "lot": lot,
                "bank": initial_bank + scored.net_pnl,
                "net_pnl": scored.net_pnl,
                "today_pnl": daily_rows[-1]["pnl"] if daily_rows else 0.0,
                "n_days": len(daily_rows),
                "n_trades": scored.n_trades,
                "n_wins": wins,
                "win_rate": scored.win_rate,
                "max_drawdown": scored.max_dd,
                "max_drawdown_pct": (scored.max_dd / initial_bank * 100.0) if initial_bank else 0.0,
                "contracts": contracts,
                "max_contracts": 10,
                "trades": trades[-80:],
                "signals": list(reversed(signals[-40:])),
                "equity": scored.equity[-200:] if scored.equity else [],
                "daily": [{"t": k, "pnl": v} for k, v in daily_map.items()],
                "candles": [
                    {
                        "t": c.timestamp.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                    }
                    for c in m5_window[-80:]
                ],
                "signal": signals[-1] if signals else None,
                "last_bar_time": m5_window[-1].timestamp.isoformat() if m5_window else None,
                "periods": {
                    "window_days": len(daily_map),
                    "levels": ["daily"],
                    "series": {"daily": [{"t": k, "pnl": v} for k, v in daily_map.items()]},
                    "avg": {
                        "daily": {
                            "avg": (sum(daily_map.values()) / len(daily_map)) if daily_map else 0.0,
                            "avg_gain": 0.0,
                            "avg_loss": 0.0,
                            "n": len(daily_map),
                            "n_gain": sum(1 for v in daily_map.values() if v > 0),
                            "n_loss": sum(1 for v in daily_map.values() if v < 0),
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


def replay_meta(timeframe: str = TIMEFRAME) -> dict[str, Any]:
    del timeframe
    return {
        "banks": [int(bank) for bank in BANKS],
        "cases": [{"key": CASE, "label": "Últimos candles"}],
        "timeframes": [{"key": TIMEFRAME, "label": "5 min"}],
        "timeframe": TIMEFRAME,
        "min_date": "2025-01-02",
        "max_date": "2026-08-26",
        "default_start": "2026-08-17",
        "default_end": "2026-08-21",
        "max_span_months": 3,
        "lots": [
            {"key": "fixed", "label": "1 contrato"},
            {"key": "scaled", "label": "Contratos da banca (estudo)"},
        ],
        "lot": "scaled",
    }
