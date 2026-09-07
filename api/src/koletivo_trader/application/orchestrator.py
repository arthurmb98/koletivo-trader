from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib

from koletivo_trader.adapters.candles import load_candles
from koletivo_trader.adapters.config import AppConfig, load_named_config
from koletivo_trader.adapters.journal import CsvJournal
from koletivo_trader.adapters.mt5.broker import Mt5Broker
from koletivo_trader.adapters.mt5.session import (
    DEMO_PLAYBOOK,
    env_credentials,
    redact_text,
    next_gold_window,
    resolve_symbol,
    session_wait_reason,
)
from koletivo_trader.domain.copy import phrase_for
from koletivo_trader.domain.enums import ChartType, OrderMode, Side, TradeResult
from koletivo_trader.domain.fibonacci import fib_boost
from koletivo_trader.domain.fusion import fuse_signals
from koletivo_trader.domain.models import Candle, Position, SessionContext, Signal, Trade
from koletivo_trader.domain.risk import RiskCalculator, contracts_for_bank, protect_levels, round_to_tick
from koletivo_trader.domain.session import SessionFilter
from koletivo_trader.ml.models import DaytradeModel, SwingModel, group_days
from koletivo_trader.domain.product import CASE, LOOKBACK_M1, TIMEFRAME
from koletivo_trader.paths import RESULTS_DIR

CONFIG_NAME = "best_candles_m5_1000_a"


def _today() -> date:
    return datetime.now().date()


def _bar_id(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M")


class LiveEngine:
    """ARMADO: no fechamento de M5 envia ordem a mercado com SL/TP fixos.

    Em posição, lê ticks a cada ~20 ms (cadência típica do WIN no MT5) e só chama
    modify SL quando o stop muda pelo menos 1 tick. Fora de posição, 100 ms.
    """

    def __init__(self) -> None:
        cfg_name = "best_bank_1000"
        from koletivo_trader.paths import CONFIGS_DIR as _cfgs

        if not (_cfgs / f"{cfg_name}.yaml").exists():
            cfg_name = CONFIG_NAME
        self.cfg = load_named_config(cfg_name)
        self.session = SessionFilter.from_config(self.cfg)
        self.risk = RiskCalculator(
            self.cfg.risk.stop_points,
            self.cfg.risk.gain_points,
            self.cfg.instrument.tick_size,
        )
        self.journal = CsvJournal()
        self.daytrade = DaytradeModel()
        self.swing = SwingModel()
        self._load_models()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.armed = False
        self.order_mode = OrderMode.PAPER
        self.error: str | None = None
        self.skip_reason: str | None = None
        self.wait_reason = "mercado_fechado"
        self.context: SessionContext | None = None
        self.signal: Signal | None = None
        self.position: Position | None = None
        self.broker: Mt5Broker | None = None
        self.symbol = self.cfg.mt5.symbol
        self.initial_bank = self.cfg.account.initial_bank
        self.bank = self.initial_bank
        self.contracts = contracts_for_bank(self.initial_bank)
        self.trades: list[Trade] = []
        self.signals: list[dict[str, Any]] = []
        self.equity: list[dict[str, Any]] = []
        self.candles: list[dict[str, Any]] = []
        self.quote: dict[str, Any] | None = None
        self.last_bar: datetime | None = None
        self.last_tick: datetime | None = None
        self.last_tick_msc = 0
        self.decided_bars: set[str] = set()
        self._last_sl_modify = 0.0
        self._mt5_payload: dict[str, Any] = {}
        self._ledger_open: float | None = None
        self.open_pnl = 0.0
        self.feed: dict[str, Any] = {"ready": False}

    def _load_models(self) -> None:
        day_path = RESULTS_DIR / "model_daytrade.joblib"
        swing_path = RESULTS_DIR / "model_swing.joblib"
        if day_path.exists():
            self.daytrade = joblib.load(day_path)
        if swing_path.exists():
            self.swing = joblib.load(swing_path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            wins = sum(1 for t in self.trades if t.pnl > 0)
            net = sum(t.pnl for t in self.trades)
            today_pnl = sum(t.pnl for t in self.trades if t.entry_time.date() == _today())
            sig = None if self.signal is None else self.signal.to_dict()
            pos = None
            if self.position is not None:
                pos = {
                    "side": self.position.side.value,
                    "entry": self.position.entry,
                    "stop": self.position.stop,
                    "take": self.position.take,
                    "time": self.position.time.isoformat(),
                    "contracts": self.position.contracts,
                    "ticket": self.position.ticket,
                    "reason": self.position.reason,
                    "mark": None if self.quote is None else self.quote.get("last"),
                    "pnl": self.open_pnl,
                }
            return {
                "running": self.running,
                "armed": self.armed,
                "done": False,
                "error": self.error,
                "config": self.cfg.name,
                "case": CASE,
                "timeframe": TIMEFRAME,
                "source": "mt5",
                "order_mode": self.order_mode.value,
                "interval_sec": self.cfg.execution.in_position_poll_ms / 1000.0,
                "last_tick": None if self.last_tick is None else self.last_tick.isoformat(),
                "last_bar_time": None if self.last_bar is None else self.last_bar.isoformat(),
                "cursor": len(self.candles),
                "n_bars": len(self.candles),
                "initial_bank": self.initial_bank,
                "lot": "scaled",
                "bank": self.bank,
                "net_pnl": net,
                "today_pnl": today_pnl,
                "avg_daily": today_pnl,
                "n_days": 1,
                "n_trades": len(self.trades),
                "n_wins": wins,
                "win_rate": (wins / len(self.trades) * 100.0) if self.trades else 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "contracts": self.contracts,
                "max_contracts": 10,
                "signal": sig,
                "position": pos,
                "pending": None,
                "trades": [t.to_dict() for t in self.trades[-40:]],
                "equity": self.equity[-200:],
                "daily": [{"t": _today().isoformat(), "pnl": today_pnl}],
                "signals": self.signals[-20:],
                "candles": self.candles[-80:],
                "quote": self.quote,
                "open_pnl": self.open_pnl,
                "skip_reason": self.skip_reason,
                "tick_msc": self.last_tick_msc or None,
                "wait_reason": self.wait_reason,
                "next_gold": next_gold_window(),
                "playbook": DEMO_PLAYBOOK,
                "mode": self.order_mode.value,
                "feed": self.feed,
                "mt5": self._mt5_payload,
                "can_send": self._can_send(),
                "context": None if self.context is None else self.context.to_dict(),
            }

    def _can_send(self) -> bool:
        if self.order_mode is OrderMode.PAPER:
            return False
        if self.order_mode is OrderMode.MT5:
            return bool(self._mt5_payload.get("demo"))
        return True

    def start(self, order_mode: str = "mt5", source: str = "mt5") -> dict[str, Any]:
        del source
        self.order_mode = OrderMode(order_mode)
        self.armed = True
        self.running = True
        self.error = None
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="koletivo-live", daemon=True)
            self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self.armed = False
        self.running = False
        self._stop.set()
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        self.stop()
        self.trades.clear()
        self.signals.clear()
        self.position = None
        self.signal = None
        self.bank = self.initial_bank
        return self.snapshot()

    def disconnect(self) -> None:
        self.stop()
        if self.broker is not None:
            try:
                self.broker.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self.broker = None

    def arm_session(self) -> None:
        """Swing D-1 before the bell; persist context for the whole session."""
        m5 = self._closed("m5", 120)
        if not m5:
            return
        days = group_days(m5)
        prev = days[-2] if len(days) >= 2 and days[-1][0].timestamp.date() == _today() else days[-1]
        if prev[-1].timestamp.date() == _today() and len(days) >= 2:
            prev = days[-2]
        self.context = self.swing.predict(prev)
        self.journal.save_context(_today(), self.context)

    def _loop(self) -> None:
        while not self._stop.is_set():
            in_pos = self.position is not None
            delay = (
                self.cfg.execution.in_position_poll_ms / 1000.0
                if in_pos
                else self.cfg.execution.idle_poll_ms / 1000.0
            )
            try:
                self._pulse()
            except Exception as exc:  # noqa: BLE001
                self.error = redact_text(str(exc))
                self._reconnect()
            time.sleep(max(0.01, delay))

    def _reconnect(self) -> None:
        try:
            if self.broker is not None:
                self.broker.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.broker = None
        time.sleep(1.0)

    def _ensure_broker(self) -> bool:
        if self.broker is not None:
            return True
        broker = Mt5Broker(
            self.cfg.mt5.symbol,
            self.cfg.mt5.magic,
            self.cfg.mt5.deviation,
            self.cfg.mt5.filling,
            self.cfg.mt5.comment,
        )
        try:
            broker.connect(select_symbol=True, **env_credentials())
        except Exception as exc:  # noqa: BLE001
            self.error = redact_text(str(exc))
            self.feed = {"ready": False, "error": self.error}
            self.wait_reason = "aguardando_login"
            return False
        symbol = resolve_symbol(broker) or broker.symbol
        broker.use_symbol(symbol)
        self.symbol = symbol
        self.broker = broker
        self.feed = {"ready": True, "symbol": symbol, "origin": "mt5"}
        return True

    def _closed(self, timeframe: str, count: int) -> list[Candle]:
        if self.broker is None:
            return []
        return self.broker.last_closed_candles(self.symbol, timeframe, count)

    def _limits_ok(self) -> str | None:
        today_trades = [t for t in self.trades if t.entry_time.date() == _today()]
        if len(today_trades) >= int(self.cfg.risk.max_trades_per_day):
            return "limite_de_operacoes"
        loss_cap = float(self.cfg.risk.daily_loss_points)
        if loss_cap > 0:
            lost = sum(min(0.0, t.points) for t in today_trades)
            if abs(lost) >= loss_cap:
                return "limite_de_perda_diaria"
        return None

    def _pulse(self) -> None:
        if not self.running:
            self.wait_reason = "pausado"
            return
        if not self._ensure_broker() or self.broker is None:
            return
        acc = self.broker.account_payload()
        term = self.broker.terminal_payload()
        self._mt5_payload = {
            "ready": True,
            "demo": acc.get("demo"),
            "login": acc.get("login"),
            "server": acc.get("server"),
            "symbol": self.symbol,
            "filling": self.broker.filling,
            "trade_allowed": term.get("trade_allowed"),
            "balance": acc.get("balance"),
            "equity": acc.get("equity"),
            "credit": acc.get("credit"),
            "profit": acc.get("profit"),
            "margin_free": acc.get("margin_free"),
            "bank": acc.get("bank"),
        }
        if acc.get("bank"):
            self.bank = float(acc["bank"])
            self.contracts = contracts_for_bank(self.bank)
        quote = self.broker.quote()
        if quote:
            self.quote = quote
            self.last_tick = datetime.now()
            self.last_tick_msc = int(quote.get("time_msc") or 0)
        now = datetime.now()
        in_pos = self.position is not None
        self.wait_reason = session_wait_reason(
            connected=True,
            account=bool(acc.get("account")),
            demo=acc.get("demo"),
            symbol=self.symbol,
            trade_allowed=bool(term.get("trade_allowed")),
            now=now,
            last_bar=self.last_bar,
            in_position=in_pos,
            session=self.session,
        )
        if self.order_mode is OrderMode.MT5 and acc.get("demo") is False:
            self.wait_reason = "conta_real"
        if self.context is None:
            self.arm_session()
            if self._mt5_payload.get("balance") is not None:
                self._ledger_open = float(self._mt5_payload["balance"])
                self.journal.append_ledger(
                    _today(),
                    {
                        "ts": now.isoformat(timespec="seconds"),
                        "event": "session_open",
                        "balance_open": self._ledger_open,
                        "equity": acc.get("equity"),
                        "balance": acc.get("balance"),
                        "margin_free": acc.get("margin_free"),
                        "open_pnl": 0,
                        "closed_pnl_day": 0,
                        "n_trades": 0,
                        "n_wins": 0,
                    },
                )
        if in_pos:
            self._manage_position(quote)
            return
        if not self.armed:
            return
        m5 = self._closed("m5", 8)
        if not m5:
            self.wait_reason = "aguardando_candle"
            return
        last = m5[-1]
        self.last_bar = last.timestamp
        self.candles = [
            {
                "t": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in m5
        ]
        bar_id = _bar_id(last.timestamp)
        if bar_id in self.decided_bars:
            return
        if not self.session.allows(now) and not self.session.allows(last.timestamp):
            self.skip_reason = self.wait_reason
            return
        self.decided_bars.add(bar_id)
        m1 = self._closed("m1", LOOKBACK_M1 + 5)
        window = m1[-LOOKBACK_M1:]
        if len(window) < LOOKBACK_M1:
            self.skip_reason = "sem_m1"
            return
        stop, take = self.risk.levels(Side.BUY, last.close)
        prior_m5 = m5[:-1][-6:]
        raw = self.daytrade.predict(window, last.close, stop, take, prior_m5)
        minutes = self.session.minutes_from_open(last.timestamp)
        boost = fib_boost(
            raw.side,
            last.close,
            window,
            tick=float(self.cfg.instrument.tick_size),
        )
        fused = fuse_signals(
            raw,
            self.context,
            swing_weight=self.cfg.filters.swing_weight,
            min_hit_pct=self.cfg.filters.min_hit_pct,
            minutes_from_open=minutes,
            first_block_minutes=self.cfg.filters.first_block_minutes,
            fib_weight=self.cfg.filters.fib_weight,
            fib_boost=boost,
        )
        if fused.side in {Side.BUY, Side.SELL}:
            tick = float(self.cfg.instrument.tick_size)
            offset = float(getattr(self.cfg.execution, "offset_points", 0.0) or 0.0)
            entry = round_to_tick(last.close + offset, tick)
            fused.stop, fused.take = self.risk.levels(fused.side, entry)
            fused.entry = entry
        self.signal = fused
        row = fused.to_dict()
        row["t"] = now.isoformat()
        self.signals.insert(0, row)
        limit = self._limits_ok()
        sent = False
        if fused.side is Side.HOLD:
            self.skip_reason = fused.reason
        elif limit:
            self.skip_reason = limit
            fused.side = Side.HOLD
            fused.phrase = phrase_for(Side.HOLD, fused.chart_type, fused.hit_pct, reason="low_hit")
            fused.reason = limit
            self.signal = fused
        else:
            self.skip_reason = None
            sent = self._send_signal(fused, bar_id)
        fused.sent = sent
        self.journal.append_order(
            _today(),
            fused,
            mode=self.order_mode.value,
            symbol=self.symbol,
            bar_id=bar_id,
            contracts=self.contracts,
            ts=now,
        )

    def _send_signal(self, signal: Signal, bar_id: str) -> bool:
        del bar_id
        if signal.side is Side.HOLD:
            return False
        if self.order_mode is OrderMode.PAPER or not self._can_send():
            self.position = Position(
                side=signal.side,
                entry=signal.entry,
                stop=signal.stop,
                take=signal.take,
                time=datetime.now(),
                contracts=self.contracts,
                reason=signal.reason,
                extreme=signal.entry,
                orig_stop=signal.stop,
                orig_take=signal.take,
            )
            return False
        if self.broker is None:
            return False
        try:
            self.broker.ensure_algo_trading()
            result = self.broker.send(signal, float(self.contracts))
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return False
        ticket = result.get("order") or result.get("deal")
        fill = float(result.get("price") or signal.entry)
        self.position = Position(
            side=signal.side,
            entry=fill,
            stop=signal.stop,
            take=signal.take,
            time=datetime.now(),
            contracts=self.contracts,
            ticket=int(ticket) if ticket else None,
            reason=signal.reason,
            extreme=fill,
            orig_stop=signal.stop,
            orig_take=signal.take,
        )
        signal.sent = True
        return True

    def _manage_position(self, quote: dict[str, Any] | None) -> None:
        pos = self.position
        if pos is None:
            return
        mark = None if quote is None else float(quote.get("last") or 0)
        if not mark and self.broker is not None:
            try:
                ticks = self.broker.ticks_since(self.last_tick_msc, 64)
            except Exception:  # noqa: BLE001
                ticks = []
            if ticks:
                ts, px = ticks[-1]
                mark = px
                self.last_tick = ts
        if mark:
            buy = pos.side is Side.BUY
            self.open_pnl = (mark - pos.entry if buy else pos.entry - mark) * self.cfg.account.point_value * pos.contracts
            new_stop, new_take, extreme = protect_levels(
                buy=buy,
                entry=pos.entry,
                stop=pos.stop,
                take=pos.take,
                orig_stop=pos.orig_stop,
                mark=mark,
                extreme=pos.extreme or pos.entry,
                tick=float(self.cfg.instrument.tick_size),
                be_trigger=self.cfg.risk.be_trigger_points,
                be_lock=self.cfg.risk.be_lock_points,
                trail_enabled=self.cfg.risk.trailing_enabled,
                trail_trigger=self.cfg.risk.trailing_trigger_points,
                trail_distance=self.cfg.risk.trailing_distance_points,
                orig_take=pos.orig_take,
                near_gain_points=self.cfg.risk.invalidate_tp_points,
            )
            pos.extreme = extreme
            changed = abs(new_stop - pos.stop) >= self.cfg.instrument.tick_size
            now_m = time.monotonic()
            if (
                changed
                and pos.ticket
                and self._can_send()
                and self.broker is not None
                and (now_m - self._last_sl_modify) * 1000 >= self.cfg.execution.sl_modify_min_ms
            ):
                try:
                    self.broker.modify_sltp(pos.ticket, new_stop, new_take)
                    self._last_sl_modify = now_m
                    pos.stop = new_stop
                    pos.take = new_take
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
            else:
                pos.stop = new_stop
                pos.take = new_take
            hit_stop = mark <= pos.stop if buy else mark >= pos.stop
            hit_gain = mark >= pos.take if buy else mark <= pos.take
            if hit_stop or hit_gain:
                self._close_local(TradeResult.STOP if hit_stop else TradeResult.GAIN, mark)
                return
        if self.broker is not None and self._can_send() and pos.ticket:
            live = self.broker.open_positions()
            if not any(int(p["ticket"]) == int(pos.ticket) for p in live):
                deal = self.broker.closing_deal(pos.ticket) or {}
                px = float(deal.get("price") or mark or pos.entry)
                result = TradeResult.GAIN if float(deal.get("profit") or 0) >= 0 else TradeResult.STOP
                self._close_local(result, px, ticket=pos.ticket)

    def _close_local(self, result: TradeResult, exit_px: float, ticket: int | None = None) -> None:
        pos = self.position
        if pos is None:
            return
        buy = pos.side is Side.BUY
        points = (exit_px - pos.entry) if buy else (pos.entry - exit_px)
        pnl = points * self.cfg.account.point_value * pos.contracts - self.cfg.account.contract_cost * pos.contracts
        trade = Trade(
            side=pos.side,
            entry_time=pos.time,
            exit_time=datetime.now(),
            entry=pos.entry,
            exit=exit_px,
            stop=pos.stop,
            take=pos.take,
            points=points,
            pnl=pnl,
            result=result,
            reason=pos.reason,
            contracts=pos.contracts,
            ticket=ticket or pos.ticket,
        )
        self.trades.append(trade)
        self.bank += pnl
        self.equity.append({"t": datetime.now().isoformat(), "bank": self.bank})
        if self.order_mode is not OrderMode.PAPER:
            self.journal.append_trade(_today(), trade)
            acc = self._mt5_payload
            wins = sum(1 for t in self.trades if t.pnl > 0 and t.entry_time.date() == _today())
            today_pnl = sum(t.pnl for t in self.trades if t.entry_time.date() == _today())
            self.journal.append_ledger(
                _today(),
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "fill",
                    "balance_open": self._ledger_open,
                    "equity": acc.get("equity"),
                    "balance": acc.get("balance"),
                    "margin_free": acc.get("margin_free"),
                    "open_pnl": 0,
                    "closed_pnl_day": today_pnl,
                    "n_trades": len([t for t in self.trades if t.entry_time.date() == _today()]),
                    "n_wins": wins,
                },
            )
        self.position = None
        self.open_pnl = 0.0


_ENGINE: LiveEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_live_engine() -> LiveEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = LiveEngine()
        return _ENGINE
