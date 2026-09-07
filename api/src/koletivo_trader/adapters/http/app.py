from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from koletivo_trader.adapters.journal import CsvJournal
from koletivo_trader.adapters.mt5.session import mt5_check as run_mt5_check
from koletivo_trader.application.orchestrator import get_live_engine
from koletivo_trader.application.replay import get_replay_engine, replay_meta
from koletivo_trader.domain.product import BANKS, CASE, TIMEFRAME
from koletivo_trader.paths import RESULTS_DIR, UI_DIST, UI_PUBLIC


class RealtimeStart(BaseModel):
    source: str | None = "mt5"
    order_mode: str | None = "mt5"


class LiveStart(BaseModel):
    case: str = CASE
    timeframe: str = TIMEFRAME
    initial_bank: float = 1000
    start: str | None = None
    end: str | None = None
    source: str = "paper"
    lot: str | None = None
    interval_sec: float = Field(default=1.0, ge=0.0, le=600.0)


class UiLogIn(BaseModel):
    line: str
    extra: Any | None = None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine = get_live_engine()
        try:
            engine.start(order_mode="paper")
        except Exception:  # noqa: BLE001
            pass
        yield
        engine.disconnect()
        get_replay_engine().stop()

    app = FastAPI(title="Koletivo Trader", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/studies")
    def studies() -> Any:
        path = RESULTS_DIR / "studies.json"
        if not path.exists():
            path = UI_PUBLIC / "studies.json"
        if not path.exists():
            raise HTTPException(404, "Estudo ainda não gerado. Rode python -m koletivo_trader train")
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/realtime")
    @app.get("/api/realtime/status")
    def realtime_status() -> dict[str, Any]:
        return get_live_engine().snapshot()

    @app.post("/api/realtime/start")
    def realtime_start(body: RealtimeStart) -> dict[str, Any]:
        return get_live_engine().start(order_mode=body.order_mode or "mt5", source=body.source or "mt5")

    @app.post("/api/realtime/stop")
    def realtime_stop() -> dict[str, Any]:
        return get_live_engine().stop()

    @app.post("/api/realtime/reset")
    def realtime_reset() -> dict[str, Any]:
        return get_live_engine().reset()

    @app.post("/api/realtime/source")
    def realtime_source(body: RealtimeStart) -> dict[str, Any]:
        engine = get_live_engine()
        if body.order_mode:
            return engine.start(order_mode=body.order_mode, source=body.source or "mt5")
        return engine.snapshot()

    @app.post("/api/realtime/ui-log")
    def ui_log(body: UiLogIn) -> dict[str, bool]:
        del body
        return {"ok": True}

    @app.get("/api/live/meta")
    def live_meta(timeframe: str = TIMEFRAME) -> dict[str, Any]:
        return replay_meta(TIMEFRAME)

    @app.get("/api/live")
    def live_status() -> dict[str, Any]:
        return get_replay_engine().snapshot()

    @app.post("/api/live/start")
    def live_start(body: LiveStart) -> dict[str, Any]:
        start = body.start or "2026-08-17"
        end = body.end or "2026-08-21"
        bank = float(body.initial_bank)
        if bank not in BANKS:
            bank = min(BANKS, key=lambda item: abs(item - bank))
        return get_replay_engine().start(
            start=start,
            end=end,
            initial_bank=bank,
            timeframe=TIMEFRAME,
            case=CASE,
            lot=body.lot or "fixed",
        )

    @app.post("/api/live/stop")
    def live_stop() -> dict[str, Any]:
        return get_replay_engine().stop()

    @app.post("/api/live/reset")
    def live_reset() -> dict[str, Any]:
        return get_replay_engine().reset()

    @app.get("/api/journal/days")
    def journal_days() -> dict[str, Any]:
        store = CsvJournal()
        days = store.real_days()
        today = date.today().isoformat()
        first = store.first_real_day() or today
        return {"days": days, "first": first, "today": today}

    @app.get("/api/journal/{day}")
    def journal_day(day: str) -> dict[str, Any]:
        try:
            parsed = date.fromisoformat(day[:10])
        except ValueError as exc:
            raise HTTPException(400, "data inválida") from exc
        return CsvJournal().day_payload(parsed)

    if UI_DIST.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = UI_DIST / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            index = UI_DIST / "index.html"
            if index.exists():
                return FileResponse(index)
            raise HTTPException(404)

    return app


app = create_app()
