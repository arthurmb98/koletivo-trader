from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Koletivo Trader")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train", help="Treina swing + daytrade e busca parâmetros")
    train = sub.add_parser("study", help="Alias de train")
    replay = sub.add_parser("replay", help="Replay paper de uma janela CSV")
    replay.add_argument("--from", dest="start", default="2026-08-17")
    replay.add_argument("--to", dest="end", default="2026-08-21")
    serve = sub.add_parser("serve", help="Sobe a API FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--debug", action="store_true")
    sub.add_parser("mt5-check", help="Testa o terminal MT5 sem enviar ordem")
    sub.add_parser("charts", help="Recalcula curvas do estudo sem retreinar")
    args = parser.parse_args(argv)

    if args.cmd in {"train", "study"}:
        from koletivo_trader.application.study import train_models

        print(train_models())
        return
    if args.cmd == "replay":
        from koletivo_trader.application.replay import get_replay_engine

        engine = get_replay_engine()
        engine.start(start=args.start, end=args.end)
        if engine._thread:
            engine._thread.join()
        print(engine.snapshot().get("n_trades"), "trades")
        return
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "koletivo_trader.adapters.http.app:app",
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level="debug" if args.debug else "info",
        )
        return
    if args.cmd == "charts":
        from koletivo_trader.application.study import refresh_study_charts

        print(refresh_study_charts())
        return
    if args.cmd == "mt5-check":
        from koletivo_trader.adapters.mt5.session import mt5_check

        raise SystemExit(mt5_check())
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
