from __future__ import annotations

from koletivo_trader.adapters.mt5.session import (
    describe_initialize_failure,
    initialize_kwargs_attempts,
    python_api_disabled,
)


def test_initialize_attempts_attach_first(tmp_path, monkeypatch) -> None:
    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"mz")
    monkeypatch.setattr(
        "koletivo_trader.adapters.mt5.session.DEFAULT_TERMINAL_PATHS",
        (str(exe),),
    )
    attempts = initialize_kwargs_attempts(None, timeout=1000)
    assert attempts[0] == {"timeout": 1000}
    assert attempts[1] == {"timeout": 1000, "path": str(exe)}


def test_initialize_attempts_skips_missing(monkeypatch) -> None:
    monkeypatch.setattr("koletivo_trader.adapters.mt5.session.DEFAULT_TERMINAL_PATHS", ())
    assert initialize_kwargs_attempts(r"C:\nope\terminal64.exe") == [{"timeout": 20_000}]


def test_python_api_disabled_reads_common_ini(tmp_path, monkeypatch) -> None:
    ini = tmp_path / "MetaQuotes" / "Terminal" / "abc" / "config" / "common.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text("[Experts]\nEnabled=0\nApi=0\n", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert python_api_disabled() is True
    ini.write_text("[Experts]\nEnabled=1\nApi=1\n", encoding="utf-8")
    assert python_api_disabled() is False


def test_describe_initialize_failure_mentions_python_api(monkeypatch) -> None:
    monkeypatch.setattr("koletivo_trader.adapters.mt5.session.python_api_disabled", lambda: True)
    monkeypatch.setattr("koletivo_trader.adapters.mt5.session.window_looks_prd", lambda: True)
    text = describe_initialize_failure((-6, "Terminal: Authorization failed"))
    assert "API Python" in text
    assert "PRD" in text
    assert "-6" in text


def test_wait_reason_closed_before_gold_pad() -> None:
    from datetime import datetime

    from koletivo_trader.adapters.mt5.session import next_gold_window, session_wait_reason
    from koletivo_trader.domain.session import SessionFilter

    now = datetime(2026, 9, 9, 7, 29)
    reason = session_wait_reason(
        connected=True,
        account=True,
        demo=False,
        symbol="WINV26",
        trade_allowed=False,
        now=now,
        last_bar=datetime(2026, 9, 8, 18, 15),
        in_position=False,
        session=SessionFilter(),
    )
    assert reason == "mercado_fechado"
    assert next_gold_window(now) == "2026-09-09T09:10"


def test_wait_reason_live_pad_is_not_fora_do_ouro() -> None:
    from datetime import datetime

    from koletivo_trader.adapters.mt5.session import session_wait_reason
    from koletivo_trader.domain.session import SessionFilter

    now = datetime(2026, 9, 9, 9, 10)
    reason = session_wait_reason(
        connected=True,
        account=True,
        demo=True,
        symbol="WINV26",
        trade_allowed=True,
        now=now,
        last_bar=datetime(2026, 9, 9, 9, 5),
        in_position=False,
        session=SessionFilter(),
    )
    assert reason == "pronto"
