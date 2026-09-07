from __future__ import annotations

import re
import subprocess

from koletivo_trader.adapters.mt5.broker import _safe_request
from koletivo_trader.adapters.mt5.session import redact_text, strip_secrets
from koletivo_trader.paths import ROOT

_SKIP_PARTS = (".git/", "node_modules/", ".venv/", "datasets/", "journal/", "ui/dist/")
_TEXT_SUFFIX = {".py", ".yaml", ".yml", ".md", ".json", ".ts", ".tsx", ".env", ".example", ".mdc", ".txt"}


def test_dotenv_is_not_git_tracked() -> None:
    tracked = subprocess.check_output(["git", "ls-files", ".env", ".env.local"], cwd=ROOT, text=True)
    assert tracked.strip() == ""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text
    assert "!.env.example" in text


def test_env_example_has_empty_password() -> None:
    assigned = [
        line
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.startswith("MT5_PASSWORD=")
    ]
    assert assigned == ["MT5_PASSWORD="]


def test_strip_secrets_drops_password() -> None:
    assert strip_secrets({"login": "1", "password": "secret", "server": "DEMO"}) == {
        "login": "1",
        "server": "DEMO",
    }
    assert "password" not in _safe_request({"symbol": "WIN$", "password": "secret"})


def test_redact_text_masks_env_password(monkeypatch) -> None:
    monkeypatch.setenv("MT5_PASSWORD", "hunter2")
    assert redact_text("login failed hunter2 on demo") == "login failed *** on demo"


def test_no_filled_mt5_password_in_tree() -> None:
    pattern = re.compile(r"^\s*MT5_PASSWORD\s*=\s*(.+)$")
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel for part in _SKIP_PARTS):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIX and path.name not in {".env", ".env.example"}:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            value = match.group(1).strip().strip("'").strip('"')
            if value and not value.startswith("#"):
                hits.append(f"{rel}: {line.strip()}")
    assert hits == []
