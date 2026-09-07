from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from koletivo_trader.adapters.http.fnhttp import JsonHandler  # noqa: E402

LOCAL_ONLY = {"detail": "Ao vivo com MT5 só roda localmente (python -m koletivo_trader serve)."}


class handler(JsonHandler):
    def do_GET(self) -> None:
        self.send_json(400, LOCAL_ONLY)

    def do_POST(self) -> None:
        self.send_json(400, LOCAL_ONLY)
