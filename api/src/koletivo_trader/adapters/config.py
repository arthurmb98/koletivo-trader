from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from koletivo_trader.paths import CONFIGS_DIR, ROOT


def _merge(dc_type: type, data: dict[str, Any] | None) -> Any:
    if data is None:
        return dc_type()
    hints = get_type_hints(dc_type)
    allowed = set(hints)
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Chaves desconhecidas em {dc_type.__name__}: {extra}")
    kwargs: dict[str, Any] = {}
    for name, typ in hints.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(typ) and isinstance(value, dict):
            kwargs[name] = _merge(typ, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


@dataclass
class AccountConfig:
    initial_bank: float = 1000.0
    contracts: int = 1
    point_value: float = 0.20
    contract_cost: float = 1.0


@dataclass
class InstrumentConfig:
    symbol: str = "WIN$"
    tick_size: int = 5


@dataclass
class DataConfig:
    train_m1: str = "datasets/WIN_1min_train.csv"
    test_m1: str = "datasets/WIN_1min_test.csv"
    train_m5: str = "datasets/WIN_5min_train.csv"
    test_m5: str = "datasets/WIN_5min_test.csv"


@dataclass
class RiskConfig:
    mode: str = "fixed"
    stop_points: float = 100.0
    gain_points: float = 200.0
    rr_ratio: float = 2.0
    atr_period: int = 14
    trailing_enabled: bool = True
    trailing_trigger_points: float = 60.0
    trailing_distance_points: float = 50.0
    be_trigger_points: float = 25.0
    be_lock_points: float = 10.0
    invalidate_tp_points: float = 30.0
    daily_loss_points: float = 0.0
    max_trades_per_day: int = 8


@dataclass
class FilterConfig:
    session_start: str = "09:15"
    session_end: str = "17:00"
    skip_lunch: bool = True
    lunch_start: str = "11:00"
    lunch_end: str = "14:30"
    gold_hours_only: bool = True
    min_hit_pct: float = 0.62
    swing_weight: float = 0.15
    fib_weight: float = 0.0
    first_block_minutes: float = 15.0


@dataclass
class ExecutionConfig:
    entry_mode: str = "market_open"
    offset_points: float = 0.0
    max_tick_age_ms: int = 1500
    in_position_poll_ms: int = 20
    idle_poll_ms: int = 100
    sl_modify_min_ms: int = 150


@dataclass
class MlConfig:
    min_train_rows: int = 200
    daytrade_lookback_m1: int = 15
    daytrade_lookback_m5: int = 3
    horizon_m5: int = 3
    horizon_m1: int = 15


@dataclass
class Mt5Config:
    enabled: bool = False
    symbol: str = "WIN$"
    magic: int = 20260907
    deviation: int = 20
    filling: str = "IOC"
    comment: str = "koletivo-trader"


@dataclass
class AppConfig:
    name: str = "default"
    account: AccountConfig = field(default_factory=AccountConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ml: MlConfig = field(default_factory=MlConfig)
    mt5: Mt5Config = field(default_factory=Mt5Config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return _merge(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_csv(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> AppConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig.from_dict(data)
    if not cfg.name or cfg.name == "default":
        cfg.name = path.stem
    return cfg


def load_named_config(name: str) -> AppConfig:
    stem = name.replace(".yaml", "")
    path = CONFIGS_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config {stem} não encontrada em {CONFIGS_DIR}")
    return load_config(path)


def load_bank_config(bank: float) -> AppConfig:
    """YAML da banca mais próxima (estudo / replay / ao vivo)."""
    from koletivo_trader.domain.product import BANKS

    chosen = min(BANKS, key=lambda item: abs(item - float(bank)))
    name = f"best_bank_{int(chosen)}"
    path = CONFIGS_DIR / f"{name}.yaml"
    if path.exists():
        return load_named_config(name)
    return load_named_config("best_candles_m5_1000_a")
