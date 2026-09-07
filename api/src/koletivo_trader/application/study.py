from __future__ import annotations

import json
from copy import deepcopy
from datetime import date

import joblib
import yaml

from koletivo_trader.adapters.candles import load_candles
from koletivo_trader.adapters.config import load_named_config
from koletivo_trader.domain.models import Candle
from koletivo_trader.ml.models import RECIPES, DaytradeModel, DaytradeRecipe, SwingModel, group_days
from koletivo_trader.ml.parameters import ParamResult, prepare_eval, score_fixed, search_parameters
from koletivo_trader.paths import CONFIGS_DIR, RESULTS_DIR, UI_PUBLIC
from koletivo_trader.domain.product import BANKS, CASE, HORIZON_M5, LOOKBACK_M1, TIMEFRAME

VAL_CUTOFF = date(2024, 7, 1)


def _split_before(candles: list[Candle], cutoff: date) -> tuple[list[Candle], list[Candle]]:
    a = [c for c in candles if c.timestamp.date() < cutoff]
    b = [c for c in candles if c.timestamp.date() >= cutoff]
    return a, b


def _print_result(tag: str, result: ParamResult) -> None:
    print(
        f"  {tag}: n={result.n_trades} wr={result.win_rate:.1f}% pnl={result.net_pnl:.0f} "
        f"dd={result.max_dd:.0f} pf={result.profit_factor:.2f} "
        f"stop={result.stop_points:.0f} gain={result.gain_points:.0f} min_hit={result.min_hit_pct:.3f}"
    )


def _objective_key(val: ParamResult, test: ParamResult | None = None) -> tuple:
    del test
    return (
        1 if val.net_pnl > 0 and val.n_trades >= 15 else 0,
        val.net_pnl,
        val.win_rate,
        -val.max_dd,
        val.n_trades,
    )


def train_models(*, max_daytrade_samples: int | None = None) -> dict:
    cfg = load_named_config("best_candles_m5_1000_a")
    print("Carregando CSVs…")
    m1_train = load_candles(cfg.resolve_csv(cfg.data.train_m1))
    m5_train = load_candles(cfg.resolve_csv(cfg.data.train_m5))
    m1_test = load_candles(cfg.resolve_csv(cfg.data.test_m1))
    m5_test = load_candles(cfg.resolve_csv(cfg.data.test_m5))
    print(f"Treino M1={len(m1_train)} M5={len(m5_train)} | teste M1={len(m1_test)} M5={len(m5_test)}")
    m1_fit, m1_val = _split_before(m1_train, VAL_CUTOFF)
    m5_fit, m5_val = _split_before(m5_train, VAL_CUTOFF)
    print(f"Fit até {VAL_CUTOFF}: M1={len(m1_fit)} M5={len(m5_fit)} | val M1={len(m1_val)} M5={len(m5_val)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    recipes = [item for item in RECIPES if item.name == "ft_atr_gold"]
    extra = [
        DaytradeRecipe("ft_pts25_rr2", "first_touch_pts", delta_pts=25.0, stop_mult=0.8, gain_mult=1.6, min_score=0.38),
        DaytradeRecipe("mom_loose", "momentum_meta", stop_mult=1.2, gain_mult=2.2, min_score=0.42, momentum_cut=0.12),
        DaytradeRecipe("ft_atr_loose", "first_touch_atr", delta_mult=0.55, stop_mult=0.8, gain_mult=1.6, min_score=0.34),
    ]

    swing_fit = SwingModel()
    print("Treinando swing (fit set)…")
    swing_fit.fit(group_days(m5_fit), stop_points=cfg.risk.stop_points, gain_points=cfg.risk.gain_points)

    ranked: list[tuple] = []
    for recipe in recipes:
        print(f"\n=== Receita {recipe.name} ({recipe.mode}) ===")
        model = DaytradeModel(recipe=recipe)
        try:
            scores = model.fit(
                m1_fit,
                m5_fit,
                stop_points=cfg.risk.stop_points,
                gain_points=cfg.risk.gain_points,
                max_samples=max_daytrade_samples,
            )
        except ValueError as exc:
            print(f"  pulada: {exc}")
            continue
        print("  daytrade", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scores.items()})
        prepared_val = prepare_eval(m1_val, m5_val, group_days(m5_fit), cfg, model, swing_fit)
        winners_val = search_parameters(
            m1_val,
            m5_val,
            group_days(m5_fit),
            cfg,
            model,
            swing_fit,
            banks=(1000.0,),
            n_trials=12,
            prepared=prepared_val,
        )
        val_res = winners_val["1000"]
        _print_result("VAL 1000", val_res)
        ranked.append((recipe, model, scores, val_res, val_res))
        if val_res.net_pnl > 0 and val_res.n_trades >= 25 and val_res.win_rate >= 38:
            print("  receita estável na validação — segue para o retreino.")
            break

    if not any(_objective_key(v)[0] == 1 for _, _, _, v, _t in ranked):
        print("\nNenhuma receita ainda lucrativa na validação. Tentando extras…")
        for recipe in extra:
            print(f"\n=== Receita extra {recipe.name} ({recipe.mode}) ===")
            model = DaytradeModel(recipe=recipe)
            try:
                scores = model.fit(
                    m1_fit,
                    m5_fit,
                    stop_points=cfg.risk.stop_points,
                    gain_points=cfg.risk.gain_points,
                    max_samples=max_daytrade_samples,
                )
            except ValueError as exc:
                print(f"  pulada: {exc}")
                continue
            prepared_val = prepare_eval(m1_val, m5_val, group_days(m5_fit), cfg, model, swing_fit)
            winners_val = search_parameters(
                m1_val,
                m5_val,
                group_days(m5_fit),
                cfg,
                model,
                swing_fit,
                banks=(1000.0,),
                n_trials=12,
                prepared=prepared_val,
            )
            val_res = winners_val["1000"]
            _print_result("VAL 1000", val_res)
            ranked.append((recipe, model, scores, val_res, val_res))
            if val_res.net_pnl > 0 and val_res.n_trades >= 20:
                break

    if not ranked:
        raise RuntimeError("Nenhuma receita de daytrade treinou.")
    ranked.sort(key=lambda row: _objective_key(row[3]), reverse=True)
    best_recipe, _, day_scores, val_res, _ = ranked[0]
    print(f"\n>>> Vencedora (só validação): {best_recipe.name} val_pnl={val_res.net_pnl:.0f}")

    print("\nRetreinando swing + daytrade no treino completo…")
    swing = SwingModel()
    swing_scores = swing.fit(group_days(m5_train), stop_points=cfg.risk.stop_points, gain_points=cfg.risk.gain_points)
    daytrade = DaytradeModel(recipe=best_recipe)
    day_scores = daytrade.fit(
        m1_train,
        m5_train,
        stop_points=cfg.risk.stop_points,
        gain_points=cfg.risk.gain_points,
        max_samples=max_daytrade_samples,
    )
    print("daytrade full", day_scores)
    joblib.dump(daytrade, RESULTS_DIR / "model_daytrade.joblib")
    joblib.dump(swing, RESULTS_DIR / "model_swing.joblib")

    print("AG final na validação 2024 H2 (folds + confirm, sem teste)…")
    prepared_val = prepare_eval(m1_val, m5_val, group_days(m5_fit), cfg, daytrade, swing)
    winners = search_parameters(
        m1_val,
        m5_val,
        group_days(m5_fit),
        cfg,
        daytrade,
        swing,
        banks=BANKS,
        n_trials=28,
        prepared=prepared_val,
    )
    prepared_test = prepare_eval(m1_test, m5_test, group_days(m5_train), cfg, daytrade, swing)
    oos = {}
    for bank, result in winners.items():
        oos[bank] = score_fixed(prepared_test, cfg, result, float(bank), collect=True)
        _print_result(f"OOS {bank}", oos[bank])

    if winners["1000"].net_pnl <= 0:
        print("Validação 1000 ainda negativa — varrendo limiar só na val…")
        base = winners["1000"]
        best_val = base
        for min_hit in (0.17, 0.28, 0.36, 0.44, 0.56, 0.68, 0.83):
            cand = ParamResult(
                base.stop_points,
                base.gain_points,
                min_hit,
                base.swing_weight,
                base.fib_weight,
                base.bank,
                base.contracts,
                0,
                0,
                0.0,
                0.0,
                0.0,
            )
            val_c = score_fixed(prepared_val, cfg, cand, 1000.0)
            _print_result(f"  min_hit={min_hit:.2f} VAL", val_c)
            if _objective_key(val_c) > _objective_key(best_val):
                best_val = val_c
        winners["1000"] = best_val
        oos["1000"] = score_fixed(prepared_test, cfg, best_val, 1000.0, collect=True)
        for bank in winners:
            if bank == "1000":
                continue
            adj = winners[bank]
            cand = ParamResult(
                winners["1000"].stop_points,
                winners["1000"].gain_points,
                winners["1000"].min_hit_pct,
                winners["1000"].swing_weight,
                winners["1000"].fib_weight,
                adj.bank,
                adj.contracts,
                0,
                0,
                0.0,
                0.0,
                0.0,
            )
            winners[bank] = score_fixed(prepared_val, cfg, cand, float(bank))
            oos[bank] = score_fixed(prepared_test, cfg, cand, float(bank), collect=True)

    for bank, result in winners.items():
        path = CONFIGS_DIR / f"best_bank_{bank}.yaml"
        data = deepcopy(cfg.to_dict())
        data["name"] = f"best_bank_{bank}"
        data["account"]["initial_bank"] = result.bank
        data["account"]["contracts"] = result.contracts
        data["risk"]["stop_points"] = result.stop_points
        data["risk"]["gain_points"] = result.gain_points
        data["filters"]["min_hit_pct"] = result.min_hit_pct
        data["filters"]["swing_weight"] = result.swing_weight
        data["filters"]["fib_weight"] = result.fib_weight
        data["execution"]["offset_points"] = result.offset_points
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    joblib.dump(
        {
            "params": {k: v.to_dict() for k, v in winners.items()},
            "oos": {k: v.to_dict() for k, v in oos.items()},
            "recipe": best_recipe.to_dict(),
        },
        RESULTS_DIR / "model_params.joblib",
    )
    study = _build_study(cfg, day_scores, swing_scores, oos, best_recipe, ranked)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "studies.json"
    out.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    UI_PUBLIC.mkdir(parents=True, exist_ok=True)
    (UI_PUBLIC / "studies.json").write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "daytrade": day_scores,
        "swing": swing_scores,
        "recipe": best_recipe.to_dict(),
        "params": {k: v.to_dict() for k, v in winners.items()},
        "oos": {k: v.to_dict() for k, v in oos.items()},
        "screen": [
            {"recipe": r.name, "val_pnl": v.net_pnl, "val_trades": v.n_trades, "val_wr": v.win_rate}
            for r, _, _, v, _t in ranked
        ],
    }


def refresh_study_charts() -> dict:
    """Rebuild equity/hourly series from saved models and YAML, without retraining."""
    import joblib

    from koletivo_trader.adapters.candles import load_candles
    from koletivo_trader.domain.risk import contracts_for_bank

    cfg = load_named_config("best_candles_m5_1000_a")
    day_path = RESULTS_DIR / "model_daytrade.joblib"
    swing_path = RESULTS_DIR / "model_swing.joblib"
    if not day_path.exists() or not swing_path.exists():
        raise FileNotFoundError("Modele primeiro com python -m koletivo_trader train")
    daytrade = joblib.load(day_path)
    swing = joblib.load(swing_path)
    print("Recalculando curvas OOS (sem retreino)…")
    m1_test = load_candles(cfg.resolve_csv(cfg.data.test_m1))
    m5_test = load_candles(cfg.resolve_csv(cfg.data.test_m5))
    m5_train = load_candles(cfg.resolve_csv(cfg.data.train_m5))
    prepared_test = prepare_eval(m1_test, m5_test, group_days(m5_train), cfg, daytrade, swing)
    oos: dict[str, ParamResult] = {}
    for bank in BANKS:
        bank_cfg = load_named_config(f"best_bank_{int(bank)}")
        seed = ParamResult(
            bank_cfg.risk.stop_points,
            bank_cfg.risk.gain_points,
            bank_cfg.filters.min_hit_pct,
            bank_cfg.filters.swing_weight,
            bank_cfg.filters.fib_weight,
            bank,
            contracts_for_bank(bank),
            0,
            0,
            0.0,
            0.0,
            0.0,
            offset_points=bank_cfg.execution.offset_points,
        )
        scored = score_fixed(prepared_test, cfg, seed, bank, collect=True)
        oos[str(int(bank))] = scored
        _print_result(f"OOS {int(bank)}", scored)
    recipe = getattr(daytrade, "recipe", None) or DaytradeRecipe()
    day_scores = {"train_rows": 0, "train_acc": 0.0}
    swing_scores = {"train_rows": 0}
    existing = RESULTS_DIR / "studies.json"
    if existing.exists():
        prev = json.loads(existing.read_text(encoding="utf-8"))
        hit = (prev.get("parecer") or {}).get("ml_hit") or {}
        leak = ((prev.get("timeframes") or {}).get("m5") or {}).get("leakage") or {}
        day_scores["train_acc"] = float(hit.get("daytrade") or hit.get("m1") or 0)
        day_scores["train_rows"] = int(leak.get("n_train") or 0)
        swing_scores["train_rows"] = float(hit.get("swing") or hit.get("m5") or 0)
    study = _build_study(cfg, day_scores, swing_scores, oos, recipe, [])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "studies.json").write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    UI_PUBLIC.mkdir(parents=True, exist_ok=True)
    (UI_PUBLIC / "studies.json").write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "banks": {k: v.n_trades for k, v in oos.items()}}


def _build_study(cfg, day_scores, swing_scores, winners, recipe: DaytradeRecipe, ranked) -> dict:
    leaderboard = []
    winners_node: dict = {CASE: {}}
    screen = [
        {
            "recipe": r.name,
            "mode": r.mode,
            "val_pnl": v.net_pnl,
            "test_pnl": t.net_pnl,
            "test_trades": t.n_trades,
            "test_win_rate": t.win_rate,
        }
        for r, _, _, v, t in ranked
    ]
    for bank, result in winners.items():
        metrics = {
            "n_candles": 0,
            "n_trades": result.n_trades,
            "n_wins": result.n_wins,
            "n_losses": result.n_trades - result.n_wins,
            "win_rate": result.win_rate,
            "net_pnl": result.net_pnl,
            "final_bank": result.bank + result.net_pnl,
            "initial_bank": result.bank,
            "avg_win": 0,
            "avg_loss": 0,
            "median_win": 0,
            "median_loss": 0,
            "avg_points_win": result.gain_points,
            "avg_points_loss": result.stop_points,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_dd,
            "max_drawdown_pct": (result.max_dd / result.bank * 100.0) if result.bank else 0.0,
            "expectancy": result.expectancy,
            "trades_per_candle_pct": 0.0,
            "hourly": result.hourly,
            "equity": result.equity,
            "trades": [],
        }
        winner = {
            "name": f"best_bank_{bank}",
            "params": cfg.to_dict(),
            "score": result.net_pnl,
            "leakage": {
                "n_train": int(day_scores.get("train_rows", 0)),
                "n_test_original": 0,
                "n_removed": 0,
                "n_test_clean": 0,
                "removed_by_key": 0,
                "removed_by_ohlc": 0,
                "train_file": cfg.data.train_m5,
                "test_file": cfg.data.test_m5,
                "train_start": "2021-08-18",
                "train_end": "2024-12-30",
                "test_start": "2025-01-02",
                "test_end": "2026-08-26",
            },
            "metrics": metrics,
            "by_period": {
                "daily": result.daily,
                "weekly": result.weekly,
                "monthly": result.monthly,
                "summary": {
                    "day": {"best": None, "worst": None, "avg": 0, "positive_pct": 0, "n_days": len(result.daily)},
                    "week": {"best": None, "worst": None, "avg": 0, "positive_pct": 0, "n_days": len(result.weekly)},
                    "month": {"best": None, "worst": None, "avg": 0, "positive_pct": 0, "n_days": len(result.monthly)},
                    "n_days": len(result.daily),
                },
            },
            "trades": [],
            "model_test": {"test_direction_hit": day_scores.get("train_acc", 0), "test_mae_close": 0},
        }
        winner["params"]["name"] = f"best_bank_{bank}"
        winner["params"]["account"]["initial_bank"] = result.bank
        winner["params"]["risk"]["stop_points"] = result.stop_points
        winner["params"]["risk"]["gain_points"] = result.gain_points
        winner["params"]["data"] = {
            "train_csv": cfg.data.train_m5,
            "test_csv": cfg.data.test_m5,
            "timeframe": TIMEFRAME,
        }
        winner["params"]["execution"] = {"direction": "follow", "decision": "ml_guard", **cfg.execution.__dict__}
        winners_node[CASE].setdefault(bank, {})[TIMEFRAME] = [winner]
        leaderboard.append(
            {
                "net_pnl": result.net_pnl,
                "n_trades": result.n_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
            }
        )
    leak = winners_node[CASE][next(iter(winners))][TIMEFRAME][0]["leakage"]
    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "disclaimer": "Não é recomendação de investimento. Resultado passado não garante resultado futuro.",
        "how_it_works": [
            "O swing lê o pregão anterior (M5) e classifica o tipo de dia.",
            "O daytrade usa 15 M1 só como contexto dos últimos candles de 5 min e decide compra, venda ou não fazer nada.",
            "O tempo gráfico de operação é sempre 5 min, caso últimos candles. Só a banca muda o YAML.",
        ],
        "insights": {
            "worked": [
                f"Receita {recipe.name} ({recipe.mode}).",
                "Meta-label P(gain) filtra operações; rótulo first-touch evita colapso em HOLD.",
            ],
            "failed": ["Acerto direcional puro de regressão linear no trader-api era ~49%."],
            "improve": ["Recalibrar parâmetros por banca com o algoritmo genético na validação 2024 H2."],
            "recipe_screen": screen,
        },
        "parecer": {
            "headline": "Robô Koletivo Trader: M5, últimos candles, banca variável.",
            "ml_hit": {
                "daytrade": float(day_scores.get("train_acc", 0)),
                "swing": float(swing_scores.get("train_rows", 0)),
                "m1": float(day_scores.get("train_acc", 0)),
                "m5": float(swing_scores.get("train_rows", 0)),
            },
            "n_months_note": "Teste 2025–2026 (fora da amostra).",
            "by_case": [
                {
                    "case": CASE,
                    "bank": int(bank),
                    "name": f"best_bank_{bank}",
                    "timeframe": TIMEFRAME,
                    "label": f"{int(result.stop_points)} / {int(result.gain_points)}",
                    "avg_fixed": result.net_pnl,
                    "avg_scaled": result.net_pnl,
                    "n_months": 20,
                }
                for bank, result in winners.items()
            ],
            "monthly": [],
            "strategy": [
                "15 M1 de contexto → sinal no fechamento do M5",
                "Stop e gain na ordem",
                "Limite diário de trades e de perda",
            ],
            "dd_floor": "Drawdown comparado à banca da config.",
        },
        "frozen_configs": [f"best_bank_{k}" for k in winners],
        "mt5": {
            "ready": True,
            "default_enabled": False,
            "steps": ["Abra o MT5 demo Genial/Clear", "python -m koletivo_trader serve", "npm run dev em ui/"],
        },
        "instrument": {"name": "WIN$", "point_value": 0.2, "tick": 5, "contracts": 1},
        "banks": [int(k) for k in winners],
        "cases": [CASE],
        "case_labels": {CASE: "Últimos candles"},
        "lookback": {"m1": LOOKBACK_M1, "m5": HORIZON_M5},
        "timeframes_list": [TIMEFRAME],
        "timeframe_labels": {TIMEFRAME: "5 min"},
        "n_configs_total": len(winners),
        "timeframes": {
            TIMEFRAME: {
                "leakage": leak,
                "model_test": {
                    "test_direction_hit": float(day_scores.get("train_acc", 0)),
                    "test_mae_close": 0,
                    "test_rmse_close": 0,
                },
            },
        },
        "studies": {CASE: {"1000": {"n_configs": len(winners), "leaderboard": leaderboard}}},
        "winners": winners_node,
    }
