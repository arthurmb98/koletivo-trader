from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from koletivo_trader.domain.copy import phrase_for
from koletivo_trader.domain.enums import ChartType, DayType, Side, TradeResult
from koletivo_trader.domain.market import classify_chart, classify_day, heuristic_side
from koletivo_trader.domain.models import Candle, SessionContext, Signal
from koletivo_trader.domain.session import SessionFilter
from koletivo_trader.ml.features import _linreg, atr_proxy, daytrade_features, slope_norm, swing_features
from koletivo_trader.ml.labels import (
    adaptive_barriers,
    direction_delta,
    first_touch_side,
    label_side,
    leak_free_windows,
    prior_m5_bars,
    same_day_m1,
    simulate_touch,
)

SIDE_LABELS = [Side.HOLD.value, Side.BUY.value, Side.SELL.value]


@dataclass
class DaytradeRecipe:
    name: str = "ft_atr_gold"
    mode: str = "first_touch_atr"
    delta_pts: float = 30.0
    delta_mult: float = 0.7
    stop_mult: float = 1.0
    gain_mult: float = 2.0
    gold_only: bool = True
    morning_only: bool = False
    min_score: float = 0.32
    momentum_cut: float = 0.18

    def to_dict(self) -> dict:
        return asdict(self)


RECIPES: list[DaytradeRecipe] = [
    DaytradeRecipe("ft_atr_gold", "first_touch_atr", stop_mult=1.0, gain_mult=2.0, min_score=0.40),
    DaytradeRecipe("ft_pts30", "first_touch_pts", delta_pts=30.0, stop_mult=1.0, gain_mult=2.0, min_score=0.40),
    DaytradeRecipe("ft_pts40_rr2", "first_touch_pts", delta_pts=40.0, stop_mult=0.9, gain_mult=1.8, min_score=0.42),
    DaytradeRecipe("momentum_meta", "momentum_meta", stop_mult=1.0, gain_mult=2.0, min_score=0.45, momentum_cut=0.22),
    DaytradeRecipe("agree_ft_mom", "agree", delta_pts=30.0, stop_mult=1.0, gain_mult=2.0, min_score=0.38),
    DaytradeRecipe("unique_adapt", "unique_gain", stop_mult=1.1, gain_mult=2.0, min_score=0.36),
    DaytradeRecipe("ft_morning", "first_touch_atr", stop_mult=1.0, gain_mult=2.0, morning_only=True, min_score=0.38),
    DaytradeRecipe("mom_tight", "momentum_meta", stop_mult=0.8, gain_mult=1.6, min_score=0.48, momentum_cut=0.30),
]


def _hgb(**kwargs) -> HistGradientBoostingClassifier:
    params = dict(
        max_depth=5,
        learning_rate=0.05,
        max_iter=220,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=7,
    )
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


def _maybe_balanced(clf: HistGradientBoostingClassifier) -> HistGradientBoostingClassifier:
    try:
        clf.set_params(class_weight="balanced")
    except (ValueError, TypeError):
        pass
    return clf


def _calibrate(base, x_val: np.ndarray, y_val: np.ndarray):
    if len(y_val) < 120 or len(np.unique(y_val)) < 2:
        return base
    method = "isotonic" if len(y_val) >= 400 else "sigmoid"
    cal = CalibratedClassifierCV(base, method=method, cv="prefit")
    try:
        cal.fit(x_val, y_val)
        return cal
    except Exception:
        return base


def _recency_weights(timestamps: list[datetime], half_life_days: float = 400.0) -> np.ndarray:
    if not timestamps:
        return np.asarray([], dtype=float)
    tmax = max(timestamps)
    return np.asarray([0.5 ** (max((tmax - ts).days, 0) / half_life_days) for ts in timestamps], dtype=float)


class DaytradeModel:
    """Two-stage: direction (or momentum) + meta-label P(gain)."""

    def __init__(self, recipe: DaytradeRecipe | None = None) -> None:
        self.recipe = recipe or DaytradeRecipe()
        self._dir = None
        self._meta = None
        self._chart = None
        self._chart_enc = LabelEncoder().fit([item.value for item in ChartType])
        self._fitted = False
        self.label_stop = 50.0
        self.label_gain = 100.0
        self.min_score = float(self.recipe.min_score)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _session(self) -> SessionFilter:
        return SessionFilter(gold_hours_only=self.recipe.gold_only)

    def _in_session(self, ts: datetime) -> bool:
        if not self._session().allows(ts):
            return False
        if self.recipe.morning_only and ts.hour >= 11:
            return False
        return True

    def _direction(self, window: list[Candle], entry: Candle, future: list[Candle]) -> Side:
        mode = self.recipe.mode
        stop, gain = adaptive_barriers(window, stop_mult=self.recipe.stop_mult, gain_mult=self.recipe.gain_mult)
        if mode == "unique_gain":
            return label_side(entry.open, stop, gain, future)
        if mode == "momentum_meta":
            sl = slope_norm(window)
            if sl >= self.recipe.momentum_cut:
                return Side.BUY
            if sl <= -self.recipe.momentum_cut:
                return Side.SELL
            return Side.HOLD
        if mode == "first_touch_pts":
            return first_touch_side(entry.open, self.recipe.delta_pts, future)
        delta = direction_delta(window, mult=self.recipe.delta_mult)
        ft = first_touch_side(entry.open, delta, future)
        if mode == "agree":
            sl = slope_norm(window)
            mom = Side.BUY if sl >= self.recipe.momentum_cut else Side.SELL if sl <= -self.recipe.momentum_cut else Side.HOLD
            return ft if ft is mom else Side.HOLD
        return ft

    def fit(
        self,
        m1: list[Candle],
        m5: list[Candle],
        *,
        stop_points: float,
        gain_points: float,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        del stop_points, gain_points
        windows = leak_free_windows(m1, m5)
        m5_index = {c.timestamp: i for i, c in enumerate(m5)}
        m1_index = {c.timestamp: i for i, c in enumerate(m1)}
        rows = []
        kept = 0
        for window, future, entry_bar in windows:
            if not self._in_session(entry_bar.timestamp):
                continue
            kept += 1
            if kept % 3:
                continue
            path = same_day_m1(m1, entry_bar.timestamp, index=m1_index, n=15)
            rows.append((window, path or future, entry_bar, prior_m5_bars(m5, entry_bar, index=m5_index)))
        if max_samples and len(rows) > max_samples:
            rng = np.random.default_rng(7)
            pick = rng.choice(len(rows), size=max_samples, replace=False)
            rows = [rows[i] for i in sorted(pick)]
        x_dir: list[np.ndarray] = []
        y_dir: list[int] = []
        t_dir: list[datetime] = []
        x_meta: list[np.ndarray] = []
        y_meta: list[int] = []
        t_meta: list[datetime] = []
        x_chart: list[np.ndarray] = []
        y_chart: list[str] = []
        n_buy = n_sell = n_hold = 0
        for window, future, entry_bar, prior in rows:
            stop, gain = adaptive_barriers(window, stop_mult=self.recipe.stop_mult, gain_mult=self.recipe.gain_mult)
            side = self._direction(window, entry_bar, future)
            feats = daytrade_features(window, prior)
            if len(future) >= 8:
                x_chart.append(feats)
                y_chart.append(classify_chart(future[:15]).value)
            if side is Side.BUY:
                n_buy += 1
                x_dir.append(feats)
                y_dir.append(1)
                t_dir.append(entry_bar.timestamp)
                outcome = simulate_touch(Side.BUY, entry_bar.open, stop, gain, future)
                if outcome is not TradeResult.NONE:
                    x_meta.append(np.concatenate([feats, [1.0]]))
                    y_meta.append(1 if outcome is TradeResult.GAIN else 0)
                    t_meta.append(entry_bar.timestamp)
            elif side is Side.SELL:
                n_sell += 1
                x_dir.append(feats)
                y_dir.append(0)
                t_dir.append(entry_bar.timestamp)
                outcome = simulate_touch(Side.SELL, entry_bar.open, stop, gain, future)
                if outcome is not TradeResult.NONE:
                    x_meta.append(np.concatenate([feats, [0.0]]))
                    y_meta.append(1 if outcome is TradeResult.GAIN else 0)
                    t_meta.append(entry_bar.timestamp)
            else:
                n_hold += 1
        print(f"  [{self.recipe.name}] BUY={n_buy} SELL={n_sell} HOLD={n_hold} meta={len(y_meta)}")
        if self.recipe.mode != "momentum_meta" and len(y_dir) < 80:
            raise ValueError("Treino daytrade insuficiente (direção).")
        self._dir = None
        dir_acc = 0.0
        if self.recipe.mode != "momentum_meta" and len(y_dir) >= 80:
            x_dir_arr = np.vstack(x_dir)
            y_dir_arr = np.asarray(y_dir, dtype=int)
            cut = max(int(len(y_dir_arr) * 0.85), 1)
            w = _recency_weights(t_dir)
            clf = _maybe_balanced(_hgb())
            try:
                clf.fit(x_dir_arr[:cut], y_dir_arr[:cut], sample_weight=w[:cut])
            except TypeError:
                clf.fit(x_dir_arr[:cut], y_dir_arr[:cut])
            self._dir = _calibrate(clf, x_dir_arr[cut:], y_dir_arr[cut:])
            dir_acc = float((self._dir.predict(x_dir_arr) == y_dir_arr).mean())
        self._meta = None
        meta_acc = 0.0
        meta_pos = 0.0
        if len(y_meta) >= 80:
            xm = np.vstack(x_meta)
            ym = np.asarray(y_meta, dtype=int)
            meta_pos = float(ym.mean())
            cut = max(int(len(ym) * 0.85), 1)
            w = _recency_weights(t_meta)
            clf = _hgb(max_depth=4, max_iter=160, min_samples_leaf=60)
            try:
                clf.fit(xm[:cut], ym[:cut], sample_weight=w[:cut])
            except TypeError:
                clf.fit(xm[:cut], ym[:cut])
            self._meta = _calibrate(clf, xm[cut:], ym[cut:])
            meta_acc = float((self._meta.predict(xm) == ym).mean())
        self._chart = None
        chart_acc = 0.0
        if len(y_chart) >= 80:
            xc = np.vstack(x_chart)
            yc = self._chart_enc.transform(y_chart)
            clf = _hgb(max_depth=4, max_iter=140, min_samples_leaf=40)
            clf.fit(xc, yc)
            self._chart = clf
            chart_acc = float((self._chart.predict(xc) == yc).mean())
        self._fitted = True
        total = n_buy + n_sell + n_hold
        return {
            "train_rows": float(total),
            "train_acc": dir_acc,
            "meta_acc": meta_acc,
            "meta_base_win": meta_pos,
            "hold_rate": n_hold / max(total, 1),
            "n_buy": float(n_buy),
            "n_sell": float(n_sell),
            "recipe": self.recipe.name,
            "chart_acc": chart_acc,
        }

    def score_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        p_buy = np.full(n, 0.5)
        if self._fitted and self._dir is not None:
            proba = self._dir.predict_proba(X)
            classes = list(self._dir.classes_)
            col = classes.index(1) if 1 in classes else min(1, proba.shape[1] - 1)
            p_buy = proba[:, col]
        p_sell = 1.0 - p_buy
        q_buy = np.full(n, 0.55)
        q_sell = np.full(n, 0.55)
        if self._meta is not None:
            ones = np.ones((n, 1))
            zeros = np.zeros((n, 1))
            pb = self._meta.predict_proba(np.hstack([X, ones]))
            ps = self._meta.predict_proba(np.hstack([X, zeros]))
            classes = list(self._meta.classes_)
            col = classes.index(1) if 1 in classes else min(1, pb.shape[1] - 1)
            q_buy = pb[:, col]
            q_sell = ps[:, col]
        return p_buy, p_sell, q_buy, q_sell

    def _proba(self, m1_window: list[Candle], prior_m5: list[Candle] | None) -> tuple[float, float, float, float, ChartType]:
        chart = classify_chart(m1_window)
        feats = daytrade_features(m1_window, prior_m5).reshape(1, -1)
        sl = slope_norm(m1_window)
        p_buy, p_sell, q_buy, q_sell = self.score_matrix(feats)
        pb, ps, qb, qs = float(p_buy[0]), float(p_sell[0]), float(q_buy[0]), float(q_sell[0])
        if self.recipe.mode == "momentum_meta":
            if sl >= self.recipe.momentum_cut:
                ps = 0.0
                pb = max(pb, 0.65)
            elif sl <= -self.recipe.momentum_cut:
                pb = 0.0
                ps = max(ps, 0.65)
            else:
                pb = ps = 0.0
        elif self.recipe.mode == "agree":
            if sl >= self.recipe.momentum_cut:
                ps *= 0.25
            elif sl <= -self.recipe.momentum_cut:
                pb *= 0.25
            else:
                pb *= 0.4
                ps *= 0.4
        return pb, ps, qb, qs, chart

    def predict(
        self,
        m1_window: list[Candle],
        entry: float,
        stop: float,
        take: float,
        prior_m5: list[Candle] | None = None,
    ) -> Signal:
        closes = np.array([c.close for c in m1_window], dtype=float)
        scale = max(atr_proxy(m1_window), 1e-9)
        slope, _ = _linreg(closes)
        slope_n = slope * max(len(closes), 1) / scale
        heuristic = heuristic_side(classify_chart(m1_window), slope_n)
        predicted_chart: ChartType | None = None
        if self._fitted:
            p_buy, p_sell, q_buy, q_sell, chart = self._proba(m1_window, prior_m5)
            if p_buy >= p_sell:
                side, p_dir, q = Side.BUY, p_buy, q_buy
            else:
                side, p_dir, q = Side.SELL, p_sell, q_sell
            hit = float(q)
            if p_dir < 0.52 or q < self.min_score:
                side = Side.HOLD
            if self._chart is not None:
                feats = daytrade_features(m1_window, prior_m5).reshape(1, -1)
                idx = int(self._chart.predict(feats)[0])
                predicted_chart = ChartType(self._chart_enc.inverse_transform([idx])[0])
        else:
            chart = classify_chart(m1_window)
            side = heuristic
            hit = 0.55 if side in {Side.BUY, Side.SELL} else 0.35
        phrase = phrase_for(side, chart, hit, predicted_chart=predicted_chart)
        return Signal(
            side=side,
            entry=entry,
            stop=stop,
            take=take,
            chart_type=chart,
            hit_pct=hit,
            phrase=phrase,
            reason="daytrade",
            predicted_chart_type=predicted_chart,
        )


class SwingModel:
    def __init__(self) -> None:
        self._side_clf = None
        self._day_clf = None
        self._side_enc = LabelEncoder().fit(SIDE_LABELS)
        self._day_enc = LabelEncoder().fit([item.value for item in DayType])
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        days_m5: list[list[Candle]],
        *,
        stop_points: float,
        gain_points: float,
    ) -> dict[str, float]:
        x_rows: list[np.ndarray] = []
        y_side: list[str] = []
        y_next: list[str] = []
        for i in range(len(days_m5) - 1):
            prev = days_m5[i]
            nxt = days_m5[i + 1]
            if len(prev) < 10 or len(nxt) < 4:
                continue
            delta = direction_delta(prev[-15:] if len(prev) >= 15 else prev, mult=0.5, floor=40.0, cap=120.0)
            open_px = nxt[0].open
            horizon = nxt[:3]
            side = label_side(open_px, stop_points, gain_points, horizon)
            if side is Side.HOLD:
                side = first_touch_side(open_px, delta, nxt[:6] if len(nxt) >= 6 else nxt)
            nxt_day = classify_day(nxt)
            if nxt_day in {DayType.VOLATILE, DayType.NON_TREND} and side in {Side.BUY, Side.SELL}:
                side = Side.HOLD
            x_rows.append(swing_features(prev))
            y_side.append(side.value)
            y_next.append(classify_day(nxt).value)
        if len(x_rows) < 40:
            raise ValueError("Treino swing insuficiente.")
        x = np.vstack(x_rows)
        self._side_clf = _maybe_balanced(_hgb(max_depth=4, max_iter=140, min_samples_leaf=20))
        y_enc = self._side_enc.transform(y_side)
        try:
            self._side_clf.fit(x, y_enc)
        except TypeError:
            self._side_clf = _hgb(max_depth=4, max_iter=140)
            self._side_clf.fit(x, y_enc)
        self._day_clf = _hgb(max_depth=4, max_iter=140, min_samples_leaf=20)
        self._day_clf.fit(x, self._day_enc.transform(y_next))
        self._fitted = True
        return {"train_rows": float(len(x_rows))}

    def predict(self, previous_m5: list[Candle]) -> SessionContext:
        day_type = classify_day(previous_m5)
        predicted = day_type
        swing_signal = Side.HOLD
        hit = 0.5
        if self._fitted and self._side_clf is not None and self._day_clf is not None:
            feats = swing_features(previous_m5).reshape(1, -1)
            proba = self._side_clf.predict_proba(feats)[0]
            classes = self._side_enc.inverse_transform(np.arange(len(proba)))
            by_name = {cls: float(p) for cls, p in zip(classes, proba)}
            buy_p = by_name.get(Side.BUY.value, 0.0)
            sell_p = by_name.get(Side.SELL.value, 0.0)
            hold_p = by_name.get(Side.HOLD.value, 0.0)
            if buy_p >= sell_p and buy_p >= hold_p and buy_p >= 0.38:
                swing_signal, hit = Side.BUY, buy_p
            elif sell_p >= hold_p and sell_p >= 0.38:
                swing_signal, hit = Side.SELL, sell_p
            else:
                swing_signal, hit = Side.HOLD, max(buy_p, sell_p)
            day_idx = int(self._day_clf.predict(feats)[0])
            predicted = DayType(self._day_enc.inverse_transform([day_idx])[0])
        else:
            if day_type is DayType.TREND_UP:
                swing_signal, hit = Side.BUY, 0.6
            elif day_type is DayType.TREND_DOWN:
                swing_signal, hit = Side.SELL, 0.6
            else:
                swing_signal, hit = Side.HOLD, 0.4
            predicted = day_type
        last = previous_m5[-1].timestamp if previous_m5 else datetime.now()
        if swing_signal is Side.HOLD:
            phrase = (
                f"D-1 classificado como {day_type.value.replace('_', ' ')}; "
                f"previsão para hoje: {predicted.value.replace('_', ' ')}. Sem viés de compra ou venda."
            )
        else:
            verb = "alta" if swing_signal is Side.BUY else "queda"
            phrase = (
                f"D-1 foi {day_type.value.replace('_', ' ')}. "
                f"Viés de {verb} para hoje ({predicted.value.replace('_', ' ')}), confiança {hit * 100:.0f}%."
            )
        return SessionContext(
            as_of=last,
            previous_date=last.date().isoformat(),
            swing_signal=swing_signal,
            day_type=day_type,
            predicted_day_type=predicted,
            swing_hit_pct=hit,
            phrase=phrase,
        )


def group_days(candles: list[Candle]) -> list[list[Candle]]:
    days: list[list[Candle]] = []
    current: list[Candle] = []
    current_day = None
    for candle in candles:
        day = candle.timestamp.date()
        if current_day is None:
            current_day = day
        if day != current_day:
            if current:
                days.append(current)
            current = [candle]
            current_day = day
        else:
            current.append(candle)
    if current:
        days.append(current)
    return days
