from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor

from koletivo_trader.domain.fibonacci import AUX_WEIGHT_CAP, clamp_decider_weights
from koletivo_trader.domain.risk import round_to_tick

# stop, gain, min_hit, swing_w, fib_w, offset
# Offset stays 0: nonzero fill offset is not executable at the M5 close and
# inflates paper P&L (AG previously pinned LO at -50).
LO = np.array([40.0, 80.0, 0.17, 0.0, 0.0, 0.0])
HI = np.array([120.0, 240.0, 0.83, 0.40, 0.40, 0.0])
TICK = 5.0


def repair_genome(raw: np.ndarray, tick: float = TICK) -> np.ndarray:
    g = np.clip(np.asarray(raw, dtype=float), LO, HI)
    g[0] = round_to_tick(g[0], tick)
    g[1] = round_to_tick(g[1], tick)
    g[5] = round_to_tick(g[5], tick)
    if g[1] < g[0] * 1.5:
        g[1] = round_to_tick(min(HI[1], g[0] * 1.5), tick)
        if g[1] < g[0] * 1.5:
            g[0] = round_to_tick(g[1] / 1.5, tick)
    _, swing, fib = clamp_decider_weights(g[3], g[4])
    g[3], g[4] = swing, fib
    return g


def random_genome(rng: np.random.Generator, tick: float = TICK) -> np.ndarray:
    mode = int(rng.integers(0, 4))
    stop = float(rng.choice([40.0, 50.0, 60.0, 80.0, 100.0, 120.0]))
    gain = float(rng.choice([80.0, 100.0, 120.0, 150.0, 160.0, 200.0, 240.0]))
    min_hit = float(rng.uniform(LO[2], HI[2]))
    if mode == 0:
        swing, fib = 0.0, 0.0
    elif mode == 1:
        swing, fib = float(rng.uniform(0.05, AUX_WEIGHT_CAP)), 0.0
    elif mode == 2:
        swing, fib = 0.0, float(rng.uniform(0.05, AUX_WEIGHT_CAP))
    else:
        swing = float(rng.uniform(0.05, AUX_WEIGHT_CAP))
        fib = float(rng.uniform(0.0, max(0.0, AUX_WEIGHT_CAP - swing)))
    offset = 0.0
    return repair_genome(np.array([stop, gain, min_hit, swing, fib, offset]), tick)


def sbx(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, eta: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (Deb)."""
    c1, c2 = a.copy(), b.copy()
    for i in range(len(a)):
        if rng.random() > 0.9:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
        c2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
    return repair_genome(c1), repair_genome(c2)


def poly_mutate(g: np.ndarray, rng: np.random.Generator, eta: float = 20.0, p: float | None = None) -> np.ndarray:
    out = g.copy()
    prob = 1.0 / len(g) if p is None else p
    for i in range(len(g)):
        if rng.random() > prob:
            continue
        u = rng.random()
        span = HI[i] - LO[i]
        if span <= 0:
            continue
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        out[i] = g[i] + delta * span
    if rng.random() < 0.12:
        out[5] = 0.0
    return repair_genome(out)


def tournament(fitness: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[int(np.argmax(fitness[idx]))])


@dataclass
class GenomeResult:
    stop: float
    gain: float
    min_hit: float
    swing_weight: float
    fib_weight: float
    offset_points: float
    fitness: float

    def vector(self) -> np.ndarray:
        return np.array(
            [self.stop, self.gain, self.min_hit, self.swing_weight, self.fib_weight, self.offset_points],
            dtype=float,
        )


def unpack(g: np.ndarray) -> dict[str, float]:
    g = repair_genome(g)
    return {
        "stop": float(g[0]),
        "gain": float(g[1]),
        "min_hit": float(g[2]),
        "swing_weight": float(g[3]),
        "fib_weight": float(g[4]),
        "offset_points": float(g[5]),
    }


class _Surrogate:
    """Small MLP to rank unevaluated genomes. Not used for live trading."""

    def __init__(self) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            solver="adam",
            max_iter=400,
            random_state=7,
            early_stopping=True,
        )
        self.ready = False

    def fit(self, xs: list[np.ndarray], ys: list[float]) -> None:
        if len(xs) < 16:
            self.ready = False
            return
        x = np.vstack(xs)
        y = np.asarray(ys, dtype=float)
        scale = max(float(np.std(y)), 1.0)
        try:
            self.model.fit(x, y / scale)
            self.ready = True
            self._scale = scale
        except Exception:
            self.ready = False

    def rank(self, pop: list[np.ndarray]) -> np.ndarray:
        if not self.ready:
            return np.zeros(len(pop))
        pred = self.model.predict(np.vstack(pop)) * getattr(self, "_scale", 1.0)
        return np.asarray(pred, dtype=float)


def run_genetic_search(
    evaluate,
    *,
    population: int = 24,
    generations: int = 16,
    elite: int = 4,
    seed: int = 7,
    patience: int = 6,
) -> GenomeResult:
    """Real-coded GA (SBX + polynomial mutation + elitism + MLP surrogate infill)."""
    rng = np.random.default_rng(seed)
    pop = [random_genome(rng) for _ in range(population)]
    pop[0] = repair_genome(np.array([100.0, 200.0, 0.55, 0.0, 0.0, 0.0]))
    pop[1] = repair_genome(np.array([80.0, 160.0, 0.45, 0.15, 0.0, 0.0]))
    cache: dict[tuple, float] = {}
    xs: list[np.ndarray] = []
    ys: list[float] = []
    surrogate = _Surrogate()

    def true_fit(g: np.ndarray) -> float:
        key = tuple(np.round(g, 4))
        if key in cache:
            return cache[key]
        score = float(evaluate(unpack(g)))
        cache[key] = score
        xs.append(g.copy())
        ys.append(score)
        return score

    fit = np.array([true_fit(g) for g in pop], dtype=float)
    best_idx = int(np.argmax(fit))
    best_g, best_f = pop[best_idx].copy(), float(fit[best_idx])
    stall = 0
    for gen in range(generations):
        order = np.argsort(fit)[::-1]
        pop = [pop[i] for i in order]
        fit = fit[order]
        if fit[0] > best_f + 1e-6:
            best_g, best_f = pop[0].copy(), float(fit[0])
            stall = 0
        else:
            stall += 1
        print(
            f"  AG gen {gen + 1}/{generations} elite={best_f:.3f} "
            f"med={float(np.median(fit)):.3f} evals={len(cache)}",
            flush=True,
        )
        if stall >= patience:
            print("  AG early-stop (elite estável).", flush=True)
            break
        if gen >= 2:
            surrogate.fit(xs, ys)
        children: list[np.ndarray] = [p.copy() for p in pop[:elite]]
        while len(children) < population:
            i, j = tournament(fit, rng), tournament(fit, rng)
            c1, c2 = sbx(pop[i], pop[j], rng)
            children.append(poly_mutate(c1, rng))
            if len(children) < population:
                children.append(poly_mutate(c2, rng))
        if rng.random() < 0.15:
            children[-1] = random_genome(rng)
        if surrogate.ready and gen >= 2:
            extra = [poly_mutate(random_genome(rng), rng, p=0.4) for _ in range(population)]
            ranked = surrogate.rank(extra)
            for idx in np.argsort(ranked)[::-1][: max(2, elite)]:
                children[-(idx % elite + 1)] = extra[int(idx)]
        pop = children[:population]
        fit = np.array([true_fit(g) for g in pop], dtype=float)
    best = unpack(best_g)
    return GenomeResult(
        stop=best["stop"],
        gain=best["gain"],
        min_hit=best["min_hit"],
        swing_weight=best["swing_weight"],
        fib_weight=best["fib_weight"],
        offset_points=best["offset_points"],
        fitness=best_f,
    )
