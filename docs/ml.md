# Machine learning — desenho, falhas e próximos passos

O produto está em [plano.md](plano.md). Esta nota cobre tensor de treino, pesos da fusão, o que já falhou e o backtest honesto.

## Objetivo

Daytrade com P(gain) calibrada no horário de ouro, custo `point_value=0.20` + `contract_cost=1`, máx. 8 trades/dia. Teste 2025–2026. Optuna só na val 2024 H2 (`2024-07-01`+ no CSV de treino).

O **daytrade é o decisor principal** (≥ 60% da fusão). Swing e Fibonacci são opcionais por peso, inclusive **zero**.

## Três entradas do daytrade

15 M1 = 3 blocos de 5 minutos. Cada bloco: 5 vetores `(t, O, H, L, C)` + 5 volumes no mesmo `t` → 20 OHLC + 5 vol.

1. **X preço** — 3 blocos, normalizados por ATR (sem índice absoluto).
2. **X volume** — 15 volumes crus +:
   - confirmação (preço e volume sobem)
   - exaustão (preço sobe, volume cai)
   - rompimento com RVOL ≥ 1,5× média 20
   - falso rompimento (rompe com RVOL baixo)
   - absorção (volume alto, range pequeno)
   - acumulação/distribuição (up-volume vs down-volume)
   - liquidez (z-score)
3. **Futuro só no treino** — 3 M5 (15 M1) para Y. Inferência nunca vê isso.

Pré-rótulo: `classify_chart` nos 15 M1 passados → feature one-hot; nos 15 M1 futuros → **alvo** `chart_type_future`. Swing: `classify_day` no D-1 (feature) e no D (alvo de `predicted_day_type`).

## Saídas

- Direção / HOLD: two-stage (BUY vs SELL limpo + meta P(gain)). Não voltar a 3 classes com HOLD majoritário.
- `chart_type_past` (rótulo do X) e `predicted_chart_type` (cabeça supervisionada no Y futuro).
- `hit_pct` = P(gain) do meta, depois da fusão ponderada.

## Fusão e Optuna (pesos)

```
w_swing, w_fib ∈ [0, 0.4]
w_swing + w_fib ≤ 0.4
w_day = 1 - w_swing - w_fib   # ≥ 0.6
hit_final = w_day * hit_day + w_swing * boost_swing + w_fib * boost_fib
```

- Peso 0: o decisor não altera `hit_final` nem veta (primeiros 15 min sem swing se `w_swing = 0`).
- Optuna sugere `swing_weight` e `fib_weight` com a restrição da soma (ex.: amostrar `w_swing`, depois `w_fib` em `[0, 0.4 - w_swing]`). Pode descobrir que **só daytrade** (`0, 0`) ganha, ou que um pouco de swing/Fib sobe o acerto.
- Não gerar `model_params.joblib` como “modelo”; persistir só `configs/best_bank_{500,1000,5000}.yaml`.
- Também: stop, gain (RR ≥ 1,5), `min_hit_pct`.

Fibonacci: confluência 38,2/50/61,8 na perna da sessão; `boost_fib` alto se o preço está no nível a favor do lado do daytrade, baixo se está longe ou contra. Distância em ticks de 5 pts.

## Pipeline atual no código (a substituir)

Arquivos: `ml/{labels,features,models,parameters}.py`, `application/study.py`.

Hoje: `window_vector` 15×OHLC plano; volume só `vol_z` + VWAP; sem Fib; `swing_weight` clampado para longe de 0 (`max(0.001, …)` em `fusion.py` — **bug relativo ao desenho novo**); Optuna não busca `fib_weight`; `ml.lookback` no YAML não é lido (15/3 hardcoded).

`predict` devolve só chart **passado**.

## O que já quebrou (não repetir)

**HOLD collapsado.** 3 classes + gain 200 pts em 15 min → só HOLD. Two-stage.

**3 M5 com stop-no-mesmo-bar.** No WIN o M5 cobre stop e gain; WR ~23% vs BE ~37%. Ao vivo o hold é até SL/TP em ticks.

Simular caminho **M1** até barreira ou fim do dia. Stop no mesmo M1 continua conservador **antes** de atualizar trailing. Trailing só com `mark=close` depois do bar — nunca high/low do bar em avaliação (isso inflou WR para ~78% com BE falso).

**Meta sem ranqueamento.** HGB nas features velhas: P(gain|acerto) ≈ P(gain|erro). Bolsões na cauda (ex. 80/160, p≥0,337, n=264, +R$600 no teste) são frágeis.

**Optuna no teste.** Busca só na val 2024 H2. Publicar métrica OOS.

`CalibratedClassifierCV(cv="prefit")` quebra no sklearn atual.

## Regras de backtest

1. Caminho M1, mesmo dia, até SL/TP ou fim da sessão.
2. Cooldown até o timestamp de saída + teto 8/dia.
3. PnL = pontos × 0,20 × contratos − 1 × contratos.
4. Não escolher hiperparâmetros no teste.

## Artefatos

`studies/results/model_{daytrade,swing}.joblib` — regenerar no próximo `train`. YAML de banca hoje **não** é OOS validado.

## Próximo trabalho

1. Features em blocos 5×M1 + volume signatures; `Signal.predicted_chart_type`.
2. `fibonacci.py` + fusão com `w_day/w_swing/w_fib` e clamp `w_swing + w_fib ≤ 0.4` (permitir 0).
3. Labels de gain no caminho M1; cabeça `chart_type_future`.
4. Testes: leakage, pesos 0, soma > 0,4 rejeitada, Fib.
5. `python -m koletivo_trader train`; Optuna grava só YAML.
6. UI: tipo previsto 15 min; pytest; `serve` + front local.
