# Machine learning e busca de parâmetros

O produto está em [plano.md](plano.md). Esta nota cobre o tensor de treino, a fusão, o AG e o que já falhou.

## Papéis

| Peça | Função | Retreina a cada geração do AG? |
| --- | --- | --- |
| Daytrade (HGB two-stage) | Padrão dos 15 M1 + P(gain) | Não — labels usam ATR, não o stop/gain do YAML |
| Swing (HGB) | Tipo de dia D-1 e viés | Não a cada geração; treina uma vez com semente 100/200 |
| AG (`ml/genetics.py`) | stop, gain, `min_hit`, pesos, offset | Não é ML de gráfico; avalia backtest no sinal já previsto |
| MLP do AG | Surrogate para ranquear genomas | Só acelera o AG; **não** entra no robô ao vivo |

Se no futuro um gene passar a rotular o Y do HGB, aí sim aquela cabeça seria retreinada. Hoje nenhum gene do AG entra no `DaytradeModel.fit`.

## Objetivo do daytrade

P(gain) no horário de ouro, custo `point_value=0.20` + `contract_cost=1`, máx. 8 trades/dia. Teste 2025–2026 só para métrica OOS.

## Três entradas

15 M1 = 3 blocos de 5 minutos. Cada bloco: 20 OHLC ATR-normalizados + 5 volumes.

1. **X preço** — treino e inferência.
2. **X volume** — confirmação, exaustão, RVOL ≥ 1,5, falso rompimento, absorção, A/D, z-score de liquidez.
3. **Futuro só no treino** — 15 M1 seguintes para Y (caminho de gain e `chart_type_future`). Inferência nunca vê isso.

Pré-rótulo: `classify_chart` no passado (feature) e no futuro (alvo). Swing: `classify_day` no D-1 (feature) e no D (alvo).

## Saídas do daytrade

- Two-stage BUY vs SELL + meta P(gain). Não voltar a 3 classes com HOLD majoritário.
- `chart_type` (passado) e `predicted_chart_type` (cabeça no Y futuro).
- `hit_pct` = P(gain) do meta, **depois** da fusão.

## Fusão

```
hit_final = hit_day + w_swing * signed_swing + w_fib * signed_fib
```

`w_swing + w_fib ≤ 0,4`. Peso 0 zera aquele termo. Discordância reduz o hit na proporção do peso; não veta o lado. HOLD só se `hit_final < min_hit_pct`.

Exemplo: 96% daytrade, swing 80% no lado oposto, peso 0,2 → 80%.

Fibonacci assinado: proximidade aos 38,2/50/61,8, sinal + se alinhado ao daytrade, − se contra.

## Offset

`entry = round_tick(close + offset_points)`. O AG varre offset em passos de 5 pts, inclusive 0 e negativos. Stop/gain a partir dessa entrada.

## Algoritmo genético

Não é Optuna. Não é uma terceira ML de sinal.

- Representação real: `[stop, gain, min_hit, swing_w, fib_w, offset]`
- Reparo: tick 5, `gain ≥ 1,5 × stop`, `swing_w + fib_w ≤ 0,4`
- SBX (η=15), mutação polinomial (η=20), torneio k=3, elitismo, imigração ~15%
- População ~24, ≤16 gerações, early-stop se a elite não sobe
- Fitness: mediana de 3 folds purged na val 2024 H2, penaliza DD > 28% da banca e WR abaixo do breakeven com custo
- Surrogate: `MLPRegressor(32, 16)` treinado nos genomas já avaliados; só sugere infill

`python -m koletivo_trader train` treina swing+daytrade e em seguida o AG; escreve `configs/best_bank_*.yaml`.

## Backtest honesto

1. Caminho **M1**, mesmo dia, até SL/TP ou fim da sessão.
2. Stop no mesmo M1 continua conservador (stop ganha se os dois tocam).
3. Trailing só com `mark=close` **depois** do bar — nunca high/low do bar (isso inflou WR para ~78% com BE falso).
4. Cooldown até o timestamp de saída + teto 8/dia.
5. PnL = pontos × 0,20 × contratos − 1 × contratos.
6. Hiperparâmetros **nunca** no teste 2025+.

## O que já quebrou (não repetir)

**HOLD collapsado.** 3 classes + gain 200 pts em 15 min → só HOLD. Two-stage.

**3 M5 com stop-no-mesmo-bar.** No WIN o M5 cobre stop e gain; WR ~23% vs BE ~37%. Ao vivo o hold é até SL/TP em ticks.

**Média ponderada que diluía o daytrade.** `w_day * hit + w * boost` + veto nos primeiros 15 min. Substituída pela fórmula aditiva acima.

**Meta sem ranqueamento.** HGB nas features velhas: P(gain\|acerto) ≈ P(gain\|erro). Bolsões na cauda são frágeis.

**Optuna no teste.** A busca (agora AG) só na val 2024 H2.

`CalibratedClassifierCV(cv="prefit")` quebra no sklearn atual — o fit ignora se falhar.

## Artefatos

- `studies/results/model_daytrade.joblib`, `model_swing.joblib`
- `configs/best_bank_{500,1000,5000,10000}.yaml` — stop, gain, min_hit, pesos, offset
- `ui/public/studies.json` — página de estudo

Não há `model_params` como “terceiro cérebro”. O joblib de params, se existir, é só metadado da última busca.
