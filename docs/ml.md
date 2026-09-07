# Machine learning — desenho, falhas e próximos passos

O produto (15×M1 → sinal para o próximo bloco de 3 M5, swing de D-1, Optuna de stop/gain) permanece o do [plano](plano.md). Esta nota registra **o que mudou no treino** e **por que o backtest de 3 M5 não descreve o robô ao vivo**.

## Objetivo

Classificador daytrade que, fora da amostra (teste 2025–2026), opere no horário de ouro com expectativa positiva após custo (`point_value=0.20`, `contract_cost=1`), respeitando `max_trades_per_day=8`.

Split fixo: treino ≤ 2024-12-30, teste ≥ 2025-01-02. Optuna **não** deve caber no teste: validação = 2024 H2 (`2024-07-01` em diante no CSV de treino).

## Pipeline atual (código)

Arquivos: `api/src/koletivo_trader/ml/{labels,features,models,parameters}.py` e `application/study.py`.

1. **Janelas sem leakage** (`leak_free_windows`): 15 M1 fechados até o M5 `T`; Y começa no **próximo** M5 (entrada no open). O M1 da janela nunca inclui o futuro.
2. **Sessão no treino**: horário de ouro (09:15–11:00 e 14:30–17:00), igual ao live. Receitas `morning_only` usam só a manhã.
3. **Amostras não sobrepostas no fit**: stride 3 (horizonte de 3 M5) para reduzir rótulos duplicados.
4. **Peso de recência**: meia-vida ~400 dias.
5. **Two-stage**
   - Direção: BUY vs SELL (não um softmax 3 classes com HOLD majoritário).
   - Meta-label: P(gain) dado o lado, com flag de lado concatenada nas features.
6. **Receitas** (`DaytradeRecipe`): `first_touch_atr`, `first_touch_pts`, `unique_gain`, `momentum_meta`, `agree`.
7. **Inferência**: lado pela direção (`p_buy` vs `p_sell`); `hit_pct` = P(gain) do meta; HOLD se `p_dir < 0.52` ou `q < min_score`.
8. **Features**: 15×OHLC normalizado por ATR, estrutura (retornos, CLV, pavios, RSI-like, VWAP da janela, rompimentos, flags de ouro), one-hot de `chart_type`, 6 M5 **já fechados** do mesmo dia (`prior_m5_bars`).
9. **Optuna**: stop/gain com RR ≥ 1,5; `min_hit` 0,30–0,62; penaliza win rate abaixo do breakeven com custo; teto 8 trades/dia; sem sobrepor enquanto o trade anterior não resolve.
10. **Busca de receitas**: treina no fit set (< 2024-07-01), escolhe parâmetros na val, reporta teste; retreina a vencedora no treino completo.

Barreiras adaptativas (ATR da janela M1, tick 5, piso/teto): usadas nos **rótulos**, não necessariamente iguais ao stop/gain fixos da ordem ao vivo.

## O que já quebrou (não repetir)

### Colapso em HOLD

Classificador 3 classes + alvo “gain único de 200 pts em 15 min” → HOLD ~62% → o modelo só prediz HOLD. Balancear as 3 classes caiu em ~50% (aleatório).

**Correção:** two-stage (direção limpa + meta). Não voltar a argmax 3 classes com HOLD majoritário.

### Backtest de 3 M5 com “stop ganha no mesmo candle”

No WIN, um M5 frequentemente tem range que cobre stop **e** gain. A regra conservadora (stop vence se os dois tocam o mesmo bar) destrói o RR aparente: win rate ~20% vs breakeven ~35–37% para 50/100, **em qualquer direção** (momentum, fade, breakout, heurística de chart).

Isso **não** é o robô ao vivo:

- A posição fica até SL/TP (com trailing em **ticks**), não 15 minutos.
- O caminho intra-M5 não é “stop primeiro por definição”.

**Correção obrigatória no próximo treino:** simular o caminho em **M1** a partir do open de entrada, até SL/TP ou fim da sessão (cap ~90–180 min). O horizonte de 3 M5 continua sendo só a cadência de **quando** decidir.

Medição honesta (teste 2025–2026, ouro, momentum, 1 mini, custo R$1, máx. 8/dia):

| Simulação | Stop/gain | Win rate | PnL (aprox.) |
| --- | --- | --- | --- |
| 3×M5, stop vence no mesmo bar | 50/100 | ~23% | ~−R$13k |
| Caminho M1 ~15 min | 50/100 | ~32% | ~−R$5k |
| Caminho M1 ~180 min | 50/100 | ~33–36% | perto de zero / leve prejuízo |
| Trailing otimista (usa máxima/mínima do **mesmo** M1 para mover o stop e depois testa o stop) | 50/100 | ~78% (quase tudo BE) | **inflado — não usar** |
| Trailing conservador (`protect_levels` **depois** do bar, `mark=close`) | 50/100 | WR sobe, EV ainda pode ser negativa | wins médios encolhem |

Breakeven com custo, RR 2 (50/100): win precisa de ~36,7% se o win médio for o gain cheio. Trailing que transforma alvo em BE+10 pts **sobe WR e pode piorar EV**.

### Meta sem poder de ranqueamento

Com primário = sinal da inclinação da janela de 15 M1 e Y = gain vs stop no caminho M1 (~120 min), HGB nas features atuais:

- P(gain) média no teste ≈ taxa base (~31–33%).
- P(gain | y=1) ≈ P(gain | y=0) (sem separação).
- Limiares na cauda às vezes mostram um bolsão lucrativo (ex.: 80/160, p≥0,337, n=264, WR 40%, PnL +600 no teste) — tratar como **hipótese frágil**, não como estratégia fechada, até repetir com walk-forward e menos overlap.

`CalibratedClassifierCV(..., cv="prefit")` falha no sklearn recente (`cv` não aceita `"prefit"`). Calibrar com split temporal explícito ou `FrozenEstimator`, se existir na versão instalada.

### Optuna no teste

A primeira busca de parâmetros rodou no CSV de teste → 0 trades / min_hit alto. A busca agora é na **val 2024 H2**. O YAML publicado deve carregar métricas **OOS** (teste), não o score in-sample.

## Regras de backtest (para o próximo ciclo)

1. Caminho de execução = M1 (ou ticks), mesmo dia, até barreira ou fim da sessão.
2. No mesmo M1: se low e high tocam stop e gain, stop continua conservador **antes** de atualizar trailing.
3. Trailing = `protect_levels` com `mark` = close do M1 (ou último tick), nunca high/low do bar que ainda está sendo avaliado.
4. Trade aberto bloqueia nova entrada (cooldown até o timestamp de saída), além do teto de 8/dia.
5. PnL = pontos × 0,20 × contratos − 1 × contratos.
6. Não escolher hiperparâmetros no teste. Iterar engenharia na val; o teste só confirma.

## Estado dos artefatos

- `studies/results/model_daytrade.joblib`, `model_swing.joblib`, `model_params.joblib` podem ser de treinos anteriores (HOLD ou Optuna no teste). Retreinar depois de ligar o simulador M1.
- `configs/best_bank_*.yaml` atuais não representam uma estratégia OOS validada.

## Próximo trabalho (ML)

1. Trocar `simulate_touch` de treino/Optuna para caminho M1 de hold (manter 3 M5 só em `leak_free_windows` para o X).
2. Meta-label = resultado **desse** caminho (gain / stop / MTM no último close se timeout).
3. Primário preferencial: momentum/auction da janela **passada** (Lopez de Prado), não first-touch do **mesmo** futuro usado no Y do meta.
4. Varredura de receitas de novo com o simulador correto; só então Optuna 40 trials nas 3 bancas.
5. Se o classificador continuar sem ranqueamento, testar filtro tabular simples (hora × chart_type) e ORB da abertura — ainda com o mesmo simulador M1 — em vez de aprofundar HGB no mesmo X.
