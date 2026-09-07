# Koletivo Trader

Robô de day trade para o mini índice B3 (WIN). Faz parte do [Koletivo Hub](https://koletivo-hub.vercel.app). Não é recomendação de investimento.

A ML lê o gráfico e estima a chance de atingir o gain antes do stop. Um **algoritmo genético** escolhe só os parâmetros de operação (stop, gain, piso de confiança, pesos, offset de entrada). A ordem sai a mercado já com SL/TP.

## Como funciona

1. **Swing** (contexto) — lê o pregão anterior em M5. Classifica o tipo de dia e um viés BUY/SELL/HOLD. **Não opera sozinho.**
2. **Daytrade** (decisor) — a cada fechamento de M5 lê 15 candles de 1 min em 3 blocos de 5 min (preço + volume). Devolve lado, tipo do gráfico passado, tipo previsto para os próximos 15 min, e `%` de acerto.
3. **Fusão** — o daytrade entra com o `%` cheio. Swing e Fibonacci só somam ou subtraem `peso × confiança`. Peso 0 = aquele decisor não existe. Daytrade nunca fica com menos de 60% de relevância (`swing_weight + fib_weight ≤ 0,4`).
4. **Offset** — a entrada não precisa ser o fechamento do último candle. `offset_points = 0` entra no close; valores positivos ou negativos (tick de 5 pts) são escolhidos pelo AG.
5. **AG** — população real-codificada (crossover SBX, mutação polinomial, elitismo, MLP só para acelerar o ranking). Fitness em walk-forward na validação 2024 H2. Teste 2025+ só reporta, não escolhe parâmetro. Grava YAML em `configs/best_bank_{500,1000,5000,10000}.yaml`.

Exemplo de fusão: daytrade 96% de compra, swing discorda a 80% com peso 0,2 → `0,96 + 0,2 × (−0,80) = 0,80`. O sinal continua se ainda estiver acima de `min_hit_pct`. Não há veto binário.

A ML de gráfico **não** é retreinada a cada geração do AG: stop, gain, pesos, piso e offset entram só na decisão. O daytrade treina com barreiras por ATR; o swing usa o stop/gain semente (100/200) uma vez.

## Documentação

- [Plano e arquitetura](docs/plano.md)
- [Machine learning e AG](docs/ml.md)

## Pastas

| Pasta | Conteúdo |
| --- | --- |
| `ui/` | React 19 + Vite (estudo `/`, replay `/replay`, ao vivo `/ao-vivo`) |
| `api/` | Python hexagonal (domínio, ML, AG, orquestrador, MT5, HTTP) |
| `datasets/` | WIN$ M1/M5 (treino ≤ 2024, teste ≥ 2025) |
| `configs/` | YAML de sessão, risco, pesos e offset |
| `studies/results/` | `model_daytrade.joblib`, `model_swing.joblib`, `studies.json` |
| `journal/` | Diário CSV local (`YYYY-MM-DD/`) — fora do Git |
| `docs/` | Plano e nota de ML |

## API (Windows)

Python 3.11+. O pacote `MetaTrader5` só no Windows.

```bash
cd api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt
pip install -e .
set PYTHONPATH=src
python -m koletivo_trader train
python -m koletivo_trader serve
```

CLI: `train` (ML + AG), `study` (alias de train), `replay`, `serve`, `mt5-check`.

API local em http://127.0.0.1:8000. O front faz poll a **4 Hz** (250 ms).

## UI

Node 20+.

```bash
cd ui
npm install
npm run dev
```

Abra http://127.0.0.1:5173

- `/` — estudo (`ui/public/studies.json`)
- `/replay` — janelas CSV, sem ordem e sem journal
- `/ao-vivo` — default hoje; seletor desde o primeiro dia real no journal

O card de sinal **sempre** mostra lado (Compra / Venda / **Não fazer nada**), tipo do gráfico, tipo previsto dos próximos 15 min, `%` e frase gerada no domínio.

## Operação ao vivo

Armado envia ordem no fechamento do M5 (demo `mt5` ou `prd` explícito), com SL/TP na mesma ordem, no preço `close + offset_points` (arredondado ao tick 5). Em posição: ticks a 20 ms; idle 100 ms; modifica stop só se mudar ≥ 1 tick e ≥ 150 ms. Perto do gain o stop sobe para travar lucro. Limites: `max_trades_per_day` (8) e `daily_loss_points`.

Horário de ouro: 09:15–11:00 e 14:30–17:00 (`America/Sao_Paulo`).

## Testes

```bash
cd api
set PYTHONPATH=src
pytest
```

Sem `order_send` nos testes. Rótulos de treino não vazam o futuro para o X.

## Repo

Público: [arthurmb98/koletivo-trader](https://github.com/arthurmb98/koletivo-trader), branch `staging`. `.env` e `journal/` gitignored. A senha do MT5 (`MT5_PASSWORD`) fica só no `.env` local e nunca vai ao Git.
