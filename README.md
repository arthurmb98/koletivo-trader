# Koletivo Trader

Robô de day trade para o mini índice (WIN), parte do Koletivo Hub. Remake do trader-api: swing de D-1 + daytrade de 15×1 min, ordem a mercado com stop/gain fixos, diário CSV e UI Koletivo.

Não é recomendação de investimento.

## Documentação

- [Plano e arquitetura](docs/plano.md) — produto travado, camadas, orquestrador, journal, UI, status
- [Machine learning](docs/ml.md) — rótulos, o que já falhou no treino, regras de backtest, próximos passos

## Pastas

- `ui/` — React (estudo, replay, ao vivo)
- `api/` — Python (domínio, ML, orquestrador, MT5)
- `datasets/` — WIN$ M1/M5
- `configs/` — YAML (stop, gain, banca, pesos)
- `studies/results/` — modelos `.joblib` e `studies.json`
- `journal/` — diário operacional local (fora do Git)

## API

```bash
cd api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt
pip install -e .
python -m koletivo_trader train
python -m koletivo_trader serve
```

## UI

```bash
cd ui
npm install
npm run dev
```

Abra http://127.0.0.1:5173 — estudo `/`, replay `/replay`, ao vivo `/ao-vivo`.

Armado envia ordem no fechamento do M5 (demo `mt5` ou `prd`), com SL/TP na mesma ordem. Em posição o motor lê ticks a cada 20 ms e só altera o stop quando ele muda (≥1 tick), no mínimo a cada 150 ms, para não saturar o terminal. Perto do gain o stop sobe para travar lucro. Limites: `max_trades_per_day` e `daily_loss_points`.

## Testes

```bash
cd api
pytest
```
