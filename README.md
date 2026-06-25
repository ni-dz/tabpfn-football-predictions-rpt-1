# TabPFN / RPT-1 / LLM Football Predictions

Three competition entries for Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/football#competition) sharing the same engineered-feature pipeline so their outputs are directly comparable.

| Script | Approach | Model |
|---|---|---|
| `predict.py` | Baseline — TabPFN on engineered features | TabPFN (Prior Labs) |
| `predict_rpt1.py` | **Approach 1** — SAP RPT-1 in-context learning | `sap-rpt-1-small` |
| `predict_llm.py` | **Approach 2** — LLM only, given a structured match briefing | `gpt-5` via Orchestration V2 |
| `predict_hybrid.py` | **Approach 3** — RPT-1 base probabilities refined by the LLM | `sap-rpt-1-small` + `gpt-5` |

## Setup

```bash
pip install -r requirements.txt
```

`.env` (gitignored) needs your SAP AI Core credentials for Approaches 1–3:

```
AICORE_AUTH_URL=...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=...
AICORE_RESOURCE_GROUP=...
```

## Run

```bash
python predict.py            # Baseline (TabPFN)
python predict_rpt1.py       # Approach 1 (RPT-1)
python predict_llm.py        # Approach 2 (LLM only)
python predict_hybrid.py     # Approach 3 (Hybrid)
```

Every run:

1. Downloads the latest international results dataset from [martj42/international_results](https://github.com/martj42/international_results) (new matches land daily — always fresh).
2. Builds features with a single chronological pass (no leakage).
3. Runs a backtest on the previous calendar month — prints accuracy + log-loss.
4. Predicts every upcoming fixture.
5. Saves `predictions_<approach>_YYYYMMDD.csv` and prints the table.

Output schema is the competition standard: `date, home_team, away_team, predicted, p_home_win, p_draw, p_away_win` (plus `reasoning` for LLM-based approaches).

## Output

```
Latest game in dataset: 2026-06-27
Data freshness: 0 days 09:44:16

Backtest 2026-05 (26 matches):
  accuracy 85%, log-loss 0.531

Predicting 18 upcoming fixtures with RPT-1...
  context=2000 rows, queries=18 (1 chunk(s))
  chunk 1/1 ok

Saved -> predictions_rpt1_20260624.csv

18 fixture predictions

  2026-06-25            Tunisia vs Netherlands       -> away_win   H 15% | D 15% | A 70%
  2026-06-25              Japan vs Sweden            -> home_win   H 68% | D 16% | A 16%
  ...
```

## Approach details

**Approach 1 (RPT-1)** — RPT-1 is a relational pretrained transformer; no separate training pass. We hand it 2000 recent played matches as context plus the upcoming fixtures with `OUTCOME='[PREDICT]'`. RPT-1 learns the relationship between engineered features and match outcomes in-context. Settings: `sap-rpt-1-small`, 2000-row context (model hard-cap is 2048), automatic chunking by the 128-query-row API limit.

**Approach 2 (LLM only)** — One LLM call per fixture (parallelized). The prompt is a JSON briefing with ELO, recent form, head-to-head record, prose summaries of recent matches, venue, and tournament importance. Strict JSON schema response format guarantees parseable output. Settings: `gpt-5` via Orchestration V2, 4096 max-completion-tokens (gpt-5 burns reasoning tokens before output).

**Approach 3 (hybrid)** — RPT-1 runs first to produce base probabilities for every upcoming fixture. The LLM then receives the RPT-1 distribution **together** with the same prose context Approach 2 uses, and is instructed to default to trusting RPT-1 unless qualitative context materially contradicts it. The `reasoning` column makes every adjustment auditable.

## Features

| Feature | Description |
|---|---|
| `elo_diff` | ELO gap (home + home advantage − away) |
| `home_elo`, `away_elo` | Current ELO ratings |
| `form5_diff`, `form10_diff` | Difference in average points per game over last 5 / 10 matches |
| `home_form5`, `away_form5` | Points per game over last 5 matches |
| `home_winrate`, `away_winrate` | Win rate over last 10 matches |
| `home_gf5`, `away_gf5` | Goals scored per game over last 5 matches |
| `home_ga5`, `away_ga5` | Goals conceded per game over last 5 matches |
| `gd10_diff` | Difference in average goal difference over last 10 matches |
| `home_streak`, `away_streak` | Current win streak |
| `home_rest`, `away_rest` | Days since last match (capped at 90) |
| `home_played`, `away_played` | Total matches played in history |
| `h2h_n` | Number of head-to-head meetings |
| `h2h_home_winrate` | Home team win rate in head-to-head |
| `h2h_draw_rate` | Draw rate in head-to-head |
| `h2h_gd` | Average goal difference in head-to-head (from home team's perspective) |
| `neutral` | 1 if played at a neutral venue |
| `importance` | Tournament importance score (60 = World Cup, 20 = friendly) |
