"""Approach 2: predict football fixtures with an LLM only (no tabular model).

The LLM gets a compact briefing (ELO, recent form, head-to-head, tournament
importance) for each fixture and returns calibrated 3-class probabilities plus
a short reasoning trace. Uses SAP AI Core's Orchestration Service V2 — it
manages the virtual LLM deployment automatically, no manual deploy needed.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss

from common import (
    OUTCOMES,
    load_data, build_features, split_played_future, backtest_window,
    recent_form_summary, h2h_summary,
    print_freshness, write_predictions, print_predictions,
)

load_dotenv()

LLM_MODEL = "gpt-5"
MAX_TOKENS = 4096   # gpt-5 burns many tokens on internal reasoning; keep this high
WORKERS = 8         # parallel LLM calls

SYSTEM_PROMPT = """You are an expert football analyst. For each international match
briefing you receive, return calibrated probabilities for the three possible outcomes
from the home team's perspective: home_win, draw, away_win.

The three probabilities MUST sum to 1.0 (within 0.01). Base your estimate on the ELO
gap, recent form, head-to-head record, venue (neutral or home), and tournament
importance. Be well-calibrated: draws are common in international football (~25%
base rate)."""

PREDICTION_SCHEMA = {
    "title": "MatchPrediction",
    "type": "object",
    "properties": {
        "p_home_win": {"type": "number", "minimum": 0, "maximum": 1},
        "p_draw":     {"type": "number", "minimum": 0, "maximum": 1},
        "p_away_win": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning":  {"type": "string"},
    },
    "required": ["p_home_win", "p_draw", "p_away_win", "reasoning"],
}


def _brief(row, df):
    """Compact JSON briefing for one fixture: numeric features + recent prose."""
    home, away, date = row["home_team"], row["away_team"], row["date"]
    return {
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home, "away_team": away,
        "neutral_venue": bool(row["neutral"]),
        "tournament_importance": float(row["importance"]),
        "elo": {"home": round(float(row["home_elo"]), 1),
                "away": round(float(row["away_elo"]), 1),
                "diff_with_home_advantage": round(float(row["elo_diff"]), 1)},
        "form_ppg_last5": {"home": round(float(row["home_form5"]), 2),
                           "away": round(float(row["away_form5"]), 2)},
        "win_rate_last10": {"home": round(float(row["home_winrate"]), 2),
                            "away": round(float(row["away_winrate"]), 2)},
        "goals_per_game_last5": {
            "home": {"scored": round(float(row["home_gf5"]), 2),
                     "conceded": round(float(row["home_ga5"]), 2)},
            "away": {"scored": round(float(row["away_gf5"]), 2),
                     "conceded": round(float(row["away_ga5"]), 2)},
        },
        "win_streak": {"home": int(row["home_streak"]), "away": int(row["away_streak"])},
        "rest_days": {"home": int(row["home_rest"]), "away": int(row["away_rest"])},
        "head_to_head": {
            "matches": int(row["h2h_n"]),
            "home_winrate": round(float(row["h2h_home_winrate"]), 2),
            "draw_rate": round(float(row["h2h_draw_rate"]), 2),
            "avg_goal_diff_for_home": round(float(row["h2h_gd"]), 2),
            "recent": h2h_summary(df, home, away, date),
        },
        "recent_form": {
            "home": recent_form_summary(df, home, date),
            "away": recent_form_summary(df, away, date),
        },
    }


def _build_service():
    """One Orchestration V2 pipeline, reused for every fixture."""
    from gen_ai_hub.orchestration_v2 import (
        Template, SystemMessage, UserMessage, LLMModelDetails,
        PromptTemplatingModuleConfig, ModuleConfig, OrchestrationConfig,
        OrchestrationService, ResponseFormatJsonSchema, JSONResponseSchema,
    )
    template = Template(
        template=[
            SystemMessage(content=SYSTEM_PROMPT),
            UserMessage(content="Match briefing (json):\n{{?brief}}\n\nReturn prediction as json."),
        ],
        response_format=ResponseFormatJsonSchema(
            json_schema=JSONResponseSchema(
                name="match_prediction",
                description="Calibrated 3-way match outcome probabilities",
                schema=PREDICTION_SCHEMA,
            ),
        ),
    )
    llm = LLMModelDetails(name=LLM_MODEL, params={"max_completion_tokens": MAX_TOKENS})
    config = OrchestrationConfig(modules=ModuleConfig(
        prompt_templating=PromptTemplatingModuleConfig(prompt=template, model=llm)))
    return OrchestrationService(config=config)


def _call(service, brief):
    """One LLM call -> (probs_array_of_3, reasoning_string)."""
    result = service.run(placeholder_values={"brief": json.dumps(brief, ensure_ascii=False)})
    text = result.final_result.choices[0].message.content
    if not text or not text.strip():
        finish = getattr(result.final_result.choices[0], "finish_reason", "unknown")
        raise ValueError(f"LLM returned empty content (finish_reason={finish!r}).")
    obj = json.loads(text)
    probs = np.array([float(obj.get(f"p_{c}", 0.0)) for c in OUTCOMES])
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"All-zero probabilities: {obj}")
    return probs / total, obj.get("reasoning", "").strip()


def llm_predict(df, fixtures):
    """Predict every fixture in parallel. `df` is the full results frame
    (needed for the recent-form / h2h prose snippets)."""
    service = _build_service()
    n = len(fixtures)
    proba = np.zeros((n, 3))
    reasoning = [""] * n
    errors = []

    def _job(idx, row):
        try:
            probs, why = _call(service, _brief(row, df))
            return idx, probs, why, None
        except Exception as e:
            return idx, np.full(3, 1 / 3), f"ERROR: {e}", e

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_job, i, row) for i, (_, row) in enumerate(fixtures.iterrows())]
        done = 0
        for fut in as_completed(futures):
            idx, probs, why, err = fut.result()
            proba[idx] = probs
            reasoning[idx] = why
            done += 1
            r = fixtures.iloc[idx]
            tag = "ERR" if err else "ok "
            line = (f"  [{done:>3}/{n}] {tag} {r['home_team']} vs {r['away_team']}: "
                    f"H {probs[0]:.0%} | D {probs[1]:.0%} | A {probs[2]:.0%}")
            if err:
                line += f"\n        -> {type(err).__name__}: {err}"
                errors.append(err)
            print(line)
    if errors and len(errors) == n:
        raise RuntimeError(f"All {n} LLM calls failed. First error: {errors[0]!r}")
    return proba, reasoning


def main():
    df = load_data()
    print_freshness(df)

    feats = build_features(df)
    played, future = split_played_future(feats)

    month, _, test = backtest_window(played)
    if len(test):
        print(f"\nBacktest {month} ({len(test)} matches):")
        proba, _ = llm_predict(df, test)
        pred = [OUTCOMES[i] for i in proba.argmax(1)]
        print(f"  accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=OUTCOMES):.3f}")

    print(f"\nPredicting {len(future)} upcoming fixtures with LLM ({LLM_MODEL})...")
    proba, reasoning = llm_predict(df, future)
    filename, out = write_predictions(future, proba, OUTCOMES, suffix="llm", reasoning=reasoning)
    print(f"\nSaved -> {filename}")
    print_predictions(out)


if __name__ == "__main__":
    main()
