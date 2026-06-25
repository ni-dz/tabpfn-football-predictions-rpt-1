"""Approach 3: hybrid — RPT-1 base probabilities, refined by the LLM.

  1. RPT-1 produces base probabilities for every upcoming fixture.
  2. For each fixture, the LLM receives RPT-1's distribution plus structured
     context (ELO, form, head-to-head, recent-match prose) and decides to keep
     or adjust RPT-1's call, with reasoning.

The LLM acts as a calibration / sanity-check layer on top of RPT-1.
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
from predict_rpt1 import rpt1_predict, RPT1_CONTEXT

load_dotenv()

LLM_MODEL = "gpt-5"
MAX_TOKENS = 4096
WORKERS = 8

SYSTEM_PROMPT = """You are an expert football analyst working alongside SAP RPT-1, a
relational foundation model that produces base probability estimates for match outcomes.

For each fixture you receive RPT-1's base probabilities together with structured context
(ELO ratings, recent form, head-to-head record, prose summaries of recent matches). Your
job is to produce the FINAL calibrated probabilities for home_win / draw / away_win.

Default to trusting RPT-1's distribution. Only deviate when the qualitative context
materially contradicts it — for example: a heavy hitter on a long losing streak, a
neutral-venue match where RPT-1 may overweight home advantage, a recent blowout in the
head-to-head that the numeric features dilute. Stay close to RPT-1 unless you can
articulate why.

The three probabilities MUST sum to 1.0 (within 0.01). Note in `reasoning` whether you
kept or adjusted RPT-1's distribution and why."""

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


def _brief(row, df, rpt_probs):
    home, away, date = row["home_team"], row["away_team"], row["date"]
    return {
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home, "away_team": away,
        "neutral_venue": bool(row["neutral"]),
        "tournament_importance": float(row["importance"]),
        "rpt1_base_probabilities": {
            "p_home_win": round(float(rpt_probs[0]), 4),
            "p_draw":     round(float(rpt_probs[1]), 4),
            "p_away_win": round(float(rpt_probs[2]), 4),
        },
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
    from gen_ai_hub.orchestration_v2 import (
        Template, SystemMessage, UserMessage, LLMModelDetails,
        PromptTemplatingModuleConfig, ModuleConfig, OrchestrationConfig,
        OrchestrationService, ResponseFormatJsonSchema, JSONResponseSchema,
    )
    template = Template(
        template=[
            SystemMessage(content=SYSTEM_PROMPT),
            UserMessage(content="Match briefing with RPT-1 base probs (json):\n{{?brief}}\n\n"
                                "Return final prediction as json."),
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


def hybrid_predict(df, fixtures, rpt_proba):
    """For each fixture, hand RPT-1's probs and context to the LLM."""
    service = _build_service()
    n = len(fixtures)
    final = np.zeros((n, 3))
    reasoning = [""] * n

    def _job(idx, row, rpt_probs):
        try:
            probs, why = _call(service, _brief(row, df, rpt_probs))
            return idx, probs, why, None
        except Exception as e:
            # Fall back to RPT-1 rather than uniform
            return idx, rpt_probs, f"ERROR (fell back to RPT-1): {e}", e

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_job, i, row, rpt_proba[i])
                   for i, (_, row) in enumerate(fixtures.iterrows())]
        done = 0
        for fut in as_completed(futures):
            idx, probs, why, err = fut.result()
            final[idx] = probs
            reasoning[idx] = why
            done += 1
            r = fixtures.iloc[idx]
            tag = "ERR" if err else "ok "
            rpt = rpt_proba[idx]
            line = (f"  [{done:>3}/{n}] {tag} {r['home_team']} vs {r['away_team']}: "
                    f"RPT(H{rpt[0]:.0%}/D{rpt[1]:.0%}/A{rpt[2]:.0%}) -> "
                    f"LLM(H{probs[0]:.0%}/D{probs[1]:.0%}/A{probs[2]:.0%})")
            if err:
                line += f"\n        -> {type(err).__name__}: {err}"
            print(line)
    return final, reasoning


def main():
    df = load_data()
    print_freshness(df)

    feats = build_features(df)
    played, future = split_played_future(feats)

    month, train_pool, test = backtest_window(played)
    if len(test):
        print(f"\nBacktest {month} ({len(test)} matches):")
        print("  step 1: RPT-1 base predictions")
        rpt_proba = rpt1_predict(train_pool.tail(RPT1_CONTEXT), test)
        print("  step 2: LLM refinement")
        proba, _ = hybrid_predict(df, test, rpt_proba)
        pred = [OUTCOMES[i] for i in proba.argmax(1)]
        print(f"  accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=OUTCOMES):.3f}")

    print(f"\nStep 1: RPT-1 base predictions for {len(future)} fixtures...")
    rpt_proba = rpt1_predict(played.tail(RPT1_CONTEXT), future)
    print(f"Step 2: LLM refinement ({LLM_MODEL})...")
    proba, reasoning = hybrid_predict(df, future, rpt_proba)
    filename, out = write_predictions(future, proba, OUTCOMES, suffix="hybrid", reasoning=reasoning)
    print(f"\nSaved -> {filename}")
    print_predictions(out)


if __name__ == "__main__":
    main()
