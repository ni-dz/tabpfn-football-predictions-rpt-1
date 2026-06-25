"""Approach 1: predict football fixtures with SAP RPT-1 via in-context learning.

RPT-1 is a relational pretrained transformer — no separate training pass. We
hand it a single classification table: 2000 recent played matches as context
plus the upcoming fixtures with OUTCOME='[PREDICT]'. It learns the
relationship between engineered features and match outcomes in-context.
"""
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss

from common import (
    FEATURES, OUTCOMES,
    load_data, build_features, split_played_future, backtest_window,
    print_freshness, write_predictions, print_predictions,
)

load_dotenv()

# RPT-1 limits (from SAP docs):
#   - max 128 query rows per call
#   - model uses at most 2048 context rows (above is auto-downsampled)
#   - recommended context: 500-2000
RPT1_MODEL = "sap-rpt-1-small"
RPT1_CONTEXT = 2000
RPT1_MAX_QUERY = 128


def _row(row, predict):
    """Build one RPT-1 row: numeric features + outcome (or [PREDICT] placeholder)."""
    payload = {"DATE": row["date"].strftime("%d-%m-%Y"),
               "HOME_TEAM": str(row["home_team"]),
               "AWAY_TEAM": str(row["away_team"])}
    for f in FEATURES:
        payload[f.upper()] = float(row[f])
    payload["OUTCOME"] = "[PREDICT]" if predict else row["outcome"]
    return payload


def _confidence_to_probs(label, confidence):
    """Top-class confidence -> length-3 prob vector, split the rest evenly."""
    if label not in OUTCOMES:
        return np.full(3, 1 / 3)
    conf = float(confidence) if confidence is not None else 1.0
    rest = max(0.0, (1.0 - conf) / 2)
    return np.array([conf if c == label else rest for c in OUTCOMES])


def rpt1_predict(context, queries):
    """Run RPT-1 over all `queries`, chunking by the 128-query-row API limit."""
    from gen_ai_hub.proxy.native.sap import (
        RPTRequest, PredictionConfig, TargetColumn, RPTClient,
    )
    from gen_ai_hub.proxy.native.sap.models import RPTException

    client = RPTClient()
    n = len(queries)
    n_chunks = (n + RPT1_MAX_QUERY - 1) // RPT1_MAX_QUERY
    print(f"  context={len(context)} rows, queries={n} ({n_chunks} chunk(s))")

    all_probs = []
    for i in range(n_chunks):
        chunk = queries.iloc[i * RPT1_MAX_QUERY:(i + 1) * RPT1_MAX_QUERY]
        rows = ([_row(r, predict=False) for _, r in context.iterrows()] +
                [_row(r, predict=True) for _, r in chunk.iterrows()])
        body = RPTRequest(
            prediction_config=PredictionConfig(
                target_columns=[TargetColumn(name="OUTCOME", task_type="classification")],
            ),
            rows=rows,
        )
        try:
            response = client.predict(body=body, model_name=RPT1_MODEL, model_version="latest")
        except RPTException as e:
            raise RuntimeError(f"RPT-1 rejected request: status={e.status} detail={e.detail}") from e

        # response.predictions is list[Prediction]; each Prediction.root maps
        # target column name -> PredictionItem (or list of items).
        for pred in response.predictions:
            root = pred.root if hasattr(pred, "root") else pred
            payload = root.get("OUTCOME") or root.get("outcome")
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                label = getattr(item, "prediction", None)
                conf = getattr(item, "confidence", None)
                all_probs.append(_confidence_to_probs(label, conf))
        print(f"  chunk {i + 1}/{n_chunks} ok")

    proba = np.array(all_probs)
    row_sums = proba.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


def main():
    df = load_data()
    print_freshness(df)

    feats = build_features(df)
    played, future = split_played_future(feats)

    month, train_pool, test = backtest_window(played)
    if len(test):
        print(f"\nBacktest {month} ({len(test)} matches):")
        proba = rpt1_predict(train_pool.tail(RPT1_CONTEXT), test)
        pred = [OUTCOMES[i] for i in proba.argmax(1)]
        print(f"  accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=OUTCOMES):.3f}")

    print(f"\nPredicting {len(future)} upcoming fixtures with RPT-1...")
    proba = rpt1_predict(played.tail(RPT1_CONTEXT), future)
    filename, out = write_predictions(future, proba, OUTCOMES, suffix="rpt1")
    print(f"\nSaved -> {filename}")
    print_predictions(out)


if __name__ == "__main__":
    main()
