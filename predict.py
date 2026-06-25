"""Baseline: predict international football fixtures with TabPFN."""
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier

from common import (
    FEATURES, MAX_TRAIN,
    load_data, build_features, split_played_future, backtest_window,
    print_freshness, write_predictions, print_predictions,
)


def train(pool):
    """Fit TabPFN on the feature matrix; ignore_pretraining_limits allows >1000 rows."""
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def main():
    df = load_data()
    print_freshness(df)

    feats = build_features(df)
    played, future = split_played_future(feats)

    month, train_pool, test = backtest_window(played)
    if len(test):
        clf = train(train_pool)
        proba = clf.predict_proba(test[FEATURES].values)
        pred = clf.classes_[proba.argmax(1)]
        print(f"\nBacktest {month} ({len(test)} matches): "
              f"accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=clf.classes_):.3f}")

    clf = train(played.tail(MAX_TRAIN))
    proba = clf.predict_proba(future[FEATURES].values)
    filename, out = write_predictions(future, proba, list(clf.classes_), suffix="tabpfn")
    print(f"\nSaved -> {filename}")
    print_predictions(out)


if __name__ == "__main__":
    main()
