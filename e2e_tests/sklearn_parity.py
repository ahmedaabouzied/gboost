"""Train sklearn GradientBoostingClassifier on iris data and output predictions as JSON."""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="directory containing iris_train.csv and iris_test.csv")
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "iris_train.csv")
    test_path = os.path.join(args.data_dir, "iris_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values
    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    test_probabilities = clf.predict_proba(X_test)[:, 1].tolist()
    test_predictions = clf.predict(X_test).tolist()
    test_accuracy = float(np.mean(clf.predict(X_test) == y_test))
    train_accuracy = float(np.mean(clf.predict(X_train) == y_train))

    result = {
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
