"""Train sklearn GBM on iris data, run shap.TreeExplainer, output SHAP stats as JSON.

This runs alongside gboost's own SHAP implementation. Because the two models
are trained independently with different RNGs they will not be point-identical,
so the Go-side parity test compares rankings and directional agreement rather
than sample-by-sample SHAP values.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_dir, "iris_train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "iris_test.csv"))

    feature_names = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_names].values
    y_train = train_df["label"].values
    X_test = test_df[feature_names].values

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)
    # shap_values for binary GBM returns (n_samples, n_features) in log-odds space.
    shap_values = explainer.shap_values(X_test)
    base_value = float(np.ravel(explainer.expected_value)[0])

    # mean(|phi|) per feature — SHAP feature importance in log-odds units.
    shap_importance = np.abs(shap_values).mean(axis=0).tolist()

    # Additivity check: sum(phi) + base ≈ raw log-odds for each sample.
    raw_logodds = clf.decision_function(X_test).tolist()
    additivity = (shap_values.sum(axis=1) + base_value - np.array(raw_logodds)).tolist()

    result = {
        "feature_names": feature_names,
        "base_value": base_value,
        "shap_values": shap_values.tolist(),
        "shap_importance": shap_importance,
        "additivity_residuals": additivity,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
