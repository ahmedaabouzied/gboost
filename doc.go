// Package gboost implements gradient boosting machines from scratch in Go.
//
// It supports regression with mean squared error (MSE) and binary classification
// with log loss, using Newton-Raphson optimized leaf values for fast convergence
// and well-calibrated probabilities.
//
// # Quick Start
//
// Train a regression model:
//
//	cfg := gboost.DefaultConfig()
//	cfg.NEstimators = 50
//	model := gboost.New(cfg)
//	model.Fit(X, y)
//	predictions := model.Predict(X)
//
// Train a binary classifier:
//
//	cfg := gboost.DefaultConfig()
//	cfg.Loss = "logloss"
//	model := gboost.New(cfg)
//	model.Fit(X, y) // y values must be 0.0 or 1.0
//	probs := model.PredictProbaAll(X)
//
// # Loading Data
//
// Load a CSV file with automatic label encoding for non-numeric columns:
//
//	ds, err := gboost.LoadCSV("data.csv", -1, true) // -1 = last column is target
//	XTrain, XTest, yTrain, yTest, err := ds.Split(0.2, 42)
//
// # Persistence
//
// Save and load trained models as JSON:
//
//	model.Save("model.json")
//	loaded, err := gboost.Load("model.json")
//
// # SHAP Explanations
//
// Explain individual predictions with per-feature contributions computed by
// TreeSHAP (Lundberg 2018). SHAP values are additive: for every sample,
// sum(phi) + BaseValue == PredictSingle.
//
//	phi, err := model.ShapValuesSingle(x)    // []float64, one value per feature
//	base := model.BaseValue()                // expected model output over training
//
//	// Additivity check (holds exactly for raw model output):
//	sum := base
//	for _, v := range phi {
//	    sum += v
//	}
//	// sum == model.PredictSingle(x)
//
// For classification (Loss="logloss"), contributions are in log-odds space —
// additivity holds on the raw output, not on probabilities.
//
// Global SHAP-based importance: mean absolute contribution per feature across
// a dataset. Unlike gain-based [GBM.FeatureImportance], it reflects actual
// impact on predictions and is in the model's output units (not normalized):
//
//	imp, err := model.ShapImportance(X)
//	// Computing over a slice (e.g. positive-class samples only) yields a
//	// different ranking — slice-level explanations gain-based cannot produce.
package gboost
