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
package gboost
