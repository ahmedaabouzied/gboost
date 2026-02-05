package gboost

import (
	"math"
	"testing"
)

func TestGBMFitPredict(t *testing.T) {
	// Simple linear relationship: y = x
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
		{5.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.5,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	gbm := New(cfg)
	err := gbm.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !gbm.isFitted {
		t.Error("expected isFitted to be true after Fit")
	}

	if len(gbm.trees) != cfg.NEstimators {
		t.Errorf("expected %d trees, got %d", cfg.NEstimators, len(gbm.trees))
	}

	// Predict on training data - should be reasonably close
	predictions := gbm.Predict(X)
	for i, pred := range predictions {
		diff := math.Abs(pred - y[i])
		if diff > 0.5 {
			t.Errorf("prediction[%d] = %.2f, want close to %.2f (diff=%.2f)", i, pred, y[i], diff)
		}
	}
}

func TestGBMFitPredictNonLinear(t *testing.T) {
	// Non-linear: y = x^2
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
		{5.0},
	}
	y := []float64{1.0, 4.0, 9.0, 16.0, 25.0}

	cfg := Config{
		NEstimators:    50,
		LearningRate:   0.3,
		MaxDepth:       4,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	gbm := New(cfg)
	err := gbm.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	predictions := gbm.Predict(X)

	// Calculate mean squared error
	var mse float64
	for i, pred := range predictions {
		diff := pred - y[i]
		mse += diff * diff
	}
	mse /= float64(len(y))

	// MSE should be reasonably low for training data
	if mse > 5.0 {
		t.Errorf("MSE = %.2f, expected < 5.0", mse)
	}
}

func TestGBMMultipleFeatures(t *testing.T) {
	// y = x1 + x2
	X := [][]float64{
		{1.0, 1.0},
		{2.0, 1.0},
		{1.0, 2.0},
		{2.0, 2.0},
		{3.0, 3.0},
	}
	y := []float64{2.0, 3.0, 3.0, 4.0, 6.0}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	gbm := New(cfg)
	err := gbm.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	predictions := gbm.Predict(X)

	for i, pred := range predictions {
		diff := math.Abs(pred - y[i])
		if diff > 1.0 {
			t.Errorf("prediction[%d] = %.2f, want close to %.2f", i, pred, y[i])
		}
	}
}

func TestGBMPredictSingle(t *testing.T) {
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
	}
	y := []float64{10.0, 20.0, 30.0}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.5,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	// PredictSingle should match Predict for same input
	pred := gbm.PredictSingle([]float64{2.0})
	preds := gbm.Predict([][]float64{{2.0}})

	if pred != preds[0] {
		t.Errorf("PredictSingle = %v, Predict[0] = %v, should match", pred, preds[0])
	}
}

func TestGBMValidation(t *testing.T) {
	gbm := New(DefaultConfig())

	tests := []struct {
		name    string
		X       [][]float64
		y       []float64
		wantErr error
	}{
		{
			name:    "empty dataset",
			X:       [][]float64{},
			y:       []float64{},
			wantErr: ErrEmptyDataset,
		},
		{
			name:    "empty features",
			X:       [][]float64{{}},
			y:       []float64{1.0},
			wantErr: ErrEmptyFeatures,
		},
		{
			name:    "length mismatch",
			X:       [][]float64{{1.0}, {2.0}},
			y:       []float64{1.0},
			wantErr: ErrLengthMismatch,
		},
		{
			name:    "feature count mismatch",
			X:       [][]float64{{1.0, 2.0}, {3.0}},
			y:       []float64{1.0, 2.0},
			wantErr: ErrFeatureCountMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := gbm.Fit(tt.X, tt.y)
			if err != tt.wantErr {
				t.Errorf("Fit() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestGBMInitialPrediction(t *testing.T) {
	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{10.0, 20.0, 30.0}

	cfg := Config{
		NEstimators:    0, // no trees
		LearningRate:   0.1,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	// With 0 estimators, prediction should be just the initial prediction (mean)
	expectedMean := 20.0
	pred := gbm.PredictSingle([]float64{1.0})

	if math.Abs(pred-expectedMean) > 0.01 {
		t.Errorf("with 0 trees, prediction = %v, want %v (mean of y)", pred, expectedMean)
	}
}

func TestGBMClassification(t *testing.T) {
	// Binary classification: class 1 if x > 5, else class 0
	X := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, // class 0
		{6.0}, {7.0}, {8.0}, {9.0}, // class 1
	}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	err := gbm.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Check probabilities are in valid range
	probs := gbm.PredictProbaAll(X)
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probability[%d] = %v, want in [0, 1]", i, p)
		}
	}

	// Class 0 samples should have low probability
	for i := 0; i < 4; i++ {
		if probs[i] > 0.5 {
			t.Errorf("class 0 sample %d has probability %v, want < 0.5", i, probs[i])
		}
	}

	// Class 1 samples should have high probability
	for i := 4; i < 8; i++ {
		if probs[i] < 0.5 {
			t.Errorf("class 1 sample %d has probability %v, want > 0.5", i, probs[i])
		}
	}
}

func TestGBMPredictProba(t *testing.T) {
	X := [][]float64{{1.0}, {5.0}, {9.0}}
	y := []float64{0, 0, 1}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	// PredictProba should return sigmoid of PredictSingle
	for _, x := range X {
		rawPred := gbm.PredictSingle(x)
		proba := gbm.PredictProba(x)
		expectedProba := 1.0 / (1.0 + math.Exp(-rawPred))

		if math.Abs(proba-expectedProba) > 0.0001 {
			t.Errorf("PredictProba(%v) = %v, want sigmoid(%v) = %v", x, proba, rawPred, expectedProba)
		}
	}
}

func TestGBMPredictProbaAll(t *testing.T) {
	X := [][]float64{{1.0}, {5.0}, {9.0}}
	y := []float64{0, 1, 1}

	cfg := Config{
		NEstimators:    5,
		LearningRate:   0.3,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	// PredictProbaAll should match individual PredictProba calls
	probas := gbm.PredictProbaAll(X)

	for i, x := range X {
		expected := gbm.PredictProba(x)
		if probas[i] != expected {
			t.Errorf("PredictProbaAll[%d] = %v, want %v", i, probas[i], expected)
		}
	}
}

func TestGBMPredictProbaBounds(t *testing.T) {
	// Even with extreme predictions, probabilities should be in (0, 1)
	X := [][]float64{{0.0}, {100.0}}
	y := []float64{0, 1}

	cfg := Config{
		NEstimators:    50,
		LearningRate:   0.5,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	testInputs := [][]float64{{-100.0}, {0.0}, {50.0}, {100.0}, {200.0}}
	for _, x := range testInputs {
		p := gbm.PredictProba(x)
		if p <= 0 || p >= 1 {
			t.Errorf("PredictProba(%v) = %v, want in (0, 1)", x, p)
		}
	}
}
