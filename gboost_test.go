package gboost

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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

func TestConfigValidation(t *testing.T) {
	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr error
	}{
		{
			name:    "negative NEstimators",
			mutate:  func(c *Config) { c.NEstimators = -1 },
			wantErr: ErrInvalidNEstimators,
		},
		{
			name:    "zero LearningRate",
			mutate:  func(c *Config) { c.LearningRate = 0 },
			wantErr: ErrInvalidLearningRate,
		},
		{
			name:    "negative LearningRate",
			mutate:  func(c *Config) { c.LearningRate = -0.1 },
			wantErr: ErrInvalidLearningRate,
		},
		{
			name:    "zero MaxDepth",
			mutate:  func(c *Config) { c.MaxDepth = 0 },
			wantErr: ErrInvalidMaxDepth,
		},
		{
			name:    "zero MinSamplesLeaf",
			mutate:  func(c *Config) { c.MinSamplesLeaf = 0 },
			wantErr: ErrInvalidMinSamplesLeaf,
		},
		{
			name:    "zero SubsampleRatio",
			mutate:  func(c *Config) { c.SubsampleRatio = 0 },
			wantErr: ErrInvalidSubsampleRatio,
		},
		{
			name:    "SubsampleRatio > 1",
			mutate:  func(c *Config) { c.SubsampleRatio = 1.5 },
			wantErr: ErrInvalidSubsampleRatio,
		},
		{
			name:    "negative SubsampleRatio",
			mutate:  func(c *Config) { c.SubsampleRatio = -0.5 },
			wantErr: ErrInvalidSubsampleRatio,
		},
		{
			name:    "invalid Loss",
			mutate:  func(c *Config) { c.Loss = "huber" },
			wantErr: ErrInvalidLoss,
		},
		{
			name:    "empty Loss",
			mutate:  func(c *Config) { c.Loss = "" },
			wantErr: ErrInvalidLoss,
		},
		{
			name:   "valid default config",
			mutate: func(c *Config) {},
		},
		{
			name:   "valid NEstimators zero",
			mutate: func(c *Config) { c.NEstimators = 0 },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tt.mutate(&cfg)
			gbm := New(cfg)
			err := gbm.Fit(X, y)
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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
		SubsampleRatio: 1.0,
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

func TestFeatureImportanceNotFitted(t *testing.T) {
	gbm := New(DefaultConfig())
	imp := gbm.FeatureImportance()
	if len(imp) != 0 {
		t.Errorf("expected empty slice before Fit, got length %d", len(imp))
	}
}

func TestFeatureImportanceSumsToOne(t *testing.T) {
	X := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
		{4.0, 40.0},
		{5.0, 50.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	if len(imp) != 2 {
		t.Fatalf("expected 2 feature importances, got %d", len(imp))
	}

	total := 0.0
	for _, v := range imp {
		if v < 0 {
			t.Errorf("feature importance = %v, want >= 0", v)
		}
		total += v
	}
	if math.Abs(total-1.0) > 1e-10 {
		t.Errorf("feature importances sum to %v, want 1.0", total)
	}
}

func TestFeatureImportanceMatchesFeatureCount(t *testing.T) {
	X := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0}

	cfg := Config{
		NEstimators:    5,
		LearningRate:   0.3,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	if len(imp) != 3 {
		t.Errorf("expected 3 feature importances, got %d", len(imp))
	}
}

func TestFeatureImportanceDominantFeature(t *testing.T) {
	// y = x0 exactly. x1 is random noise. Feature 0 should dominate.
	X := [][]float64{
		{1.0, 99.0},
		{2.0, 3.0},
		{3.0, 55.0},
		{4.0, 12.0},
		{5.0, 77.0},
		{6.0, 41.0},
		{7.0, 8.0},
		{8.0, 63.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	if imp[0] <= imp[1] {
		t.Errorf("expected feature 0 (%.4f) > feature 1 (%.4f) since y = x0", imp[0], imp[1])
	}
}

func TestFeatureImportanceIrrelevantFeature(t *testing.T) {
	// y depends only on x0. x1 is constant (no variance, can never split on it).
	X := [][]float64{
		{1.0, 5.0},
		{2.0, 5.0},
		{3.0, 5.0},
		{4.0, 5.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	if imp[1] != 0 {
		t.Errorf("expected feature 1 importance = 0 (constant feature), got %v", imp[1])
	}
	if imp[0] != 1.0 {
		t.Errorf("expected feature 0 importance = 1.0 (only feature used), got %v", imp[0])
	}
}

func TestFeatureImportanceZeroEstimators(t *testing.T) {
	// With 0 estimators, no trees, no splits, all importances should be 0
	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	cfg := Config{
		NEstimators:    0,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	for i, v := range imp {
		if v != 0 {
			t.Errorf("imp[%d] = %v, want 0 with no estimators", i, v)
		}
	}
}

func TestSameSeedSameModel(t *testing.T) {
	X := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
		{5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
		{9.0, 10.0}, {10.0, 11.0},
	}
	y := []float64{3, 5, 7, 9, 11, 13, 15, 17, 19, 21}

	cfg := Config{
		Seed:           42,
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 0.8,
		Loss:           "mse",
	}

	gbm1 := New(cfg)
	gbm1.Fit(X, y)
	preds1 := gbm1.Predict(X)

	gbm2 := New(cfg)
	gbm2.Fit(X, y)
	preds2 := gbm2.Predict(X)

	for i := range preds1 {
		if preds1[i] != preds2[i] {
			t.Errorf("prediction[%d]: gbm1=%v, gbm2=%v — same seed should produce identical results", i, preds1[i], preds2[i])
		}
	}
}

func TestDifferentSeedDifferentModel(t *testing.T) {
	X := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
		{5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
		{9.0, 10.0}, {10.0, 11.0},
	}
	y := []float64{3, 5, 7, 9, 11, 13, 15, 17, 19, 21}

	cfg1 := Config{
		Seed:           42,
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 0.8,
		Loss:           "mse",
	}
	cfg2 := cfg1
	cfg2.Seed = 99

	gbm1 := New(cfg1)
	gbm1.Fit(X, y)
	preds1 := gbm1.Predict(X)

	gbm2 := New(cfg2)
	gbm2.Fit(X, y)
	preds2 := gbm2.Predict(X)

	allSame := true
	for i := range preds1 {
		if preds1[i] != preds2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("different seeds produced identical predictions — randomness is not seed-dependent")
	}
}

func TestRefitResetsSeed(t *testing.T) {
	X := [][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
		{5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
		{9.0, 10.0}, {10.0, 11.0},
	}
	y := []float64{3, 5, 7, 9, 11, 13, 15, 17, 19, 21}

	cfg := Config{
		Seed:           42,
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 0.8,
		Loss:           "mse",
	}

	// Fit once
	gbm := New(cfg)
	gbm.Fit(X, y)

	// Fit again on the same instance — RNG should reset
	gbm.Fit(X, y)
	predsRefit := gbm.Predict(X)

	// Fresh instance with same config
	fresh := New(cfg)
	fresh.Fit(X, y)
	predsFresh := fresh.Predict(X)

	for i := range predsRefit {
		if predsRefit[i] != predsFresh[i] {
			t.Errorf("prediction[%d]: refit=%v, fresh=%v — re-fitting should reset RNG", i, predsRefit[i], predsFresh[i])
		}
	}
}

func TestFeatureImportanceClassification(t *testing.T) {
	// Binary classification: class determined by x0 > 5
	X := [][]float64{
		{1.0, 50.0}, {2.0, 40.0}, {3.0, 30.0}, {4.0, 20.0},
		{6.0, 10.0}, {7.0, 60.0}, {8.0, 70.0}, {9.0, 80.0},
	}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	imp := gbm.FeatureImportance()
	if len(imp) != 2 {
		t.Fatalf("expected 2 importances, got %d", len(imp))
	}

	total := 0.0
	for _, v := range imp {
		total += v
	}
	if math.Abs(total-1.0) > 1e-10 {
		t.Errorf("importances sum to %v, want 1.0", total)
	}

	if imp[0] <= imp[1] {
		t.Errorf("expected feature 0 (%.4f) > feature 1 (%.4f) for classification on x0", imp[0], imp[1])
	}
}

func TestConvergence(t *testing.T) {
	X, y := generateLinearData()
	model := New(DefaultConfig())

	err := model.Fit(X, y)
	assert.NoError(t, err)

	preds := make([]float64, len(X))
	for i := range preds {
		preds[i] = model.initialPrediction
	}

	lastMse := math.Inf(1)

	for _, tree := range model.trees {
		for j := range X {
			preds[j] += model.Config.LearningRate * tree.predict(X[j])
		}
		mse := mse(preds, y)
		assert.LessOrEqual(t, mse, lastMse)
		lastMse = mse
	}
}

func TestOverfitting(t *testing.T) {
	X, y := generateLinearData()

	config := DefaultConfig()
	config.MaxDepth = 10
	config.NEstimators = 200

	model := New(config)
	assert.NoError(t, model.Fit(X, y))
	preds := model.Predict(X)
	assert.True(t, (mse(preds, y) < 1e-6))
	// Model is over-fitted.
}

func TestIdenticalTargets(t *testing.T) {
	X, _ := generateLinearData()
	y := make([]float64, len(X))
	for i := range y {
		y[i] = 5.0
	}

	model := New(DefaultConfig())
	assert.NoError(t, model.Fit(X, y))

	X_test, _ := generateLinearData()
	preds := model.Predict(X_test)
	for _, pred := range preds {
		assert.Equal(t, 5.0, pred)
	}
}

func TestIdenticalFeatures(t *testing.T) {
	_, y := generateLinearData()
	// Build an identical features data set
	X := make([][]float64, len(y))
	for i := range X {
		X[i] = []float64{1.0, 2.0}
	}

	model := New(DefaultConfig())
	assert.NoError(t, model.Fit(X, y))

	X_test, _ := generateLinearData()
	preds := model.Predict(X_test)
	mean_y := mean(y)
	for _, pred := range preds {
		assert.Equal(t, mean_y, pred)
	}
}

func TestDataWithSingleFeature(t *testing.T) {
	X, y := generateLinearDataWithSingleFeature()

	model := New(DefaultConfig())
	assert.NoError(t, model.Fit(X, y))

	assert.True(t, mse(model.Predict(X), y) < 1.0)
}

func TestDataWithSingleSample(t *testing.T) {
	X := [][]float64{{0.1, 0.2, 0.3}}
	y := []float64{1.2}

	model := New(DefaultConfig())
	assert.NoError(t, model.Fit(X, y))

	assert.True(t, mse(model.Predict(X), y) < 1.0)
}

func mse(x, y []float64) float64 {
	mse := 0.0
	for j := range y {
		diff := x[j] - y[j]
		mse += diff * diff
	}
	mse /= float64(len(y))
	return mse
}

func generateLinearData() ([][]float64, []float64) {
	X := make([][]float64, 50)
	y := make([]float64, 50)

	rnd := rand.New(rand.NewSource(0))

	f := func(x1, x2 float64) float64 {
		return 2*x1 + 3*x2
	}

	for i := range 50 {
		x1 := rnd.Float64()
		x2 := rnd.Float64()
		y[i] = f(x1, x2)
		X[i] = []float64{x1, x2}
	}

	return X, y
}

func generateLinearDataWithSingleFeature() ([][]float64, []float64) {
	X := make([][]float64, 50)
	y := make([]float64, 50)

	rnd := rand.New(rand.NewSource(0))

	// One feature only
	f := func(x1 float64) float64 {
		return 2 * x1
	}

	for i := range 50 {
		x1 := rnd.Float64()
		y[i] = f(x1)
		X[i] = []float64{x1}
	}

	return X, y
}
