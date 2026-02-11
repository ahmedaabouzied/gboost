package gboost

import (
	"context"
	"encoding/json"
	"flag"
	"math"
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var sklearnFlag = flag.Bool("sklearn", false, "run sklearn parity tests")

type sklearnResult struct {
	TestProbabilities []float64 `json:"test_probabilities"`
	TestPredictions   []float64 `json:"test_predictions"`
	TestAccuracy      float64   `json:"test_accuracy"`
	TrainAccuracy     float64   `json:"train_accuracy"`
}

func TestSklearnParity(t *testing.T) {
	if !*sklearnFlag {
		t.Skip("sklearn parity test requires -sklearn flag")
	}

	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not found in PATH, skipping sklearn parity test")
	}

	// Run the Python script via uv with a 120s timeout (first run downloads deps).
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "uv", "run", "--project", "e2e_tests", "e2e_tests/sklearn_parity.py", "--data-dir", "data")
	out, err := cmd.Output()
	require.NoError(t, err, "sklearn script failed: %s", string(out))

	var skResult sklearnResult
	require.NoError(t, json.Unmarshal(out, &skResult), "failed to parse sklearn JSON output: %s", string(out))

	// Load the same train/test CSVs in Go.
	trainDS, err := LoadCSV("data/iris_train.csv", -1, true)
	require.NoError(t, err)

	testDS, err := LoadCSV("data/iris_test.csv", -1, true)
	require.NoError(t, err)

	// Train Go model with matching hyperparameters.
	cfg := DefaultConfig()
	cfg.NEstimators = 100
	cfg.LearningRate = 0.1
	cfg.MaxDepth = 3
	cfg.MinSamplesLeaf = 1
	cfg.SubsampleRatio = 1.0
	cfg.Loss = "logloss"
	cfg.Seed = 42

	model := New(cfg)
	require.NoError(t, model.Fit(trainDS.X, trainDS.Y))

	goProbs := model.PredictProbaAll(testDS.X)

	// Sanity: all Go probabilities must be in (0, 1).
	for i, p := range goProbs {
		assert.Greater(t, p, 0.0, "goProbs[%d] should be > 0", i)
		assert.Less(t, p, 1.0, "goProbs[%d] should be < 1", i)
	}

	// Per-sample probability tolerance.
	require.Equal(t, len(skResult.TestProbabilities), len(goProbs), "test set size mismatch")
	for i := range goProbs {
		diff := math.Abs(goProbs[i] - skResult.TestProbabilities[i])
		assert.Less(t, diff, 0.15, "probability mismatch at sample %d: go=%.4f sklearn=%.4f", i, goProbs[i], skResult.TestProbabilities[i])
	}

	// Class disagreements: at most 2 out of test samples.
	disagreements := 0
	for i := range goProbs {
		goPred := 0.0
		if goProbs[i] >= 0.5 {
			goPred = 1.0
		}
		if goPred != skResult.TestPredictions[i] {
			disagreements++
		}
	}
	assert.LessOrEqual(t, disagreements, 2, "too many class disagreements: %d", disagreements)

	// Compute Go accuracy.
	correct := 0
	for i, p := range goProbs {
		pred := 0.0
		if p >= 0.5 {
			pred = 1.0
		}
		if pred == testDS.Y[i] {
			correct++
		}
	}
	goAccuracy := float64(correct) / float64(len(testDS.Y))

	// Accuracy difference.
	accDiff := math.Abs(goAccuracy - skResult.TestAccuracy)
	assert.Less(t, accDiff, 0.10, "accuracy difference too large: go=%.2f sklearn=%.2f", goAccuracy, skResult.TestAccuracy)

	// Sanity: Go accuracy should be at least 0.7.
	assert.GreaterOrEqual(t, goAccuracy, 0.7, "Go accuracy too low: %.2f", goAccuracy)

	t.Logf("Go accuracy: %.2f, sklearn accuracy: %.2f, disagreements: %d/%d",
		goAccuracy, skResult.TestAccuracy, disagreements, len(goProbs))
}
