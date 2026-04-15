package gboost

import (
	"context"
	"encoding/json"
	"flag"
	"math"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	sklearnFlag = flag.Bool("sklearn", false, "run sklearn parity tests")
	shapFlag    = flag.Bool("shap", false, "run shap parity tests")
)

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

type shapParityResult struct {
	FeatureNames        []string    `json:"feature_names"`
	BaseValue           float64     `json:"base_value"`
	ShapValues          [][]float64 `json:"shap_values"`
	ShapImportance      []float64   `json:"shap_importance"`
	AdditivityResiduals []float64   `json:"additivity_residuals"`
}

type goShapOutput struct {
	BaseValue      float64     `json:"base_value"`
	ShapValues     [][]float64 `json:"shap_values"`
	FeatureNames   []string    `json:"feature_names"`
	ShapImportance []float64   `json:"shap_importance"`
}

// TestShapParity runs gboost's SHAP implementation and shap.TreeExplainer on
// separately-trained models (same hyperparameters, different RNG paths) and
// verifies algorithmic behavior is consistent. Because trees differ, we
// cannot compare sample-by-sample; we compare:
//
//  1. sklearn-side additivity (sanity of the Python setup)
//  2. feature importance ranking agreement
//  3. per-sample sign agreement on the dominant feature
//
// Gated by -shap flag (like -sklearn).
func TestShapParity(t *testing.T) {
	if !*shapFlag {
		t.Skip("shap parity test requires -shap flag")
	}

	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not found in PATH")
	}

	// Ensure Go artifacts exist by running cmd/iris, which fits the model,
	// writes iris_train.csv / iris_test.csv and iris_go_shap.json.
	ctxBuild, cancelBuild := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancelBuild()
	build := exec.CommandContext(ctxBuild, "go", "run", "./cmd/iris")
	buildOut, err := build.CombinedOutput()
	require.NoError(t, err, "cmd/iris run failed: %s", string(buildOut))

	// Run the Python shap script.
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	pyCmd := exec.CommandContext(ctx, "uv", "run", "--project", "e2e_tests",
		"e2e_tests/shap_parity.py", "--data-dir", "data")
	pyOut, err := pyCmd.Output()
	require.NoError(t, err, "shap_parity.py failed: %s", string(pyOut))

	var sk shapParityResult
	require.NoError(t, json.Unmarshal(pyOut, &sk), "python JSON: %s", string(pyOut))

	// Load Go-side SHAP output.
	goBytes, err := os.ReadFile("data/iris_go_shap.json")
	require.NoError(t, err)
	var goShap goShapOutput
	require.NoError(t, json.Unmarshal(goBytes, &goShap))

	require.Equal(t, len(sk.FeatureNames), len(goShap.FeatureNames),
		"feature count differs")
	require.Equal(t, len(sk.ShapValues), len(goShap.ShapValues),
		"test sample count differs")

	// 1. sklearn-side additivity should be ~0.
	for i, r := range sk.AdditivityResiduals {
		assert.Less(t, math.Abs(r), 1e-6,
			"sklearn shap additivity residual too large at sample %d: %v", i, r)
	}

	// 2. Feature importance ranking agreement.
	rankGo := rankDescending(goShap.ShapImportance)
	rankSk := rankDescending(sk.ShapImportance)
	t.Logf("Go SHAP importance ranking:      %v (values: %v)", rankGo, goShap.ShapImportance)
	t.Logf("sklearn SHAP importance ranking: %v (values: %v)", rankSk, sk.ShapImportance)
	assert.Equal(t, rankGo[0], rankSk[0],
		"top feature disagrees: go=%d sklearn=%d", rankGo[0], rankSk[0])

	// 3. Per-sample sign agreement on the top feature.
	topFeat := rankSk[0]
	agree := 0
	for i := range sk.ShapValues {
		if sameSign(goShap.ShapValues[i][topFeat], sk.ShapValues[i][topFeat]) {
			agree++
		}
	}
	agreementRate := float64(agree) / float64(len(sk.ShapValues))
	t.Logf("sign agreement on top feature: %d / %d (%.1f%%)", agree, len(sk.ShapValues), agreementRate*100)
	assert.GreaterOrEqual(t, agreementRate, 0.85,
		"sign agreement on top feature too low: %.1f%%", agreementRate*100)
}

func rankDescending(xs []float64) []int {
	idx := make([]int, len(xs))
	for i := range idx {
		idx[i] = i
	}
	// Simple insertion sort by |xs|, descending — n<=10 here.
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0 && math.Abs(xs[idx[j]]) > math.Abs(xs[idx[j-1]]); j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	return idx
}

func sameSign(a, b float64) bool {
	return (a >= 0) == (b >= 0)
}
