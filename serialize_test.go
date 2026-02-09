package gboost

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	// Train a model
	X := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
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

	// Save to temp file
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	err := gbm.Save(path)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("model file was not created")
	}

	// Load the model
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Compare predictions
	originalPreds := gbm.Predict(X)
	loadedPreds := loaded.Predict(X)

	for i := range originalPreds {
		if math.Abs(originalPreds[i]-loadedPreds[i]) > 0.0001 {
			t.Errorf("prediction[%d]: original=%v, loaded=%v", i, originalPreds[i], loadedPreds[i])
		}
	}
}

func TestSaveLoadClassification(t *testing.T) {
	X := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0},
		{6.0}, {7.0}, {8.0}, {9.0},
	}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	// Compare probabilities
	originalProbs := gbm.PredictProbaAll(X)
	loadedProbs := loaded.PredictProbaAll(X)

	for i := range originalProbs {
		if math.Abs(originalProbs[i]-loadedProbs[i]) > 0.0001 {
			t.Errorf("probability[%d]: original=%v, loaded=%v", i, originalProbs[i], loadedProbs[i])
		}
	}
}

func TestSaveNotFitted(t *testing.T) {
	gbm := New(DefaultConfig())

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	err := gbm.Save(path)
	if err != ErrModelNotFitted {
		t.Errorf("Save unfitted model: got %v, want ErrModelNotFitted", err)
	}
}

func TestLoadNonexistent(t *testing.T) {
	_, err := Load("/nonexistent/path/model.json")
	if err == nil {
		t.Error("Load nonexistent file should return error")
	}
}

func TestSaveLoadConfig(t *testing.T) {
	cfg := Config{
		NEstimators:    42,
		LearningRate:   0.123,
		MaxDepth:       7,
		MinSamplesLeaf: 5,
		SubsampleRatio: 0.8,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	if loaded.Config.NEstimators != cfg.NEstimators {
		t.Errorf("NEstimators: got %d, want %d", loaded.Config.NEstimators, cfg.NEstimators)
	}
	if loaded.Config.LearningRate != cfg.LearningRate {
		t.Errorf("LearningRate: got %v, want %v", loaded.Config.LearningRate, cfg.LearningRate)
	}
	if loaded.Config.MaxDepth != cfg.MaxDepth {
		t.Errorf("MaxDepth: got %d, want %d", loaded.Config.MaxDepth, cfg.MaxDepth)
	}
	if loaded.Config.MinSamplesLeaf != cfg.MinSamplesLeaf {
		t.Errorf("MinSamplesLeaf: got %d, want %d", loaded.Config.MinSamplesLeaf, cfg.MinSamplesLeaf)
	}
	if loaded.Config.Loss != cfg.Loss {
		t.Errorf("Loss: got %s, want %s", loaded.Config.Loss, cfg.Loss)
	}
}

func TestSaveLoadPreservesTreeCount(t *testing.T) {
	cfg := Config{
		NEstimators:    15,
		LearningRate:   0.1,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}}
	y := []float64{1.0, 4.0, 9.0, 16.0, 25.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	if len(loaded.trees) != cfg.NEstimators {
		t.Errorf("tree count: got %d, want %d", len(loaded.trees), cfg.NEstimators)
	}
}

func TestJSONFormat(t *testing.T) {
	cfg := Config{
		NEstimators:    2,
		LearningRate:   0.5,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)

	// Read and verify it's valid JSON
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	// Should contain expected fields
	content := string(data)
	expectedFields := []string{
		`"config"`,
		`"initial_prediction"`,
		`"trees"`,
		`"feature_index"`,
		`"threshold"`,
		`"value"`,
	}

	for _, field := range expectedFields {
		if !contains(content, field) {
			t.Errorf("JSON missing field: %s", field)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
