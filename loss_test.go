package gboost

import (
	"math"
	"slices"
	"testing"
)

// ============ MSELoss Tests ============

func TestMSELossInitialPrediction(t *testing.T) {
	loss := &MSELoss{}

	tests := []struct {
		name     string
		y        []float64
		expected float64
	}{
		{
			name:     "simple mean",
			y:        []float64{1, 2, 3, 4, 5},
			expected: 3.0,
		},
		{
			name:     "all same values",
			y:        []float64{5, 5, 5, 5},
			expected: 5.0,
		},
		{
			name:     "negative values",
			y:        []float64{-2, -1, 0, 1, 2},
			expected: 0.0,
		},
		{
			name:     "single value",
			y:        []float64{42},
			expected: 42.0,
		},
		{
			name:     "decimals",
			y:        []float64{1.5, 2.5, 3.5},
			expected: 2.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := loss.InitialPrediction(tt.y)
			if math.Abs(got-tt.expected) > 0.0001 {
				t.Errorf("InitialPrediction(%v) = %v, want %v", tt.y, got, tt.expected)
			}
		})
	}
}

func TestMSELossNegativeGradient(t *testing.T) {
	loss := &MSELoss{}

	tests := []struct {
		name     string
		y        []float64
		pred     []float64
		expected []float64
	}{
		{
			name:     "perfect predictions",
			y:        []float64{1, 2, 3},
			pred:     []float64{1, 2, 3},
			expected: []float64{0, 0, 0},
		},
		{
			name:     "under predictions",
			y:        []float64{5, 10, 15},
			pred:     []float64{3, 8, 12},
			expected: []float64{2, 2, 3},
		},
		{
			name:     "over predictions",
			y:        []float64{1, 2, 3},
			pred:     []float64{3, 4, 5},
			expected: []float64{-2, -2, -2},
		},
		{
			name:     "mixed",
			y:        []float64{0, 5, 10},
			pred:     []float64{1, 5, 8},
			expected: []float64{-1, 0, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := loss.NegativeGradient(tt.y, tt.pred)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("NegativeGradient(%v, %v) = %v, want %v", tt.y, tt.pred, got, tt.expected)
			}
		})
	}
}

func TestMSELossHessian(t *testing.T) {
	loss := &MSELoss{}

	y := []float64{1, 2, 3, 4, 5}
	pred := []float64{0.5, 1.5, 2.5, 3.5, 4.5}
	got := loss.Hessian(y, pred)

	if len(got) != len(y) {
		t.Fatalf("len(Hessian) = %d, want %d", len(got), len(y))
	}
	for i, h := range got {
		if h != 1.0 {
			t.Errorf("Hessian[%d] = %v, want 1.0", i, h)
		}
	}
}

// ============ LogLoss Tests ============

func TestLogLossInitialPrediction(t *testing.T) {
	loss := &LogLoss{}

	tests := []struct {
		name        string
		y           []float64
		expectedMin float64
		expectedMax float64
	}{
		{
			name:        "balanced classes",
			y:           []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
			expectedMin: -0.01, // log(0.5/0.5) = 0
			expectedMax: 0.01,
		},
		{
			name:        "mostly positive",
			y:           []float64{1, 1, 1, 1, 0},
			expectedMin: 1.0, // log(0.8/0.2) ≈ 1.39
			expectedMax: 1.5,
		},
		{
			name:        "mostly negative",
			y:           []float64{0, 0, 0, 0, 1},
			expectedMin: -1.5, // log(0.2/0.8) ≈ -1.39
			expectedMax: -1.0,
		},
		{
			name:        "all positive - should be clipped",
			y:           []float64{1, 1, 1, 1, 1},
			expectedMin: 6.0, // log(0.999/0.001) ≈ 6.9
			expectedMax: 7.5,
		},
		{
			name:        "all negative - should be clipped",
			y:           []float64{0, 0, 0, 0, 0},
			expectedMin: -7.5, // log(0.001/0.999) ≈ -6.9
			expectedMax: -6.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := loss.InitialPrediction(tt.y)
			if math.IsNaN(got) || math.IsInf(got, 0) {
				t.Errorf("InitialPrediction(%v) = %v, want finite value", tt.y, got)
			}
			if got < tt.expectedMin || got > tt.expectedMax {
				t.Errorf("InitialPrediction(%v) = %v, want in range [%v, %v]", tt.y, got, tt.expectedMin, tt.expectedMax)
			}
		})
	}
}

func TestLogLossInitialPredictionNoNaN(t *testing.T) {
	loss := &LogLoss{}

	// Edge cases that could cause NaN or Inf without clipping
	edgeCases := [][]float64{
		{1, 1, 1, 1, 1},       // all positive
		{0, 0, 0, 0, 0},       // all negative
		{1},                   // single positive
		{0},                   // single negative
	}

	for _, y := range edgeCases {
		got := loss.InitialPrediction(y)
		if math.IsNaN(got) {
			t.Errorf("InitialPrediction(%v) = NaN, want finite value", y)
		}
		if math.IsInf(got, 0) {
			t.Errorf("InitialPrediction(%v) = Inf, want finite value", y)
		}
	}
}

func TestLogLossNegativeGradient(t *testing.T) {
	loss := &LogLoss{}

	tests := []struct {
		name string
		y    []float64
		pred []float64 // log-odds space
	}{
		{
			name: "predictions at zero log-odds",
			y:    []float64{0, 1, 0, 1},
			pred: []float64{0, 0, 0, 0}, // sigmoid(0) = 0.5
		},
		{
			name: "positive predictions",
			y:    []float64{1, 1, 1},
			pred: []float64{2, 2, 2}, // sigmoid(2) ≈ 0.88
		},
		{
			name: "negative predictions",
			y:    []float64{0, 0, 0},
			pred: []float64{-2, -2, -2}, // sigmoid(-2) ≈ 0.12
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := loss.NegativeGradient(tt.y, tt.pred)

			if len(got) != len(tt.y) {
				t.Fatalf("len(NegativeGradient) = %d, want %d", len(got), len(tt.y))
			}

			// Verify: gradient = y - sigmoid(pred)
			for i := range got {
				expected := tt.y[i] - sigmoid(tt.pred[i])
				if math.Abs(got[i]-expected) > 0.0001 {
					t.Errorf("gradient[%d] = %v, want %v", i, got[i], expected)
				}
			}
		})
	}
}

func TestLogLossNegativeGradientBounds(t *testing.T) {
	loss := &LogLoss{}

	// Gradient should always be in (-1, 1) since y is 0 or 1 and sigmoid is in (0, 1)
	y := []float64{0, 1, 0, 1, 0, 1}
	pred := []float64{-10, -5, 0, 5, 10, 100}

	got := loss.NegativeGradient(y, pred)

	for i, g := range got {
		if g <= -1 || g >= 1 {
			t.Errorf("gradient[%d] = %v, want in range (-1, 1)", i, g)
		}
	}
}

func TestLogLossNegativeGradientDirection(t *testing.T) {
	loss := &LogLoss{}

	// When y=1 and prediction is low (sigmoid < 0.5), gradient should be positive (push up)
	// When y=0 and prediction is high (sigmoid > 0.5), gradient should be negative (push down)

	t.Run("y=1 with low prediction should have positive gradient", func(t *testing.T) {
		y := []float64{1}
		pred := []float64{-2} // sigmoid(-2) ≈ 0.12
		got := loss.NegativeGradient(y, pred)
		if got[0] <= 0 {
			t.Errorf("gradient = %v, want positive", got[0])
		}
	})

	t.Run("y=0 with high prediction should have negative gradient", func(t *testing.T) {
		y := []float64{0}
		pred := []float64{2} // sigmoid(2) ≈ 0.88
		got := loss.NegativeGradient(y, pred)
		if got[0] >= 0 {
			t.Errorf("gradient = %v, want negative", got[0])
		}
	})

	t.Run("y=1 with high prediction should have small positive gradient", func(t *testing.T) {
		y := []float64{1}
		pred := []float64{5} // sigmoid(5) ≈ 0.99
		got := loss.NegativeGradient(y, pred)
		if got[0] <= 0 || got[0] > 0.1 {
			t.Errorf("gradient = %v, want small positive", got[0])
		}
	})

	t.Run("y=0 with low prediction should have small negative gradient", func(t *testing.T) {
		y := []float64{0}
		pred := []float64{-5} // sigmoid(-5) ≈ 0.01
		got := loss.NegativeGradient(y, pred)
		if got[0] >= 0 || got[0] < -0.1 {
			t.Errorf("gradient = %v, want small negative", got[0])
		}
	})
}

func TestLogLossHessian(t *testing.T) {
	loss := &LogLoss{}

	t.Run("values equal p*(1-p)", func(t *testing.T) {
		y := []float64{0, 1, 0, 1}
		pred := []float64{0, 0, 2, -2}
		got := loss.Hessian(y, pred)

		if len(got) != len(y) {
			t.Fatalf("len(Hessian) = %d, want %d", len(got), len(y))
		}
		for i := range got {
			p := sigmoid(pred[i])
			expected := p * (1 - p)
			if math.Abs(got[i]-expected) > 1e-10 {
				t.Errorf("Hessian[%d] = %v, want %v", i, got[i], expected)
			}
		}
	})

	t.Run("zero log-odds gives maximum hessian", func(t *testing.T) {
		// At pred=0, sigmoid=0.5, so hessian = 0.25 (the maximum)
		y := []float64{1}
		pred := []float64{0}
		got := loss.Hessian(y, pred)
		if math.Abs(got[0]-0.25) > 1e-10 {
			t.Errorf("Hessian at pred=0 = %v, want 0.25", got[0])
		}
	})

	t.Run("extreme predictions give small hessian", func(t *testing.T) {
		y := []float64{1, 0}
		pred := []float64{10, -10} // very confident predictions
		got := loss.Hessian(y, pred)
		for i, h := range got {
			if h >= 0.01 {
				t.Errorf("Hessian[%d] = %v, want small value for extreme prediction", i, h)
			}
			if h <= 0 {
				t.Errorf("Hessian[%d] = %v, want positive", i, h)
			}
		}
	})

	t.Run("hessian is independent of y", func(t *testing.T) {
		pred := []float64{1.5}
		h0 := loss.Hessian([]float64{0}, pred)
		h1 := loss.Hessian([]float64{1}, pred)
		if math.Abs(h0[0]-h1[0]) > 1e-10 {
			t.Errorf("Hessian should not depend on y: got %v vs %v", h0[0], h1[0])
		}
	})
}

// ============ Integration Tests ============

func TestLossInterfaceCompliance(t *testing.T) {
	// Ensure both loss types implement the Loss interface
	var _ Loss = &MSELoss{}
	var _ Loss = &LogLoss{}
}

func TestCreateLossFunction(t *testing.T) {
	tests := []struct {
		name     string
		lossName string
		wantType string
	}{
		{
			name:     "mse",
			lossName: "mse",
			wantType: "*gboost.MSELoss",
		},
		{
			name:     "logloss",
			lossName: "logloss",
			wantType: "*gboost.LogLoss",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{Loss: tt.lossName}
			loss := createLossFunction(cfg)
			if loss == nil {
				t.Fatal("createLossFunction returned nil")
			}
		})
	}
}

func TestCreateLossFunctionPanicsOnUnknown(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("createLossFunction did not panic on unknown loss")
		}
	}()

	cfg := Config{Loss: "unknown"}
	createLossFunction(cfg)
}
