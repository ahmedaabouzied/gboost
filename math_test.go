package gboost

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected float64
		epsilon  float64 // tolerance for comparison
	}{
		{
			name:     "zero returns 0.5",
			input:    0,
			expected: 0.5,
			epsilon:  0.0001,
		},
		{
			name:     "large positive approaches 1",
			input:    10,
			expected: 0.9999,
			epsilon:  0.001,
		},
		{
			name:     "large negative approaches 0",
			input:    -10,
			expected: 0.0001,
			epsilon:  0.001,
		},
		{
			name:     "positive value",
			input:    2,
			expected: 0.8808, // 1 / (1 + e^-2)
			epsilon:  0.001,
		},
		{
			name:     "negative value",
			input:    -2,
			expected: 0.1192, // 1 / (1 + e^2)
			epsilon:  0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sigmoid(tt.input)
			if math.Abs(got-tt.expected) > tt.epsilon {
				t.Errorf("sigmoid(%v) = %v, want %v (±%v)", tt.input, got, tt.expected, tt.epsilon)
			}
		})
	}
}

func TestSigmoidBounds(t *testing.T) {
	// Sigmoid should always return values between 0 and 1 (inclusive due to float precision)
	inputs := []float64{-100, -10, -1, 0, 1, 10, 100}

	for _, x := range inputs {
		got := sigmoid(x)
		if got < 0 || got > 1 {
			t.Errorf("sigmoid(%v) = %v, want value in [0, 1]", x, got)
		}
	}
}

func TestSigmoidSymmetry(t *testing.T) {
	// sigmoid(-x) = 1 - sigmoid(x)
	inputs := []float64{0.5, 1, 2, 5, 10}

	for _, x := range inputs {
		pos := sigmoid(x)
		neg := sigmoid(-x)
		sum := pos + neg

		if math.Abs(sum-1.0) > 0.0001 {
			t.Errorf("sigmoid(%v) + sigmoid(%v) = %v, want 1.0", x, -x, sum)
		}
	}
}
