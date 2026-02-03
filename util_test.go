package gboost

import "testing"

func TestHasSimilarLength(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]float64
		expected bool
	}{
		{
			name:     "all rows same length",
			input:    [][]float64{{1, 2}, {3, 4}, {5, 6}},
			expected: true,
		},
		{
			name:     "different lengths",
			input:    [][]float64{{1, 2}, {3, 4, 5}},
			expected: false,
		},
		{
			name:     "single row",
			input:    [][]float64{{1, 2, 3}},
			expected: true,
		},
		{
			name:     "first row longer",
			input:    [][]float64{{1, 2, 3}, {4, 5}},
			expected: false,
		},
		{
			name:     "first row shorter",
			input:    [][]float64{{1}, {2, 3}},
			expected: false,
		},
		{
			name:     "empty rows all same",
			input:    [][]float64{{}, {}, {}},
			expected: true,
		},
		{
			name:     "single empty row",
			input:    [][]float64{{}},
			expected: true,
		},
		{
			name:     "many rows same length",
			input:    [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			expected: true,
		},
		{
			name:     "mismatch in middle",
			input:    [][]float64{{1, 2}, {3, 4}, {5}, {6, 7}},
			expected: false,
		},
		{
			name:     "mismatch at end",
			input:    [][]float64{{1, 2}, {3, 4}, {5, 6, 7}},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasSimilarLength(tt.input)
			if got != tt.expected {
				t.Errorf("hasSimilarLength(%v) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
