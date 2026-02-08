package gboost

import (
	"math"
	"slices"
	"testing"
)

func TestSort(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{
			name:     "already sorted",
			input:    []float64{1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "reverse order",
			input:    []float64{3, 2, 1},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "single element",
			input:    []float64{42},
			expected: []float64{42},
		},
		{
			name:     "empty slice",
			input:    []float64{},
			expected: []float64{},
		},
		{
			name:     "duplicates",
			input:    []float64{3, 1, 2, 1, 3},
			expected: []float64{1, 1, 2, 3, 3},
		},
		{
			name:     "negative values",
			input:    []float64{-1, 5, -3, 0, 2},
			expected: []float64{-3, -1, 0, 2, 5},
		},
		{
			name:     "floats with decimals",
			input:    []float64{1.5, 1.1, 1.9, 1.3},
			expected: []float64{1.1, 1.3, 1.5, 1.9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := slices.Clone(tt.input) // clone to test mutation separately
			got := sort(input)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("sort(%v) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestSortMutatesInput(t *testing.T) {
	input := []float64{3, 1, 2}
	original := slices.Clone(input)
	sort(input)

	if slices.Equal(input, original) {
		t.Log("sort() did not mutate input - this is fine if intentional")
	} else {
		t.Log("sort() mutates input in-place - callers should be aware")
	}
}

func TestUniq(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64 // must be sorted
		expected []float64
	}{
		{
			name:     "no duplicates",
			input:    []float64{1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "all duplicates",
			input:    []float64{1, 1, 1},
			expected: []float64{1},
		},
		{
			name:     "some duplicates sorted",
			input:    []float64{1, 1, 2, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "single element",
			input:    []float64{42},
			expected: []float64{42},
		},
		{
			name:     "empty slice",
			input:    []float64{},
			expected: []float64{},
		},
		{
			name:     "negative values sorted",
			input:    []float64{-3, -1, -1, 2, 2, 3},
			expected: []float64{-3, -1, 2, 3},
		},
		{
			name:     "duplicates at start",
			input:    []float64{1, 1, 1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "duplicates at end",
			input:    []float64{1, 2, 3, 3, 3},
			expected: []float64{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := uniq(tt.input)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("uniq(%v) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestUniqLength(t *testing.T) {
	input := []float64{1, 1, 2, 2, 3, 3} // sorted
	got := uniq(input)
	if len(got) != 3 {
		t.Errorf("uniq(%v) has length %d, want 3", input, len(got))
	}
}

func TestSortThenUniq(t *testing.T) {
	input := []float64{3, 1, 2, 1, 3, 2}
	got := uniq(sort(input)) // sort first, then uniq
	expected := []float64{1, 2, 3}
	if !slices.Equal(got, expected) {
		t.Errorf("uniq(sort(%v)) = %v, want %v", input, got, expected)
	}
}

func TestExtract(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		indices  []int
		expected []float64
	}{
		{
			name:     "extract subset",
			data:     []float64{10, 20, 30, 40, 50},
			indices:  []int{0, 2, 4},
			expected: []float64{10, 30, 50},
		},
		{
			name:     "extract all",
			data:     []float64{1, 2, 3},
			indices:  []int{0, 1, 2},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "extract single",
			data:     []float64{1, 2, 3},
			indices:  []int{1},
			expected: []float64{2},
		},
		{
			name:     "extract none",
			data:     []float64{1, 2, 3},
			indices:  []int{},
			expected: []float64{},
		},
		{
			name:     "non-sequential indices",
			data:     []float64{10, 20, 30, 40, 50},
			indices:  []int{4, 1, 3},
			expected: []float64{50, 20, 40},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractRows(tt.data, tt.indices)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("extractRows(%v, %v) = %v, want %v", tt.data, tt.indices, got, tt.expected)
			}
		})
	}
}

func TestPartition(t *testing.T) {
	X := [][]float64{
		{1.0, 5.0},
		{2.0, 4.0},
		{3.0, 3.0},
		{4.0, 2.0},
		{5.0, 1.0},
	}

	tests := []struct {
		name          string
		indices       []int
		featureIndex  int
		threshold     float64
		expectedLeft  []int
		expectedRight []int
	}{
		{
			name:          "split on feature 0, threshold 3",
			indices:       []int{0, 1, 2, 3, 4},
			featureIndex:  0,
			threshold:     3.0,
			expectedLeft:  []int{0, 1},       // values 1, 2 < 3
			expectedRight: []int{2, 3, 4},    // values 3, 4, 5 >= 3
		},
		{
			name:          "split on feature 1, threshold 3",
			indices:       []int{0, 1, 2, 3, 4},
			featureIndex:  1,
			threshold:     3.0,
			expectedLeft:  []int{3, 4},       // values 2, 1 < 3
			expectedRight: []int{0, 1, 2},    // values 5, 4, 3 >= 3
		},
		{
			name:          "split subset of indices",
			indices:       []int{1, 3},
			featureIndex:  0,
			threshold:     3.0,
			expectedLeft:  []int{1},          // value 2 < 3
			expectedRight: []int{3},          // value 4 >= 3
		},
		{
			name:          "all go left",
			indices:       []int{0, 1},
			featureIndex:  0,
			threshold:     10.0,
			expectedLeft:  []int{0, 1},
			expectedRight: []int{},
		},
		{
			name:          "all go right",
			indices:       []int{3, 4},
			featureIndex:  0,
			threshold:     1.0,
			expectedLeft:  []int{},
			expectedRight: []int{3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			left, right := partition(X, tt.indices, tt.featureIndex, tt.threshold)
			if !slices.Equal(left, tt.expectedLeft) {
				t.Errorf("left = %v, want %v", left, tt.expectedLeft)
			}
			if !slices.Equal(right, tt.expectedRight) {
				t.Errorf("right = %v, want %v", right, tt.expectedRight)
			}
		})
	}
}

func TestFindBestSplit(t *testing.T) {
	// Simple dataset: y increases with X[0]
	// Best split should be on feature 0
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
	}
	y := []float64{1.0, 2.0, 10.0, 11.0} // clear split between indices 1 and 2
	indices := []int{0, 1, 2, 3}

	split := findBestSplit(X, y, indices, 1)

	if split == nil {
		t.Fatal("expected a split, got nil")
	}

	if split.FeatureIndex != 0 {
		t.Errorf("FeatureIndex = %d, want 0", split.FeatureIndex)
	}

	// Threshold should split [1,2] from [10,11], so threshold should be 3.0
	if split.Threshold != 3.0 {
		t.Errorf("Threshold = %v, want 3.0", split.Threshold)
	}

	if !slices.Equal(split.LeftIndices, []int{0, 1}) {
		t.Errorf("LeftIndices = %v, want [0, 1]", split.LeftIndices)
	}

	if !slices.Equal(split.RightIndices, []int{2, 3}) {
		t.Errorf("RightIndices = %v, want [2, 3]", split.RightIndices)
	}

	if split.Gain <= 0 {
		t.Errorf("Gain = %v, want > 0", split.Gain)
	}
}

func TestFindBestSplitNoValidSplit(t *testing.T) {
	// All same values - no valid split possible
	X := [][]float64{
		{1.0},
		{1.0},
	}
	y := []float64{5.0, 5.0}
	indices := []int{0, 1}

	split := findBestSplit(X, y, indices, 1)

	if split != nil {
		t.Errorf("expected nil split for identical data, got %+v", split)
	}
}

func TestFindBestSplitMinSamplesLeaf(t *testing.T) {
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
	}
	y := []float64{1.0, 2.0, 10.0}
	indices := []int{0, 1, 2}

	// With minSamplesLeaf=2, the only valid split is [0,1] vs [2]
	// but [2] has only 1 sample, so no valid split
	split := findBestSplit(X, y, indices, 2)

	if split != nil {
		// Check that both sides have at least 2 samples
		if len(split.LeftIndices) < 2 || len(split.RightIndices) < 2 {
			t.Errorf("split violates minSamplesLeaf: left=%d, right=%d",
				len(split.LeftIndices), len(split.RightIndices))
		}
	}
}

func TestBuildTree(t *testing.T) {
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
	}
	y := []float64{1.0, 1.0, 10.0, 10.0}
	indices := []int{0, 1, 2, 3}
	cfg := Config{
		MaxDepth:       3,
		MinSamplesLeaf: 1,
	}

	hessians := make([]float64, len(y))
	for i := range hessians {
		hessians[i] = 1.0
	}

	tree := buildTree(X, y, hessians, indices, 0, cfg)

	if tree == nil {
		t.Fatal("expected a tree, got nil")
	}

	// Should be an internal node (has children)
	if tree.Left == nil || tree.Right == nil {
		t.Error("expected internal node with children")
	}

	// Children should be leaf nodes
	if tree.Left.Left != nil || tree.Left.Right != nil {
		t.Error("expected left child to be a leaf")
	}
	if tree.Right.Left != nil || tree.Right.Right != nil {
		t.Error("expected right child to be a leaf")
	}

	// Check leaf values: with hessians=1.0, sum(y)/sum(h) = mean(y)
	if tree.Left.Value != 1.0 {
		t.Errorf("left leaf value = %v, want 1.0", tree.Left.Value)
	}
	if tree.Right.Value != 10.0 {
		t.Errorf("right leaf value = %v, want 10.0", tree.Right.Value)
	}
}

func TestBuildTreeMaxDepth(t *testing.T) {
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0}
	indices := []int{0, 1, 2, 3}
	cfg := Config{
		MaxDepth:       0, // force immediate leaf
		MinSamplesLeaf: 1,
	}
	hessians := make([]float64, len(y))
	for i := range hessians {
		hessians[i] = 1.0
	}

	tree := buildTree(X, y, hessians, indices, 0, cfg)

	if tree == nil {
		t.Fatal("expected a tree, got nil")
	}

	// Should be a leaf node (no children)
	if tree.Left != nil || tree.Right != nil {
		t.Error("expected leaf node due to MaxDepth=0")
	}

	// Value should be mean of all y (since hessians are all 1.0)
	expectedMean := 2.5
	if tree.Value != expectedMean {
		t.Errorf("leaf value = %v, want %v", tree.Value, expectedMean)
	}
}

func TestBuildTreeSingleSample(t *testing.T) {
	X := [][]float64{
		{1.0},
	}
	y := []float64{5.0}
	hessians := []float64{1.0}
	indices := []int{0}
	cfg := Config{
		MaxDepth:       10,
		MinSamplesLeaf: 1,
	}

	tree := buildTree(X, y, hessians, indices, 0, cfg)

	if tree == nil {
		t.Fatal("expected a tree, got nil")
	}

	// Should be a leaf node
	if tree.Left != nil || tree.Right != nil {
		t.Error("expected leaf node for single sample")
	}

	if tree.Value != 5.0 {
		t.Errorf("leaf value = %v, want 5.0", tree.Value)
	}
}

func TestBuildLeafNodeNewtonRaphson(t *testing.T) {
	// With uniform hessians, leaf value = mean(gradients)
	t.Run("uniform hessians", func(t *testing.T) {
		grads := []float64{2.0, 4.0, 6.0}
		hess := []float64{1.0, 1.0, 1.0}
		leaf := buildLeafNode(grads, hess)
		// sum(grads)/sum(hess) = 12/3 = 4.0
		if math.Abs(leaf.Value-4.0) > 1e-10 {
			t.Errorf("leaf value = %v, want 4.0", leaf.Value)
		}
	})

	// With non-uniform hessians, high-hessian samples get more weight
	t.Run("non-uniform hessians", func(t *testing.T) {
		grads := []float64{1.0, 3.0}
		hess := []float64{0.1, 0.9}
		leaf := buildLeafNode(grads, hess)
		// sum(grads)/sum(hess) = 4.0/1.0 = 4.0
		if math.Abs(leaf.Value-4.0) > 1e-10 {
			t.Errorf("leaf value = %v, want 4.0", leaf.Value)
		}
	})

	// Simulating LogLoss: uncertain samples (p≈0.5) have higher hessian
	t.Run("logloss-like hessians", func(t *testing.T) {
		// Sample 0: confident (p=0.9), hessian = 0.9*0.1 = 0.09, gradient = 0.1
		// Sample 1: uncertain (p=0.5), hessian = 0.5*0.5 = 0.25, gradient = 0.5
		grads := []float64{0.1, 0.5}
		hess := []float64{0.09, 0.25}
		leaf := buildLeafNode(grads, hess)
		// sum(grads)/sum(hess) = 0.6/0.34 ≈ 1.7647
		expected := 0.6 / 0.34
		if math.Abs(leaf.Value-expected) > 1e-4 {
			t.Errorf("leaf value = %v, want %v", leaf.Value, expected)
		}
	})
}

func TestBuildTreeWithNonUniformHessians(t *testing.T) {
	// When hessians differ, leaf values should be sum(grad)/sum(hess), not mean(grad)
	X := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
	}
	grads := []float64{1.0, 1.0, 10.0, 10.0}
	hessians := []float64{0.5, 0.5, 0.25, 0.25}
	indices := []int{0, 1, 2, 3}
	cfg := Config{
		MaxDepth:       3,
		MinSamplesLeaf: 1,
	}

	tree := buildTree(X, grads, hessians, indices, 0, cfg)

	if tree == nil {
		t.Fatal("expected a tree, got nil")
	}
	if tree.Left == nil || tree.Right == nil {
		t.Fatal("expected internal node with children")
	}

	// Left group: grads=[1,1], hess=[0.5,0.5] → 2.0/1.0 = 2.0 (not mean=1.0)
	if math.Abs(tree.Left.Value-2.0) > 1e-10 {
		t.Errorf("left leaf value = %v, want 2.0", tree.Left.Value)
	}
	// Right group: grads=[10,10], hess=[0.25,0.25] → 20.0/0.5 = 40.0 (not mean=10.0)
	if math.Abs(tree.Right.Value-40.0) > 1e-10 {
		t.Errorf("right leaf value = %v, want 40.0", tree.Right.Value)
	}
}
