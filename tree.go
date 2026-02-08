package gboost

// Node is the basic tree node.
// A leaf node has Left == Right == nil.
type Node struct {
	FeatureIndex int     // Which feature column to check
	Threshold    float64 // The split value ("if feature < threshold, go left")
	Left         *Node   // Pointer to the left child. Will be nil for leaf nodes.
	Right        *Node   // Pointer to the right child. Will be nil for leaf nodes.
	Value        float64 // The predicted value for a leaf node. The output basically for GBM.

	Gain float64 // Recording how much gain the split at this node contributed, then we can get the important features.
}

type Split struct {
	FeatureIndex int     // Feature column to split on
	Threshold    float64 // The split value
	LeftIndices  []int   // Row indices where X[i][FeatureIndex] < Threshold
	RightIndices []int   // Row indices where X[i][FeatureIndex] >= Threshold
	Gain         float64 // The variance reduction
}

func buildLeafNode(y, hessians []float64) *Node {
	return &Node{
		FeatureIndex: -1, // Not relevant in this case
		Threshold:    0,  // Not relevant in this case
		Value:        sum(y) / sum(hessians),
	}
}

// buildTree recursively builds a decision tree picking up the best split it can.
func buildTree(X [][]float64, y []float64, hessians []float64, indices []int, depth int, cfg Config) *Node {
	if depth >= cfg.MaxDepth || len(indices) < 2 {
		return buildLeafNode(
			extractRows(y, indices),
			extractRows(hessians, indices),
		)
	}

	split := findBestSplit(X, y, indices, cfg.MinSamplesLeaf)
	if split == nil {
		// Return leaf node
		return buildLeafNode(
			extractRows(y, indices),
			extractRows(hessians, indices),
		)
	}

	node := &Node{
		FeatureIndex: split.FeatureIndex,
		Threshold:    split.Threshold,
		Gain:         split.Gain,
	}
	node.Left = buildTree(X, y, hessians, split.LeftIndices, depth+1, cfg)
	node.Right = buildTree(X, y, hessians, split.RightIndices, depth+1, cfg)
	return node
}

func findBestSplit(X [][]float64, y []float64, indices []int, minSamplesLeaf int) *Split {
	var bestSplit *Split
	var bestGain float64 = 0.0

	numFeatures := len(X[0])

	parentVariance := variance(extractRows(y, indices))

	for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
		featureValues := extractFeatureValues(X, indices, featureIndex)
		candidateThresholds := uniq(sort(featureValues))

		for _, threshold := range candidateThresholds {
			leftIndices, rightIndices := partition(X, indices, featureIndex, threshold)
			if len(leftIndices) < minSamplesLeaf || len(rightIndices) < minSamplesLeaf {
				continue
			}
			split := &Split{
				FeatureIndex: featureIndex,
				Threshold:    threshold,
				LeftIndices:  leftIndices,
				RightIndices: rightIndices,
			}
			gain := split.ComputeGain(y, indices, parentVariance)
			if gain > bestGain {
				bestGain = gain
				bestSplit = split
			}
		}
	}
	return bestSplit
}

func (s *Split) ComputeGain(y []float64, indices []int, parentVariance float64) float64 {
	n := len(indices)
	nLeft := len(s.LeftIndices)
	nRight := len(s.RightIndices)

	leftVariance := variance(extractRows(y, s.LeftIndices))
	rightVariance := variance(extractRows(y, s.RightIndices))

	weightedChildVariance := (float64(nLeft)/float64(n))*leftVariance +
		(float64(nRight)/float64(n))*rightVariance

	gain := parentVariance - weightedChildVariance
	s.Gain = gain

	return gain
}

// Tree predict single
func (n *Node) predict(x []float64) float64 {
	if n.Left == nil && n.Right == nil {
		// Leaf node. Return value
		return n.Value
	}

	if x[n.FeatureIndex] < n.Threshold {
		return n.Left.predict(x)
	} else {
		return n.Right.predict(x)
	}

}

func (n *Node) collectGains(index []float64) {
	if n.Left == nil && n.Right == nil {
		// Leaf node. Return value
		return
	}
	index[n.FeatureIndex] += n.Gain
	n.Left.collectGains(index)
	n.Right.collectGains(index)
}

func extractRows[T any](y []T, indices []int) []T {
	res := make([]T, len(indices))
	for j, i := range indices {
		res[j] = y[i]
	}
	return res
}

func extractFeatureValues[T any](X [][]T, indices []int, featureIndex int) []T {
	res := make([]T, len(indices))
	for i, idx := range indices {
		res[i] = X[idx][featureIndex]
	}
	return res
}

func partition(X [][]float64, indices []int, featureIndex int, threshold float64) (left, right []int) {
	leftIndices := []int{}
	rightIndices := []int{}

	for _, idx := range indices {
		if X[idx][featureIndex] < threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}
	return leftIndices, rightIndices
}
