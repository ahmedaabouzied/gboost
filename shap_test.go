package gboost

import (
	"math"
	"testing"
)

const shapEpsilon = 1e-12

func nearlyEqual(a, b float64) bool {
	return math.Abs(a-b) < shapEpsilon
}

func TestNewPath(t *testing.T) {
	p := newPath(3)

	if len(p) != 1 {
		t.Errorf("len(p) = %d, want 1", len(p))
	}
	if cap(p) != 4 {
		t.Errorf("cap(p) = %d, want 4 (maxDepth+1)", cap(p))
	}

	s := p[0]
	if s.featureIndex != -1 {
		t.Errorf("sentinel.featureIndex = %d, want -1", s.featureIndex)
	}
	if s.zFraction != 1 || s.oFraction != 1 || s.pweight != 1 {
		t.Errorf("sentinel = (z=%v, o=%v, w=%v), want (1, 1, 1)",
			s.zFraction, s.oFraction, s.pweight)
	}
}

// TestExtendHotBranchFromSentinel verifies the canonical first-extend case:
// sentinel path + hot branch with cover ratio 0.6 should yield weights [0.3, 0.5].
func TestExtendHotBranchFromSentinel(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)

	if len(p) != 2 {
		t.Fatalf("len(p) = %d, want 2", len(p))
	}
	if !nearlyEqual(p[0].pweight, 0.3) {
		t.Errorf("p[0].pweight = %v, want 0.3", p[0].pweight)
	}
	if !nearlyEqual(p[1].pweight, 0.5) {
		t.Errorf("p[1].pweight = %v, want 0.5", p[1].pweight)
	}
	if p[1].featureIndex != 0 {
		t.Errorf("p[1].featureIndex = %d, want 0", p[1].featureIndex)
	}
	if !nearlyEqual(p[1].zFraction, 0.6) || !nearlyEqual(p[1].oFraction, 1.0) {
		t.Errorf("p[1] fractions = (z=%v, o=%v), want (0.6, 1.0)",
			p[1].zFraction, p[1].oFraction)
	}
}

// TestExtendColdBranchFromSentinel verifies that a cold branch (po=0) zeros out
// the new element's accumulator: weights become [0.2, 0].
func TestExtendColdBranchFromSentinel(t *testing.T) {
	p := newPath(5)
	p.extend(0.4, 0.0, 0)

	if len(p) != 2 {
		t.Fatalf("len(p) = %d, want 2", len(p))
	}
	if !nearlyEqual(p[0].pweight, 0.2) {
		t.Errorf("p[0].pweight = %v, want 0.2", p[0].pweight)
	}
	if !nearlyEqual(p[1].pweight, 0) {
		t.Errorf("p[1].pweight = %v, want 0", p[1].pweight)
	}
}

// TestExtendTwoHotBranches covers the loop body for paths longer than one
// element. Starting from the sentinel, extend(0.6, 1, 0) then extend(0.5, 1, 1)
// should produce weights [1/10, 11/60, 1/3].
func TestExtendTwoHotBranches(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.extend(0.5, 1.0, 1)

	if len(p) != 3 {
		t.Fatalf("len(p) = %d, want 3", len(p))
	}
	want := []float64{0.1, 11.0 / 60.0, 1.0 / 3.0}
	for i, w := range want {
		if !nearlyEqual(p[i].pweight, w) {
			t.Errorf("p[%d].pweight = %v, want %v", i, p[i].pweight, w)
		}
	}
}

func pathsEqual(a, b path) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].featureIndex != b[i].featureIndex {
			return false
		}
		if !nearlyEqual(a[i].zFraction, b[i].zFraction) {
			return false
		}
		if !nearlyEqual(a[i].oFraction, b[i].oFraction) {
			return false
		}
		if !nearlyEqual(a[i].pweight, b[i].pweight) {
			return false
		}
	}
	return true
}

// TestUnwindHotRestoresSentinel: extending hot then unwinding the new element
// should restore the original sentinel-only path.
func TestUnwindHotRestoresSentinel(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.unwind(1)

	want := newPath(5)
	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindColdRestoresSentinel: extending cold then unwinding exercises the
// else (oFraction == 0) branch of unwind.
func TestUnwindColdRestoresSentinel(t *testing.T) {
	p := newPath(5)
	p.extend(0.4, 0.0, 0)
	p.unwind(1)

	want := newPath(5)
	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindLastOfTwoExtends: two hot extends, then unwind the last element.
// This exercises the n-rolling-state for two loop iterations and catches any
// off-by-one in the n update (bug 1 from review).
// Expected result: identical to just the first extend from a fresh sentinel.
func TestUnwindLastOfTwoExtends(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.extend(0.5, 1.0, 1)
	p.unwind(2)

	want := newPath(5)
	want.extend(0.6, 1.0, 0)

	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindInteriorOfTwoExtends: two hot extends, then unwind the interior
// element (the first-extended feature). This exercises the shift loop and
// catches bugs in the post-unwind element reshuffle (bug 2 from review).
// Expected: identical to just the second extend from a fresh sentinel.
func TestUnwindInteriorOfTwoExtends(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.extend(0.5, 1.0, 1)
	p.unwind(1)

	want := newPath(5)
	want.extend(0.5, 1.0, 1)

	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindMiddleOfThreeExtends: three hot extends, unwind the middle element.
// Shift loop needs to move more than one element down; a copy-paste bug that
// only copies one slot would fail here even if the two-extend test passes.
// Expected: identical to extend(first) followed by extend(third).
func TestUnwindMiddleOfThreeExtends(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.extend(0.5, 1.0, 1)
	p.extend(0.4, 1.0, 2)
	p.unwind(2) // removes the middle element (feature 1)

	want := newPath(5)
	want.extend(0.6, 1.0, 0)
	want.extend(0.4, 1.0, 2)

	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindColdInterior: mixed cold + hot extends, then unwind the cold
// interior element. Exercises the else branch of the weight loop combined
// with the shift loop.
// Expected: identical to just the hot extend from a fresh sentinel.
func TestUnwindColdInterior(t *testing.T) {
	p := newPath(5)
	p.extend(0.4, 0.0, 0) // cold
	p.extend(0.5, 1.0, 1) // hot
	p.unwind(1)           // remove the cold feature-0 element

	want := newPath(5)
	want.extend(0.5, 1.0, 1)

	if !pathsEqual(p, want) {
		t.Errorf("path mismatch:\n  got:  %+v\n  want: %+v", p, want)
	}
}

// TestUnwindLengthShrinks: sanity check that unwind shrinks the path by
// exactly one and preserves the remaining element count.
func TestUnwindLengthShrinks(t *testing.T) {
	p := newPath(5)
	p.extend(0.6, 1.0, 0)
	p.extend(0.5, 1.0, 1)
	p.extend(0.4, 1.0, 2)

	beforeLen := len(p)
	p.unwind(2)
	if len(p) != beforeLen-1 {
		t.Errorf("len(p) after unwind = %d, want %d", len(p), beforeLen-1)
	}
}

// ==========================================================================
// End-to-end SHAP tests (steps 3c / 3d / 4)
//
// Many of the ground-truth values below come from the step-3d worked example:
// a 2-feature tree with leaves 10, 20, 50 and covers 4, 3, 3. SHAP values were
// computed by brute-force subset enumeration (f(S) for each S, then the
// 1/2-weighted sum) and documented in the conversation.
// ==========================================================================

const shapFloatTol = 1e-9

func shapAlmostEqual(a, b float64) bool {
	return math.Abs(a-b) < shapFloatTol
}

// buildByHandTree reproduces the worked example tree:
//
//	         [x0 < 0.5]  cover=10
//	         /        \
//	    leaf=10      [x1 < 0.5]  cover=6
//	    cover=4      /        \
//	             leaf=20    leaf=50
//	             cover=3    cover=3
func buildByHandTree() *Node {
	return &Node{
		FeatureIndex: 0,
		Threshold:    0.5,
		NSamples:     10,
		Left:         &Node{FeatureIndex: -1, Value: 10, NSamples: 4},
		Right: &Node{
			FeatureIndex: 1,
			Threshold:    0.5,
			NSamples:     6,
			Left:         &Node{FeatureIndex: -1, Value: 20, NSamples: 3},
			Right:        &Node{FeatureIndex: -1, Value: 50, NSamples: 3},
		},
	}
}

// buildDuplicateFeatureTree forces the same feature to be split twice on a
// single root-to-leaf path, exercising the duplicate-feature branch of the
// TreeSHAP recursion.
//
//	         [x0 < 0.3]  cover=10
//	         /        \
//	    leaf=5       [x0 < 0.7]  cover=7
//	    cover=3      /        \
//	             leaf=10    leaf=20
//	             cover=4    cover=3
func buildDuplicateFeatureTree() *Node {
	return &Node{
		FeatureIndex: 0,
		Threshold:    0.3,
		NSamples:     10,
		Left:         &Node{FeatureIndex: -1, Value: 5, NSamples: 3},
		Right: &Node{
			FeatureIndex: 0,
			Threshold:    0.7,
			NSamples:     7,
			Left:         &Node{FeatureIndex: -1, Value: 10, NSamples: 4},
			Right:        &Node{FeatureIndex: -1, Value: 20, NSamples: 3},
		},
	}
}

// buildSingleLeafTree represents a degenerate tree with no splits. Useful
// for boundary testing.
func buildSingleLeafTree(v float64, n int) *Node {
	return &Node{FeatureIndex: -1, Value: v, NSamples: n}
}

// manualGBM wraps pre-built trees into a fitted *GBM for SHAP tests that
// don't want to invoke Fit.
func manualGBM(trees []*Node, numFeatures int, initialPrediction, learningRate float64) *GBM {
	return &GBM{
		Config: Config{
			LearningRate: learningRate,
			MaxDepth:     10,
			NEstimators:  len(trees),
			Loss:         "mse",
		},
		trees:             trees,
		initialPrediction: initialPrediction,
		numFeatures:       numFeatures,
		isFitted:          true,
		loss:              &MSELoss{},
	}
}

// treeExpectedValue computes E[tree(X)] over the training distribution, using
// cover ratios. Sum over leaves of (leaf.Value * leaf.NSamples) / root.NSamples.
func treeExpectedValue(n *Node) float64 {
	if n.Left == nil && n.Right == nil {
		return n.Value
	}
	rL := float64(n.Left.NSamples) / float64(n.NSamples)
	rR := float64(n.Right.NSamples) / float64(n.NSamples)
	return rL*treeExpectedValue(n.Left) + rR*treeExpectedValue(n.Right)
}

// ensembleExpectedValue is the base value SHAP values sum to, above which each
// feature's contribution is measured. For a gradient-boosted ensemble:
// initialPrediction + sum over trees of LearningRate * E[tree].
func ensembleExpectedValue(g *GBM) float64 {
	v := g.initialPrediction
	for _, tree := range g.trees {
		v += g.Config.LearningRate * treeExpectedValue(tree)
	}
	return v
}

// --- Ground-truth SHAP values on the worked 2-feature example ----------------

func TestShapByHandExample_HotHot(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{1, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(phi[0], 13) || !shapAlmostEqual(phi[1], 12) {
		t.Errorf("phi = %v, want [13, 12]", phi)
	}
}

func TestShapByHandExample_ColdA(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{0, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(phi[0], -19.5) || !shapAlmostEqual(phi[1], 4.5) {
		t.Errorf("phi = %v, want [-19.5, 4.5]", phi)
	}
}

func TestShapByHandExample_HotCold(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{1, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(phi[0], 7) || !shapAlmostEqual(phi[1], -12) {
		t.Errorf("phi = %v, want [7, -12]", phi)
	}
}

func TestShapByHandExample_ColdCold(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{0, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(phi[0], -10.5) || !shapAlmostEqual(phi[1], -4.5) {
		t.Errorf("phi = %v, want [-10.5, -4.5]", phi)
	}
}

// --- Learning rate scaling is applied ---------------------------------------

func TestShapScalesByLearningRate(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 0.1)
	phi, err := g.ShapValuesSingle([]float64{1, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Same tree as hot-hot case, but LR=0.1 → phi should be 1/10th.
	if !shapAlmostEqual(phi[0], 1.3) || !shapAlmostEqual(phi[1], 1.2) {
		t.Errorf("phi = %v, want [1.3, 1.2]", phi)
	}
}

// --- Duplicate feature on the root-to-leaf path -----------------------------

func TestShapDuplicateFeatureOnPath(t *testing.T) {
	g := manualGBM([]*Node{buildDuplicateFeatureTree()}, 1, 0.0, 1.0)

	cases := []struct {
		x0       float64
		wantPred float64
	}{
		{0.1, 5},  // path: left leaf
		{0.5, 10}, // path: right then left — uses duplicate-feature handling
		{0.9, 20}, // path: right then right — uses duplicate-feature handling
	}
	for _, c := range cases {
		phi, err := g.ShapValuesSingle([]float64{c.x0})
		if err != nil {
			t.Fatalf("x0=%v: %v", c.x0, err)
		}
		// With a single feature, phi[0] = prediction − ensemble expected value.
		wantPhi := c.wantPred - ensembleExpectedValue(g)
		if !shapAlmostEqual(phi[0], wantPhi) {
			t.Errorf("x0=%v: phi[0]=%v, want %v", c.x0, phi[0], wantPhi)
		}
	}
}

// --- Additivity: sum(phi) + BaseValue() == PredictSingle --------------------

func assertAdditivity(t *testing.T, g *GBM, x []float64) {
	t.Helper()
	phi, err := g.ShapValuesSingle(x)
	if err != nil {
		t.Fatalf("x=%v: %v", x, err)
	}
	sum := g.BaseValue()
	for _, v := range phi {
		sum += v
	}
	pred := g.PredictSingle(x)
	if !shapAlmostEqual(sum, pred) {
		t.Errorf("additivity broken at x=%v: sum(phi)+BaseValue()=%v, PredictSingle=%v",
			x, sum, pred)
	}
}

func TestShapAdditivityByHandExample(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	for _, x := range [][]float64{{1, 1}, {0, 1}, {1, 0}, {0, 0}} {
		assertAdditivity(t, g, x)
	}
}

func TestShapAdditivityOnTrainedRegressor(t *testing.T) {
	X, y := syntheticRegressionData(50, 3, 42)

	cfg := DefaultConfig()
	cfg.NEstimators = 15
	cfg.MaxDepth = 3
	cfg.LearningRate = 0.1
	cfg.Seed = 7

	g := New(cfg)
	if err := g.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	for _, x := range X {
		assertAdditivity(t, g, x)
	}
}

func TestShapAdditivityOnTrainedClassifier(t *testing.T) {
	X, y := syntheticBinaryData(60, 3, 42)

	cfg := DefaultConfig()
	cfg.Loss = "logloss"
	cfg.NEstimators = 15
	cfg.MaxDepth = 3
	cfg.LearningRate = 0.1
	cfg.Seed = 7

	g := New(cfg)
	if err := g.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Additivity holds in log-odds (raw) space, not probability space.
	for _, x := range X {
		assertAdditivity(t, g, x)
	}
}

// --- Error paths ------------------------------------------------------------

func TestShapValuesSingleUnfitted(t *testing.T) {
	g := New(DefaultConfig())
	_, err := g.ShapValuesSingle([]float64{0, 0})
	if err != ErrModelNotFitted {
		t.Errorf("err = %v, want ErrModelNotFitted", err)
	}
}

func TestShapValuesSingleFeatureMismatch(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	_, err := g.ShapValuesSingle([]float64{0}) // only 1 feature, model has 2
	if err != ErrFeatureCountMismatch {
		t.Errorf("err = %v, want ErrFeatureCountMismatch", err)
	}
}

func TestShapValuesPropagatesError(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	X := [][]float64{{1, 1}, {0}} // second row is wrong width
	_, err := g.ShapValues(X)
	if err != ErrFeatureCountMismatch {
		t.Errorf("err = %v, want ErrFeatureCountMismatch", err)
	}
}

// --- Batch API --------------------------------------------------------------

func TestShapValuesBatchMatchesSingle(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	X := [][]float64{{1, 1}, {0, 1}, {1, 0}, {0, 0}}

	batch, err := g.ShapValues(X)
	if err != nil {
		t.Fatalf("ShapValues: %v", err)
	}
	if len(batch) != len(X) {
		t.Fatalf("len(batch) = %d, want %d", len(batch), len(X))
	}
	for i, x := range X {
		single, err := g.ShapValuesSingle(x)
		if err != nil {
			t.Fatalf("single[%d]: %v", i, err)
		}
		for j := range single {
			if !shapAlmostEqual(batch[i][j], single[j]) {
				t.Errorf("batch[%d][%d] = %v, single = %v", i, j, batch[i][j], single[j])
			}
		}
	}
}

// --- BaseValue --------------------------------------------------------------

func TestBaseValueUnfitted(t *testing.T) {
	g := New(DefaultConfig())
	if got := g.BaseValue(); got != 0 {
		t.Errorf("BaseValue on unfitted = %v, want 0", got)
	}
}

// TestBaseValueMatchesEnsembleExpectation verifies that BaseValue() equals
// initialPrediction + sum over trees of LearningRate * E[tree]. This is the
// value SHAP contributions are measured *above*, and it is what makes
// sum(phi) + BaseValue() == PredictSingle(x) hold universally.
func TestBaseValueMatchesEnsembleExpectation(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	want := ensembleExpectedValue(g)
	if !shapAlmostEqual(g.BaseValue(), want) {
		t.Errorf("BaseValue = %v, want %v (initialPrediction + LR*E[tree])",
			g.BaseValue(), want)
	}
}

// --- Edge cases -------------------------------------------------------------

// TestShapSingleLeafTree: a tree with only a root leaf uses no features;
// every contribution must be zero.
func TestShapSingleLeafTree(t *testing.T) {
	g := manualGBM([]*Node{buildSingleLeafTree(7, 10)}, 3, 0.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, v := range phi {
		if !shapAlmostEqual(v, 0) {
			t.Errorf("phi[%d] = %v, want 0 (single-leaf tree uses no features)", i, v)
		}
	}
}

// TestShapFeatureNeverSplit: if a feature appears in X but is never split on
// by any tree, its SHAP value must be zero for every sample.
func TestShapFeatureNeverSplit(t *testing.T) {
	// Tree only splits on feature 0; feature 1 exists in X but is unused.
	tree := &Node{
		FeatureIndex: 0,
		Threshold:    0.5,
		NSamples:     10,
		Left:         &Node{FeatureIndex: -1, Value: 1, NSamples: 5},
		Right:        &Node{FeatureIndex: -1, Value: 9, NSamples: 5},
	}
	g := manualGBM([]*Node{tree}, 2, 0.0, 1.0)

	for _, x := range [][]float64{{0, 100}, {1, -50}, {0.3, 0}, {0.7, 7}} {
		phi, err := g.ShapValuesSingle(x)
		if err != nil {
			t.Fatalf("x=%v: %v", x, err)
		}
		if !shapAlmostEqual(phi[1], 0) {
			t.Errorf("x=%v: phi[1] = %v, want 0 (feature 1 never split)", x, phi[1])
		}
	}
}

// TestShapLearningRateZero: with LR=0, trees contribute nothing to predictions
// or SHAP values. All phi must be zero.
func TestShapLearningRateZero(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 5.0, 0.0)
	phi, err := g.ShapValuesSingle([]float64{1, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, v := range phi {
		if !shapAlmostEqual(v, 0) {
			t.Errorf("phi[%d] = %v, want 0 (LR=0)", i, v)
		}
	}
}

// TestShapEmptyEnsemble: a GBM with zero trees should return all-zero phi.
func TestShapEmptyEnsemble(t *testing.T) {
	g := manualGBM([]*Node{}, 2, 3.0, 1.0)
	phi, err := g.ShapValuesSingle([]float64{1, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(phi) != 2 {
		t.Fatalf("len(phi) = %d, want 2", len(phi))
	}
	for i, v := range phi {
		if !shapAlmostEqual(v, 0) {
			t.Errorf("phi[%d] = %v, want 0 (no trees)", i, v)
		}
	}
}

// TestShapDeterministic: same seed + same data should yield identical SHAP.
func TestShapDeterministic(t *testing.T) {
	X, y := syntheticRegressionData(30, 3, 123)
	cfg := DefaultConfig()
	cfg.NEstimators = 10
	cfg.MaxDepth = 3
	cfg.Seed = 42

	g1 := New(cfg)
	if err := g1.Fit(X, y); err != nil {
		t.Fatalf("g1.Fit: %v", err)
	}
	g2 := New(cfg)
	if err := g2.Fit(X, y); err != nil {
		t.Fatalf("g2.Fit: %v", err)
	}
	for _, x := range X {
		p1, _ := g1.ShapValuesSingle(x)
		p2, _ := g2.ShapValuesSingle(x)
		for j := range p1 {
			if !shapAlmostEqual(p1[j], p2[j]) {
				t.Errorf("x=%v: phi[%d] differs: %v vs %v", x, j, p1[j], p2[j])
			}
		}
	}
}

// TestShapMultiTreeEnsemble: with multiple trees, each tree's contribution is
// scaled by LR and summed. Check against by-hand computation: two copies of
// the worked tree with LR=0.5 → phi should equal single-tree LR=1 exactly.
func TestShapMultiTreeEnsemble(t *testing.T) {
	tree := buildByHandTree()
	g := manualGBM([]*Node{tree, tree}, 2, 0.0, 0.5)
	phi, err := g.ShapValuesSingle([]float64{1, 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 2 trees * 0.5 * [13, 12] = [13, 12]
	if !shapAlmostEqual(phi[0], 13) || !shapAlmostEqual(phi[1], 12) {
		t.Errorf("phi = %v, want [13, 12]", phi)
	}
}

// --- Synthetic data helpers for the trained-model tests ---------------------

func syntheticRegressionData(n, f int, seed int64) ([][]float64, []float64) {
	r := newDetermRand(seed)
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, f)
		for j := 0; j < f; j++ {
			X[i][j] = r.Float64()
		}
		// Nonlinear target with noise.
		y[i] = X[i][0]*2 + X[i][1]*X[i][1] + 0.1*r.Float64()
	}
	return X, y
}

func syntheticBinaryData(n, f int, seed int64) ([][]float64, []float64) {
	r := newDetermRand(seed)
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, f)
		for j := 0; j < f; j++ {
			X[i][j] = r.Float64()
		}
		if X[i][0]+X[i][1] > 1 {
			y[i] = 1
		}
	}
	return X, y
}

// minimal deterministic RNG wrapper to avoid pulling math/rand into the test
// surface uncleanly; we only need repeatable floats.
type detRand struct{ state uint64 }

func newDetermRand(seed int64) *detRand { return &detRand{state: uint64(seed) + 1} }
func (r *detRand) Float64() float64 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return float64(r.state%1000000) / 1000000.0
}

// ==========================================================================
// ShapImportance tests (step 5)
//
// Ground-truth numbers come from the by-hand example's SHAP values:
//   x=(1,1) -> phi=[13, 12]       x=(0,1) -> phi=[-19.5, 4.5]
//   x=(1,0) -> phi=[7, -12]       x=(0,0) -> phi=[-10.5, -4.5]
// ==========================================================================

// TestShapImportance_AllFourByHand: importance over all four points equals
// column-wise mean(|phi|) = [(13+19.5+7+10.5)/4, (12+4.5+12+4.5)/4] = [12.5, 8.25].
func TestShapImportance_AllFourByHand(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	X := [][]float64{{1, 1}, {0, 1}, {1, 0}, {0, 0}}

	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(imp[0], 12.5) || !shapAlmostEqual(imp[1], 8.25) {
		t.Errorf("imp = %v, want [12.5, 8.25]", imp)
	}
}

// TestShapImportance_SingleSample: with len(X)=1, importance equals |phi|
// of that single sample.
func TestShapImportance_SingleSample(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	imp, err := g.ShapImportance([][]float64{{1, 1}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(imp[0], 13) || !shapAlmostEqual(imp[1], 12) {
		t.Errorf("imp = %v, want [13, 12]", imp)
	}
}

// TestShapImportance_AbsNotSignedMean: if feature j's phi is positive on one
// sample and negative on another, importance must NOT cancel to zero — we take
// mean of absolute values.
func TestShapImportance_AbsNotSignedMean(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	// x=(1,1) -> phi[1]=12;  x=(1,0) -> phi[1]=-12.  Signed mean = 0, |mean| = 12.
	X := [][]float64{{1, 1}, {1, 0}}
	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if shapAlmostEqual(imp[1], 0) {
		t.Errorf("imp[1] = 0 suggests signed-mean (cancellation); want mean(|phi|)=12")
	}
	if !shapAlmostEqual(imp[1], 12) {
		t.Errorf("imp[1] = %v, want 12", imp[1])
	}
}

// TestShapImportance_MatchesManualBatchMean: ShapImportance must equal what
// you'd get by calling ShapValues and averaging |.| manually.
func TestShapImportance_MatchesManualBatchMean(t *testing.T) {
	X, y := syntheticRegressionData(40, 3, 11)
	cfg := DefaultConfig()
	cfg.NEstimators = 10
	cfg.MaxDepth = 3
	cfg.Seed = 77
	g := New(cfg)
	if err := g.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("ShapImportance: %v", err)
	}

	shap, err := g.ShapValues(X)
	if err != nil {
		t.Fatalf("ShapValues: %v", err)
	}
	want := make([]float64, len(shap[0]))
	for _, row := range shap {
		for j, v := range row {
			want[j] += math.Abs(v)
		}
	}
	for j := range want {
		want[j] /= float64(len(X))
	}

	for j := range imp {
		if !shapAlmostEqual(imp[j], want[j]) {
			t.Errorf("imp[%d] = %v, want %v", j, imp[j], want[j])
		}
	}
}

// TestShapImportance_NonNegative: by construction, all values are mean of
// absolute values, so every entry must be >= 0.
func TestShapImportance_NonNegative(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	X := [][]float64{{1, 1}, {0, 1}, {1, 0}, {0, 0}}
	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for j, v := range imp {
		if v < 0 {
			t.Errorf("imp[%d] = %v < 0", j, v)
		}
	}
}

// TestShapImportance_Length: return slice has length == numFeatures.
func TestShapImportance_Length(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	imp, err := g.ShapImportance([][]float64{{1, 1}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(imp) != 2 {
		t.Errorf("len(imp) = %d, want 2", len(imp))
	}
}

// TestShapImportance_FeatureNeverSplit: a feature never used by any tree has
// importance exactly 0, no matter what values appear for it in X.
func TestShapImportance_FeatureNeverSplit(t *testing.T) {
	// Tree only splits on feature 0; feature 1 is present in X but unused.
	tree := &Node{
		FeatureIndex: 0,
		Threshold:    0.5,
		NSamples:     10,
		Left:         &Node{FeatureIndex: -1, Value: 1, NSamples: 5},
		Right:        &Node{FeatureIndex: -1, Value: 9, NSamples: 5},
	}
	g := manualGBM([]*Node{tree}, 2, 0.0, 1.0)
	X := [][]float64{{0, 100}, {1, -50}, {0.3, 0}, {0.7, 7}}
	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !shapAlmostEqual(imp[1], 0) {
		t.Errorf("imp[1] = %v, want 0 (feature 1 never split)", imp[1])
	}
	if imp[0] == 0 {
		t.Errorf("imp[0] = 0; feature 0 is used and should have non-zero importance")
	}
}

// TestShapImportance_SingleLeafTree: a tree with no splits uses no features;
// every importance value must be 0.
func TestShapImportance_SingleLeafTree(t *testing.T) {
	g := manualGBM([]*Node{buildSingleLeafTree(7, 10)}, 3, 0.0, 1.0)
	X := [][]float64{{1, 2, 3}, {4, 5, 6}}
	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for j, v := range imp {
		if !shapAlmostEqual(v, 0) {
			t.Errorf("imp[%d] = %v, want 0", j, v)
		}
	}
}

// TestShapImportance_SliceGivesDifferentRankings: the SHAP-importance workflow
// advantage — computing importance over different subsets of X surfaces
// different feature rankings. Gain-based importance cannot do this.
func TestShapImportance_SliceGivesDifferentRankings(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)

	// Hot-A group (feature 0 is large): x0 = 1
	hot := [][]float64{{1, 1}, {1, 0}}
	impHot, _ := g.ShapImportance(hot)

	// Cold-A group (feature 0 is small): x0 = 0
	cold := [][]float64{{0, 1}, {0, 0}}
	impCold, _ := g.ShapImportance(cold)

	// Hot: feature 1 dominates (|phi[1]| avg = 12 vs |phi[0]| avg = 10).
	if !(impHot[1] > impHot[0]) {
		t.Errorf("hot group: feature 1 should dominate, got imp=%v", impHot)
	}
	// Cold: feature 0 dominates (|phi[0]| avg = 15 vs |phi[1]| avg = 4.5).
	if !(impCold[0] > impCold[1]) {
		t.Errorf("cold group: feature 0 should dominate, got imp=%v", impCold)
	}
}

// TestShapImportance_Deterministic: same seed, same X => same importance.
func TestShapImportance_Deterministic(t *testing.T) {
	X, y := syntheticRegressionData(30, 3, 42)
	cfg := DefaultConfig()
	cfg.NEstimators = 10
	cfg.MaxDepth = 3
	cfg.Seed = 9

	g1 := New(cfg)
	if err := g1.Fit(X, y); err != nil {
		t.Fatalf("g1.Fit: %v", err)
	}
	g2 := New(cfg)
	if err := g2.Fit(X, y); err != nil {
		t.Fatalf("g2.Fit: %v", err)
	}

	i1, _ := g1.ShapImportance(X)
	i2, _ := g2.ShapImportance(X)
	for j := range i1 {
		if !shapAlmostEqual(i1[j], i2[j]) {
			t.Errorf("imp[%d] differs: %v vs %v", j, i1[j], i2[j])
		}
	}
}

// TestShapImportance_MonotonicityWithDominantFeature: when y clearly depends
// on feature 0, its SHAP importance should be the largest.
func TestShapImportance_MonotonicityWithDominantFeature(t *testing.T) {
	r := newDetermRand(123)
	n, f := 80, 3
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, f)
		for j := 0; j < f; j++ {
			X[i][j] = r.Float64()
		}
		// y driven almost entirely by feature 0; 1 and 2 are noise.
		y[i] = 10*X[i][0] + 0.01*X[i][1] + 0.01*X[i][2]
	}

	cfg := DefaultConfig()
	cfg.NEstimators = 30
	cfg.MaxDepth = 3
	cfg.Seed = 1
	g := New(cfg)
	if err := g.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("ShapImportance: %v", err)
	}
	if !(imp[0] > imp[1] && imp[0] > imp[2]) {
		t.Errorf("feature 0 should dominate, got imp=%v", imp)
	}
}

// --- Error paths -----------------------------------------------------------

func TestShapImportance_Unfitted(t *testing.T) {
	g := New(DefaultConfig())
	_, err := g.ShapImportance([][]float64{{0, 0}})
	if err != ErrModelNotFitted {
		t.Errorf("err = %v, want ErrModelNotFitted", err)
	}
}

func TestShapImportance_EmptyX(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	_, err := g.ShapImportance([][]float64{})
	if err != ErrEmptyDataset {
		t.Errorf("err = %v, want ErrEmptyDataset", err)
	}
}

func TestShapImportance_FeatureMismatchPropagates(t *testing.T) {
	g := manualGBM([]*Node{buildByHandTree()}, 2, 0.0, 1.0)
	X := [][]float64{{1, 1}, {0}} // second row wrong width
	_, err := g.ShapImportance(X)
	if err != ErrFeatureCountMismatch {
		t.Errorf("err = %v, want ErrFeatureCountMismatch", err)
	}
}

// TestShapImportance_LogLoss: works with classifier; values are in log-odds.
func TestShapImportance_LogLoss(t *testing.T) {
	X, y := syntheticBinaryData(60, 3, 99)
	cfg := DefaultConfig()
	cfg.Loss = "logloss"
	cfg.NEstimators = 15
	cfg.MaxDepth = 3
	cfg.Seed = 3
	g := New(cfg)
	if err := g.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	imp, err := g.ShapImportance(X)
	if err != nil {
		t.Fatalf("ShapImportance: %v", err)
	}
	if len(imp) != 3 {
		t.Fatalf("len(imp) = %d, want 3", len(imp))
	}
	for j, v := range imp {
		if v < 0 {
			t.Errorf("imp[%d] = %v < 0", j, v)
		}
	}
}
