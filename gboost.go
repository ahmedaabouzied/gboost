package gboost

import (
	"math"
	"math/rand"
)

// GBM is a gradient boosting machine model. Create one with [New], train it
// with [GBM.Fit], and make predictions with [GBM.Predict] or [GBM.PredictProba].
type GBM struct {
	Config            Config
	rnd               *rand.Rand
	isFitted          bool
	trees             []*Node
	initialPrediction float64
	loss              Loss

	featureImportance []float64
	numFeatures       int
}

// New creates an untrained GBM model with the given configuration.
// Call [GBM.Fit] to train the model on data.
func New(cfg Config) *GBM {
	return &GBM{
		Config:   cfg,
		isFitted: false,
	}
}

// Fit trains the model on the given feature matrix X and target values y.
// X is a slice of samples where each sample is a slice of feature values.
// For regression (Loss="mse"), y contains continuous target values.
// For classification (Loss="logloss"), y must contain only 0.0 and 1.0.
//
// Fit validates the configuration and input data, returning an error if
// either is invalid. Calling Fit on an already-trained model retrains from scratch.
func (g *GBM) Fit(X [][]float64, y []float64) error {
	if err := g.Config.validate(); err != nil {
		return err
	}

	switch {
	case len(X) < 1:
		return ErrEmptyDataset
	case len(X[0]) < 1:
		return ErrEmptyFeatures
	case len(X) != len(y):
		return ErrLengthMismatch
	case !hasSimilarLength(X):
		return ErrFeatureCountMismatch
	}

	// Reset state for re-fitting
	g.trees = nil
	g.rnd = rand.New(rand.NewSource(g.Config.Seed))

	// Set the number of features from the X set.
	g.numFeatures = len(X[0])

	//
	// 1. Create loss function based on cfg.Loss
	lossFunc := createLossFunction(g.Config)
	g.loss = lossFunc

	// 2. Get the basic initial prediction
	initialPrediction := lossFunc.InitialPrediction(y)
	g.initialPrediction = initialPrediction

	// 3. Initial predictions slice
	predictions := make([]float64, len(y))
	for i := range predictions {
		predictions[i] = initialPrediction
	}

	// 4. All indices
	allIndices := make([]int, len(y))
	for i := range allIndices {
		allIndices[i] = i
	}

	// Training ...
	for range g.Config.NEstimators {
		trainIndices := allIndices
		if g.Config.SubsampleRatio > 0 && g.Config.SubsampleRatio < 1.0 {
			trainIndices = g.sampleIndices(allIndices)
		}
		residuals := lossFunc.NegativeGradient(y, predictions)
		hessians := lossFunc.Hessian(y, predictions)
		tree := buildTree(X, residuals, hessians, trainIndices, 0, g.Config)
		for j := range predictions {
			predictions[j] += g.Config.LearningRate * tree.predict(X[j])
		}

		g.trees = append(g.trees, tree)
	}
	// Calculate the featureImportance
	g.calculateFeatureImportance()

	g.isFitted = true
	return nil
}

// Predict returns raw predictions for each sample in X.
// For regression, these are the predicted target values.
// For classification, these are log-odds; use [GBM.PredictProbaAll] for probabilities.
func (g *GBM) Predict(X [][]float64) []float64 {
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = g.PredictSingle(x)
	}
	return results
}

// PredictSingle returns the raw prediction for a single sample.
// For regression, this is the predicted value. For classification, this is the log-odds.
func (g *GBM) PredictSingle(x []float64) float64 {
	prediction := g.initialPrediction
	for _, tree := range g.trees {
		prediction += g.Config.LearningRate * tree.predict(x)
	}
	return prediction
}

// PredictProba returns P(y=1) for a single sample by applying the sigmoid
// function to the raw log-odds prediction. Only meaningful for classification (Loss="logloss").
func (g *GBM) PredictProba(x []float64) float64 {
	return sigmoid(g.PredictSingle(x))
}

// PredictProbaAll returns P(y=1) for each sample in X.
// Only meaningful for classification (Loss="logloss").
func (g *GBM) PredictProbaAll(X [][]float64) []float64 {
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = g.PredictProba(x)
	}
	return results
}

// FeatureImportance returns the gain-based feature importance scores, normalized
// to sum to 1.0. Each value represents the fraction of total variance reduction
// contributed by that feature across all splits in all trees.
// Returns an empty slice if the model has not been trained.
func (g *GBM) FeatureImportance() []float64 {
	if !g.isFitted {
		return []float64{}
	}
	return g.featureImportance
}

// ShapValues returns per-sample, per-feature SHAP contributions computed with
// TreeSHAP (Lundberg 2018). The returned matrix has shape len(X) × numFeatures:
// result[i][j] is feature j's contribution to the raw prediction for X[i].
//
// SHAP values are additive in the model's raw output space. For every sample,
//
//	sum(result[i]) + [GBM.BaseValue]() == [GBM.PredictSingle](X[i])
//
// For classification (Loss="logloss"), contributions are in log-odds; there is
// no additive decomposition in probability space.
//
// Returns [ErrModelNotFitted] if the model has not been trained, or
// [ErrFeatureCountMismatch] if any row of X does not have numFeatures columns.
func (g *GBM) ShapValues(X [][]float64) ([][]float64, error) {
	result := make([][]float64, len(X))

	for i, x := range X {
		contrib, err := g.ShapValuesSingle(x)
		if err != nil {
			return nil, err
		}
		result[i] = contrib
	}

	return result, nil
}

// BaseValue returns the expected model output over the training distribution,
// above which SHAP contributions are measured. It equals the initial prediction
// plus the sum of each tree's cover-weighted expected value scaled by the
// learning rate.
//
// For every sample x:
//
//	sum([GBM.ShapValuesSingle](x)) + BaseValue() == [GBM.PredictSingle](x)
//
// For classification, this is in log-odds space. Returns 0 if the model has
// not been trained.
func (g *GBM) BaseValue() float64 {
	if !g.isFitted {
		return 0
	}

	v := g.initialPrediction
	for _, tree := range g.trees {
		v += g.Config.LearningRate * tree.expectedValue()
	}

	return v
}

// ShapValuesSingle returns per-feature SHAP contributions for a single sample x
// using TreeSHAP. result[j] is feature j's contribution to the raw prediction.
//
// SHAP values satisfy additivity:
//
//	sum(result) + [GBM.BaseValue]() == [GBM.PredictSingle](x)
//
// For classification (Loss="logloss"), the contributions are in log-odds space
// and sum to the raw log-odds, not to the probability.
//
// Returns [ErrModelNotFitted] if the model has not been trained, or
// [ErrFeatureCountMismatch] if len(x) does not match the number of features
// the model was trained on.
func (g *GBM) ShapValuesSingle(x []float64) ([]float64, error) {
	if !g.isFitted {
		return nil, ErrModelNotFitted
	}

	if len(x) != g.numFeatures {
		return nil, ErrFeatureCountMismatch
	}

	phi := make([]float64, g.numFeatures)
	phiTmp := make([]float64, g.numFeatures)

	for _, tree := range g.trees {
		for i := range phiTmp {
			phiTmp[i] = 0
		}

		treeShap(tree, x, phiTmp, newPath(g.Config.MaxDepth))

		for i := range phi {
			phi[i] += g.Config.LearningRate * phiTmp[i]
		}
	}

	return phi, nil
}

// ShapImportance returns SHAP-based global feature importance computed over X.
// importance[j] is the mean absolute SHAP value of feature j across the rows
// of X:
//
//	importance[j] = mean over i of |ShapValues(X)[i][j]|
//
// Unlike the gain-based [GBM.FeatureImportance], the result is in the model's
// output units (log-odds for classification) and is not normalized to sum to
// 1. Computing importance over different subsets of X (e.g. positive vs.
// negative class samples) surfaces different feature rankings — useful for
// slice-level explanations that gain-based importance cannot produce.
//
// Returns [ErrModelNotFitted] if the model has not been trained,
// [ErrEmptyDataset] if X is empty, or [ErrFeatureCountMismatch] if any row of
// X does not have numFeatures columns.
func (g *GBM) ShapImportance(X [][]float64) ([]float64, error) {
	if !g.isFitted {
		return nil, ErrModelNotFitted
	}

	if len(X) == 0 {
		return nil, ErrEmptyDataset
	}

	shap, err := g.ShapValues(X)
	if err != nil {
		return nil, err
	}

	importance := make([]float64, g.numFeatures)
	for _, row := range shap {
		for j := range g.numFeatures {
			importance[j] += math.Abs(row[j])
		}
	}

	for j := range g.numFeatures {
		importance[j] /= float64(len(X))
	}
	return importance, nil
}

func (g *GBM) sampleIndices(indices []int) []int {
	sampleRatio := g.Config.SubsampleRatio

	n := len(indices)
	sampleSize := int(float64(n) * sampleRatio)
	shuffled := make([]int, n)
	copy(shuffled, indices)
	g.rnd.Shuffle(n, func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled[0:sampleSize]
}

func (g *GBM) calculateFeatureImportance() {
	res := make([]float64, g.numFeatures)
	for _, tree := range g.trees {
		tree.collectGains(res)
	}
	// Normalize the gains
	sumOfGains := sum(res)
	if sumOfGains != 0 {
		for i := range res {
			res[i] = res[i] / sumOfGains
		}
	}
	g.featureImportance = res
}

func createLossFunction(cfg Config) Loss {
	switch cfg.Loss {
	case "mse":
		return &MSELoss{}
	case "logloss":
		return &LogLoss{}
	default:
		panic("unreachable: config.validate() should reject invalid loss")
	}
}
