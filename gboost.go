package gboost

import "math/rand"

type GBM struct {
	Config            Config
	isFitted          bool
	trees             []*Node
	initialPrediction float64 // The base score from the loss func
	loss              Loss
}

func New(cfg Config) *GBM {
	return &GBM{
		Config:   cfg,
		isFitted: false,
	}
}

func (g *GBM) Fit(X [][]float64, y []float64) error {
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

	g.isFitted = true
	return nil
}

func (g *GBM) Predict(X [][]float64) []float64 {
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = g.PredictSingle(x)
	}
	return results
}

func (g *GBM) PredictSingle(x []float64) float64 {
	prediction := g.initialPrediction
	for _, tree := range g.trees {
		prediction += g.Config.LearningRate * tree.predict(x)
	}
	return prediction
}

// PredictProba returns probability for a single sample (applies sigmoid to log-odds)
func (g *GBM) PredictProba(x []float64) float64 {
	return sigmoid(g.PredictSingle(x))
}

// PredictProbaAll returns probabilities for multiple samples
func (g *GBM) PredictProbaAll(X [][]float64) []float64 {
	results := make([]float64, len(X))
	for i, x := range X {
		results[i] = g.PredictProba(x)
	}
	return results
}

func (g *GBM) sampleIndices(indices []int) []int {
	sampleRatio := g.Config.SubsampleRatio

	n := len(indices)
	sampleSize := int(float64(n) * sampleRatio)
	shuffled := make([]int, n)
	copy(shuffled, indices)
	rand.Shuffle(n, func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled[0:sampleSize]
}

func createLossFunction(cfg Config) Loss {
	switch cfg.Loss {
	case "mse":
		return &MSELoss{}
	case "logloss":
		return &LogLoss{}
	default:
		panic("unsupported loss function")
	}
}
