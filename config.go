package gboost

// Config controls the hyperparameters for training a [GBM] model.
type Config struct {
	// Seed for the random number generator used in subsampling.
	// A fixed seed produces deterministic, reproducible models.
	Seed int64

	// NEstimators is the number of boosting rounds (trees) to build.
	// More trees reduce training error but increase training time and risk of overfitting.
	NEstimators int

	// LearningRate (shrinkage) scales each tree's contribution.
	// Smaller values require more trees but generally produce better generalization.
	LearningRate float64

	// MaxDepth is the maximum depth of each decision tree.
	// Deeper trees capture more complex interactions but are more prone to overfitting.
	MaxDepth int

	// MinSamplesLeaf is the minimum number of samples required in a leaf node.
	// Higher values prevent the model from learning overly specific patterns.
	MinSamplesLeaf int

	// SubsampleRatio is the fraction of training samples used to build each tree.
	// Values less than 1.0 enable stochastic gradient boosting, which can reduce overfitting.
	// Must be in the range (0, 1].
	SubsampleRatio float64

	// Loss is the loss function name: "mse" for regression or "logloss" for binary classification.
	Loss string
}

func (c Config) validate() error {
	switch {
	case c.NEstimators < 0:
		return ErrInvalidNEstimators
	case c.LearningRate <= 0:
		return ErrInvalidLearningRate
	case c.MaxDepth < 1:
		return ErrInvalidMaxDepth
	case c.MinSamplesLeaf < 1:
		return ErrInvalidMinSamplesLeaf
	case c.SubsampleRatio <= 0 || c.SubsampleRatio > 1.0:
		return ErrInvalidSubsampleRatio
	case c.Loss != "mse" && c.Loss != "logloss":
		return ErrInvalidLoss
	}
	return nil
}

// DefaultConfig returns a Config with sensible defaults for regression:
// 100 trees, learning rate 0.1, max depth 6, no subsampling, MSE loss.
func DefaultConfig() Config {
	return Config{
		Seed:           0,
		NEstimators:    100,
		LearningRate:   0.1,
		MaxDepth:       6,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}
}
