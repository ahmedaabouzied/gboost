package gboost

type Config struct {
	Seed           int64   // Seed for randomizing to reproduce the same models from the same data.
	NEstimators    int     // Number of trees
	LearningRate   float64 // shrinkage factor
	MaxDepth       int     // Maximum tree depth
	MinSamplesLeaf int     // Minimum samples per leaf
	SubsampleRatio float64 // Row sampling ratio
	Loss           string  // Loss function name
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
