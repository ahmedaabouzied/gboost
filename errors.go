package gboost

import "errors"

var (
	ErrEmptyDataset         = errors.New("empty dataset")
	ErrEmptyFeatures        = errors.New("empty features")
	ErrLengthMismatch       = errors.New("mismatch length of input matrix")
	ErrFeatureCountMismatch = errors.New("feature count mismatch")
	ErrModelNotFitted       = errors.New("model not fitted")

	ErrInvalidNEstimators    = errors.New("NEstimators must be >= 0")
	ErrInvalidLearningRate   = errors.New("LearningRate must be > 0")
	ErrInvalidMaxDepth       = errors.New("MaxDepth must be >= 1")
	ErrInvalidMinSamplesLeaf = errors.New("MinSamplesLeaf must be >= 1")
	ErrInvalidSubsampleRatio = errors.New("SubsampleRatio must be in (0, 1]")
	ErrInvalidLoss           = errors.New("Loss must be \"mse\" or \"logloss\"")
)
