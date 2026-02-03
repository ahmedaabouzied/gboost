package gboost

import "errors"

var (
	ErrEmptyDataset         = errors.New("empty dataset")
	ErrEmptyFeatures        = errors.New("empty features")
	ErrLengthMismatch       = errors.New("mismatch length of input matrix")
	ErrFeatureCountMismatch = errors.New("feature count mismatch")
)
