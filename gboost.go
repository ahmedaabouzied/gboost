package gboost

type GBM struct {
	Config       Config
	isFitted     bool
	trainedTrees any
}

func New(cfg Config) *GBM {
	return &GBM{
		Config:       cfg,
		isFitted:     false,
		trainedTrees: nil,
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
	return nil
}

func (g *GBM) Predict(X [][]float64) []float64 {
	return []float64{}
}

func (g *GBM) PredictSingle(x []float64) float64 {
	return 0.0
}
