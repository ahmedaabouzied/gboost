package gboost

type Loss interface {
	InitialPrediction(y []float64) float64
	NegativeGradient(y, pred []float64) []float64
}

// MSELoss: implements "Mean Squared Error" loss.
// It measures the average squared difference between predictions and actual values: (1/n) * Σ(y - pred)².
// It's useful for regression problems where the target is a continuous value (e.g., predicting house prices, temperatures, scores)
// One of its characteristics is that it's sensitive to outliers (large errors are squared, so they dominate).
// For outlier-robust regression, you'd use MAE or Huber loss instead.
type MSELoss struct{}

// InitialPrediction returns the mean of y — this minimizes MSE when you have no other information (the "best constant prediction")
func (l *MSELoss) InitialPrediction(y []float64) float64 {
	return mean(y)
}

// NegativeGradient returns (y - pred) (the residuals). Each tree in GBM fits these residuals, gradually correcting the model's errors
func (l *MSELoss) NegativeGradient(y, pred []float64) []float64 {
	return vsub(y, pred)
}
