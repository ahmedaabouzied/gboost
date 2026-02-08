package gboost

import "math"

type Loss interface {
	InitialPrediction(y []float64) float64
	NegativeGradient(y, pred []float64) []float64
	Hessian(y, pred []float64) []float64
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

// Hessian returns the "Hessian" for the MSE loss function.
// It's always 1.0 since first derivative of MSE is F - y.
// Therefore the second derivative is 1.
func (l *MSELoss) Hessian(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range res {
		res[i] = 1.0
	}
	return res
}

type LogLoss struct{}

func (l *LogLoss) InitialPrediction(y []float64) float64 {
	p := mean(y)
	p = max(0.001, min(0.999, p)) // clip to safe range
	logOdds := math.Log(p / (1 - p))
	return logOdds
}

func (l *LogLoss) NegativeGradient(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range y {
		res[i] = y[i] - sigmoid(pred[i])
	}
	return res
}

func (l *LogLoss) Hessian(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range y {
		p := sigmoid(pred[i])
		res[i] = p * (1 - p)
	}
	return res
}
