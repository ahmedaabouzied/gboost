package gboost

import "math"

// Loss defines the interface for a loss function used by [GBM] during training.
// It provides the initial constant prediction, first-order gradients, and
// second-order Hessians needed for Newton-Raphson optimized tree building.
type Loss interface {
	// InitialPrediction returns the optimal constant prediction for the target values y.
	// For MSE this is the mean; for LogLoss this is the log-odds of the positive class.
	InitialPrediction(y []float64) float64

	// NegativeGradient returns the negative gradient of the loss with respect to predictions.
	// Each tree in the ensemble fits these pseudo-residuals.
	NegativeGradient(y, pred []float64) []float64

	// Hessian returns the second derivative of the loss with respect to predictions.
	// Used for Newton-Raphson leaf value optimization: leaf = sum(gradient) / sum(hessian).
	Hessian(y, pred []float64) []float64
}

// MSELoss implements mean squared error for regression: L(y, F) = (1/2)(y - F)².
// The gradient is simply the residual (y - F) and the Hessian is constant (1.0).
type MSELoss struct{}

// InitialPrediction returns the mean of y, the optimal constant prediction under MSE.
func (l *MSELoss) InitialPrediction(y []float64) float64 {
	return mean(y)
}

// NegativeGradient returns the residuals (y - pred).
func (l *MSELoss) NegativeGradient(y, pred []float64) []float64 {
	return vsub(y, pred)
}

// Hessian returns 1.0 for every sample (the second derivative of MSE is constant).
func (l *MSELoss) Hessian(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range res {
		res[i] = 1.0
	}
	return res
}

// LogLoss implements binary cross-entropy for classification:
// L(y, F) = -[y*log(p) + (1-y)*log(1-p)] where p = sigmoid(F).
// The Hessian is p*(1-p), which enables Newton-Raphson leaf optimization
// for faster convergence and better probability calibration.
type LogLoss struct{}

// InitialPrediction returns the log-odds of the positive class: log(p / (1-p)).
func (l *LogLoss) InitialPrediction(y []float64) float64 {
	p := mean(y)
	p = max(0.001, min(0.999, p)) // clip to safe range
	logOdds := math.Log(p / (1 - p))
	return logOdds
}

// NegativeGradient returns y - sigmoid(pred) for each sample.
func (l *LogLoss) NegativeGradient(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range y {
		res[i] = y[i] - sigmoid(pred[i])
	}
	return res
}

// Hessian returns p*(1-p) for each sample, where p = sigmoid(pred).
func (l *LogLoss) Hessian(y, pred []float64) []float64 {
	res := make([]float64, len(y))
	for i := range y {
		p := sigmoid(pred[i])
		res[i] = p * (1 - p)
	}
	return res
}
