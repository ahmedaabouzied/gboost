package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/ahmedaabouzied/gboost"
)

func main() {
	fmt.Println("=== GBoost Demo ===\n")

	// Generate synthetic data: y = 2*x1 + 3*x2 + noise
	n := 100
	X, y := generateData(n)

	// Split into train/test (80/20)
	splitIdx := int(float64(n) * 0.8)
	XTrain, yTrain := X[:splitIdx], y[:splitIdx]
	XTest, yTest := X[splitIdx:], y[splitIdx:]

	fmt.Printf("Training samples: %d\n", len(XTrain))
	fmt.Printf("Test samples: %d\n\n", len(XTest))

	// Configure and train
	cfg := gboost.Config{
		NEstimators:    50,
		LearningRate:   0.1,
		MaxDepth:       4,
		MinSamplesLeaf: 1,
		Loss:           "mse",
	}

	fmt.Printf("Config: %d trees, lr=%.2f, maxDepth=%d\n\n", cfg.NEstimators, cfg.LearningRate, cfg.MaxDepth)

	gbm := gboost.New(cfg)
	err := gbm.Fit(XTrain, yTrain)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Training complete!\n")

	// Evaluate on training data
	trainPreds := gbm.Predict(XTrain)
	trainMSE := mse(yTrain, trainPreds)
	trainRMSE := math.Sqrt(trainMSE)

	// Evaluate on test data
	testPreds := gbm.Predict(XTest)
	testMSE := mse(yTest, testPreds)
	testRMSE := math.Sqrt(testMSE)

	fmt.Println("=== Results ===")
	fmt.Printf("Train RMSE: %.4f\n", trainRMSE)
	fmt.Printf("Test RMSE:  %.4f\n\n", testRMSE)

	// Show some predictions
	fmt.Println("=== Sample Predictions (Test Set) ===")
	fmt.Println("  X1     X2    | Actual | Predicted | Error")
	fmt.Println("--------------------------------------------")
	for i := 0; i < min(10, len(XTest)); i++ {
		actual := yTest[i]
		pred := testPreds[i]
		err := actual - pred
		fmt.Printf("%6.2f %6.2f | %6.2f | %9.2f | %+.2f\n",
			XTest[i][0], XTest[i][1], actual, pred, err)
	}
}

// generateData creates synthetic data: y = 2*x1 + 3*x2 + noise
func generateData(n int) ([][]float64, []float64) {
	X := make([][]float64, n)
	y := make([]float64, n)

	for i := 0; i < n; i++ {
		x1 := rand.Float64() * 10
		x2 := rand.Float64() * 10
		noise := rand.NormFloat64() * 0.5

		X[i] = []float64{x1, x2}
		y[i] = 2*x1 + 3*x2 + noise
	}
	return X, y
}

func mse(actual, predicted []float64) float64 {
	var sum float64
	for i := range actual {
		diff := actual[i] - predicted[i]
		sum += diff * diff
	}
	return sum / float64(len(actual))
}
