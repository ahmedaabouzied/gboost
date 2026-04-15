package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/ahmedaabouzied/gboost"
)

func main() {
	// Load the binary Iris dataset (versicolor=0, virginica=1).
	ds, err := gboost.LoadCSV("data/iris_binary.csv", -1, true)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d samples, %d features\n", len(ds.X), len(ds.X[0]))

	// Split: 80% train, 20% test, fixed seed for reproducibility.
	XTrain, XTest, yTrain, yTest, err := ds.Split(0.2, 42)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Train: %d samples, Test: %d samples\n\n", len(XTrain), len(XTest))

	// Save train/test splits to CSV so Python can use the same data.
	saveSplit("data/iris_train.csv", ds.Header[:len(ds.Header)-1], XTrain, yTrain)
	saveSplit("data/iris_test.csv", ds.Header[:len(ds.Header)-1], XTest, yTest)
	fmt.Println("Saved data/iris_train.csv and data/iris_test.csv")

	// Train with specific hyperparameters (match these in Python).
	cfg := gboost.DefaultConfig()
	cfg.NEstimators = 100
	cfg.LearningRate = 0.1
	cfg.MaxDepth = 3
	cfg.MinSamplesLeaf = 1
	cfg.SubsampleRatio = 1.0
	cfg.Loss = "logloss"

	fmt.Println("\n--- Hyperparameters ---")
	fmt.Printf("NEstimators:    %d\n", cfg.NEstimators)
	fmt.Printf("LearningRate:   %.2f\n", cfg.LearningRate)
	fmt.Printf("MaxDepth:       %d\n", cfg.MaxDepth)
	fmt.Printf("MinSamplesLeaf: %d\n", cfg.MinSamplesLeaf)
	fmt.Printf("SubsampleRatio: %.2f\n", cfg.SubsampleRatio)

	model := gboost.New(cfg)
	if err := model.Fit(XTrain, yTrain); err != nil {
		log.Fatal(err)
	}

	// Evaluate on test set.
	fmt.Println("\n--- Test Set Predictions ---")
	fmt.Printf("%-6s %-8s %-10s %-10s\n", "Index", "Actual", "Predicted", "Prob(1)")

	correct := 0
	for i, x := range XTest {
		prob := model.PredictProba(x)
		predicted := 0.0
		if prob >= 0.5 {
			predicted = 1.0
		}
		if predicted == yTest[i] {
			correct++
		}
		fmt.Printf("%-6d %-8.0f %-10.0f %-10.4f\n", i, yTest[i], predicted, prob)
	}

	accuracy := float64(correct) / float64(len(yTest)) * 100
	fmt.Printf("\n--- Results ---\n")
	fmt.Printf("Correct: %d / %d\n", correct, len(yTest))
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)

	// Also report train accuracy.
	trainCorrect := 0
	for i, x := range XTrain {
		prob := model.PredictProba(x)
		predicted := 0.0
		if prob >= 0.5 {
			predicted = 1.0
		}
		if predicted == yTrain[i] {
			trainCorrect++
		}
	}
	trainAccuracy := float64(trainCorrect) / float64(len(yTrain)) * 100
	fmt.Printf("Train Accuracy: %.2f%%\n", trainAccuracy)

	// Report log loss on test set.
	logloss := 0.0
	for i, x := range XTest {
		p := model.PredictProba(x)
		p = math.Max(1e-15, math.Min(1-1e-15, p)) // clip for numerical stability
		if yTest[i] == 1.0 {
			logloss -= math.Log(p)
		} else {
			logloss -= math.Log(1 - p)
		}
	}
	logloss /= float64(len(yTest))
	fmt.Printf("Test Log Loss: %.4f\n", logloss)

	// Report gain-based feature importance.
	featureNames := ds.Header[:len(ds.Header)-1]
	importance := model.FeatureImportance()
	fmt.Println("\n--- Gain-Based Feature Importance ---")
	for i, name := range featureNames {
		fmt.Printf("  %-15s %.4f\n", name, importance[i])
	}

	// --- SHAP section ---
	// Per-device explanations (log-odds space) + global SHAP importance.
	shapValues, err := model.ShapValues(XTest)
	if err != nil {
		log.Fatalf("ShapValues: %v", err)
	}
	baseValue := model.BaseValue()

	fmt.Println("\n--- SHAP Explanations (first 5 test samples, log-odds) ---")
	fmt.Printf("Base value (E[f(X)] in log-odds): %.4f\n\n", baseValue)
	for i := 0; i < min(5, len(XTest)); i++ {
		phi := shapValues[i]
		sum := baseValue
		for _, v := range phi {
			sum += v
		}
		fmt.Printf("Sample %d  label=%.0f  raw=%.4f  prob=%.4f\n",
			i, yTest[i], sum, sigmoid(sum))
		// Sort features by |phi| descending.
		type kv struct {
			name string
			val  float64
		}
		kvs := make([]kv, len(phi))
		for j, v := range phi {
			kvs[j] = kv{featureNames[j], v}
		}
		sort.Slice(kvs, func(a, b int) bool {
			return math.Abs(kvs[a].val) > math.Abs(kvs[b].val)
		})
		for _, k := range kvs {
			sign := "+"
			if k.val < 0 {
				sign = "-"
			}
			fmt.Printf("    %s %-15s %.4f\n", sign, k.name, math.Abs(k.val))
		}
	}

	shapImp, err := model.ShapImportance(XTest)
	if err != nil {
		log.Fatalf("ShapImportance: %v", err)
	}
	fmt.Println("\n--- SHAP Feature Importance (mean |phi| on test set, log-odds) ---")
	for i, name := range featureNames {
		fmt.Printf("  %-15s %.4f\n", name, shapImp[i])
	}

	// Save SHAP values + metadata to JSON so the parity test can consume them.
	out := map[string]any{
		"base_value":     baseValue,
		"shap_values":    shapValues,
		"feature_names":  featureNames,
		"shap_importance": shapImp,
	}
	f, err := os.Create("data/iris_go_shap.json")
	if err != nil {
		log.Fatalf("open iris_go_shap.json: %v", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		log.Fatalf("encode iris_go_shap.json: %v", err)
	}
	fmt.Println("\nSaved data/iris_go_shap.json")
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func saveSplit(path string, featureHeaders []string, X [][]float64, y []float64) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Write header.
	header := append(featureHeaders, "label")
	w.Write(header)

	// Write rows.
	for i, row := range X {
		record := make([]string, len(row)+1)
		for j, v := range row {
			record[j] = strconv.FormatFloat(v, 'f', -1, 64)
		}
		record[len(row)] = strconv.FormatFloat(y[i], 'f', 0, 64)
		w.Write(record)
	}
}
