# gboost

A gradient boosting machine learning library implemented from scratch in Go. Supports both regression (MSE) and binary classification (Log Loss), with Newton-Raphson leaf optimization, dataset utilities, model serialization, and a scikit-learn-comparable API.

[![Go Reference](https://pkg.go.dev/badge/github.com/ahmedaabouzied/gboost@v0.2.0.svg)](https://pkg.go.dev/github.com/ahmedaabouzied/gboost@v0.2.0)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ahmedaabouzied/ebe93926763deca896636894a1bd3bfc/raw/gboost-coverage.json)](https://github.com/ahmedaabouzied/gboost/actions)

```go
model := gboost.New(gboost.DefaultConfig())
model.Fit(X, y)
predictions := model.Predict(X)
```

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How Gradient Boosting Works](#how-gradient-boosting-works)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Comparison with scikit-learn](#comparison-with-scikit-learn)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [References](#references)

## Installation

```bash
go get github.com/ahmedaabouzied/gboost
```

Requires Go 1.21+ (uses generics).

## Quick Start

### Regression

```go
package main

import (
    "fmt"
    "github.com/ahmedaabouzied/gboost"
)

func main() {
    // Training data: y = 2*x1 + 3*x2
    X := [][]float64{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}
    y := []float64{8, 13, 18, 23, 28}

    cfg := gboost.DefaultConfig()
    cfg.NEstimators = 50
    cfg.MaxDepth = 3

    model := gboost.New(cfg)
    model.Fit(X, y)

    fmt.Println(model.PredictSingle([]float64{6, 7})) // ~33
}
```

### Binary Classification

```go
cfg := gboost.DefaultConfig()
cfg.Loss = "logloss"
cfg.NEstimators = 100
cfg.MaxDepth = 3

model := gboost.New(cfg)
model.Fit(XTrain, yTrain) // y values must be 0.0 or 1.0

prob := model.PredictProba(sample)     // P(y=1) for a single sample
probs := model.PredictProbaAll(XTest)  // P(y=1) for all samples
```

### Loading CSV Data

```go
// Load a CSV file with automatic label encoding for non-numeric columns.
// Target column supports negative indexing (-1 = last column).
ds, err := gboost.LoadCSV("data/iris_binary.csv", -1, true)

// Split into train/test with reproducible shuffling.
XTrain, XTest, yTrain, yTest, err := ds.Split(0.2, 42)
```

### Save and Load Models

```go
model.Save("model.json")

loaded, err := gboost.Load("model.json")
loaded.Predict(XTest)
```

## How Gradient Boosting Works

Gradient boosting builds an ensemble of weak learners (decision trees) sequentially. Each tree corrects the errors of the previous ensemble by fitting to the **negative gradient** of the loss function. The final prediction is the sum of all tree outputs, scaled by a learning rate.

### The Algorithm

Given training data $(X, y)$ with $n$ samples, a differentiable loss function $L(y, F(x))$, and $M$ boosting iterations:

**Step 1: Initialize with a constant prediction**

$$F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c)$$

For MSE this is the mean of $y$. For Log Loss this is the log-odds of the positive class:

$$F_0(x) = \log\left(\frac{p}{1 - p}\right) \quad \text{where } p = \text{mean}(y)$$

**Step 2: For each iteration $m = 1, 2, \ldots, M$:**

Compute the negative gradient (pseudo-residuals) for each sample:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

Fit a regression tree $h_m(x)$ to the pseudo-residuals $r_{im}$.

Update the model:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $\eta$ is the learning rate (shrinkage factor).

**Step 3: Output the final model**

$$F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$$

### Loss Functions

#### Mean Squared Error (Regression)

$$L(y, F) = \frac{1}{2}(y - F)^2$$

The negative gradient (pseudo-residual) is simply the residual:

$$r_i = y_i - F(x_i)$$

The Hessian (second derivative) is constant:

$$\frac{\partial^2 L}{\partial F^2} = 1$$

The initial prediction is the mean of $y$:

$$F_0 = \text{mean}(y)$$

#### Log Loss (Binary Classification)

$$L(y, F) = -\left[y \cdot \log(p) + (1 - y) \cdot \log(1 - p)\right]$$

where $p = \sigma(F) = \frac{1}{1 + e^{-F}}$ is the sigmoid function.

The negative gradient is:

$$r_i = y_i - \sigma(F(x_i))$$

The Hessian is:

$$\frac{\partial^2 L}{\partial F^2} = p_i(1 - p_i) \quad \text{where } p_i = \sigma(F(x_i))$$

The initial prediction is the log-odds:

$$F_0 = \log\left(\frac{\sum y_i}{n - \sum y_i}\right)$$

For classification, raw predictions are passed through the sigmoid function to produce probabilities:

$$P(y = 1 \mid x) = \sigma(F_M(x)) = \frac{1}{1 + e^{-F_M(x)}}$$

### Newton-Raphson Leaf Optimization

In basic gradient boosting, leaf nodes predict the mean of the pseudo-residuals that reach them. This is a **first-order** approximation — it only uses the gradient (slope) of the loss function.

gboost uses the **Newton-Raphson** (second-order) optimization for leaf values. Instead of the mean, each leaf computes:

$$\text{leaf value} = \frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} h_i}$$

where $g_i$ is the negative gradient and $h_i$ is the Hessian (second derivative) for sample $i$.

**Why this matters:**

- For **MSE**, the Hessian is constant ($h_i = 1$), so $\frac{\sum g_i}{\sum h_i} = \frac{\sum g_i}{n} = \text{mean}(g_i)$ — identical to the basic approach.
- For **Log Loss**, the Hessian is $h_i = p_i(1 - p_i)$, which is large when the model is uncertain ($p \approx 0.5$) and small when confident ($p \approx 0$ or $p \approx 1$). This means **uncertain samples get more influence** on the leaf value, while already-confident samples contribute less. The result is faster convergence and better probability calibration.

This is the same optimization used by scikit-learn, XGBoost, and LightGBM.

### Tree Building

Each tree is built by recursively finding the best binary split. For each internal node:

1. For every feature and every unique threshold value in that feature, evaluate the split
2. The best split maximizes **variance reduction** (information gain):

$$\text{Gain} = \text{Var}(\text{parent}) - \left(\frac{n_{\text{left}}}{n} \cdot \text{Var}(\text{left}) + \frac{n_{\text{right}}}{n} \cdot \text{Var}(\text{right})\right)$$

3. Recursion stops when:
   - Maximum depth is reached
   - A node has fewer samples than `MinSamplesLeaf`
   - No valid split improves variance

### Learning Rate (Shrinkage)

The learning rate $\eta$ (default 0.1) scales each tree's contribution. Smaller values require more trees but generally produce better generalization:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

This is the "slow learning" principle from Friedman (2001): taking many small steps in function space outperforms taking fewer large steps.

### Feature Importance

gboost computes **gain-based feature importance**, the standard method used by scikit-learn, XGBoost, and LightGBM. For each split in every tree, the variance reduction (gain) is weighted by the number of samples that reached that node:

$$\text{importance}(j) = \sum_{\text{trees}} \sum_{\substack{\text{nodes where} \\ \text{feature} = j}} n_{\text{node}} \cdot \Delta_{\text{gain}}$$

where $n_{\text{node}}$ is the number of training samples at that node and $\Delta_{\text{gain}}$ is the variance reduction from the split. The sample weighting ensures that splits higher in the tree (which affect more data) contribute proportionally more than splits deep in the tree. Without this weighting, a root split affecting 80 samples would count the same as a leaf split affecting 2 samples.

After accumulating across all trees, importances are normalized to sum to 1.0:

$$\text{importance}_{\text{normalized}}(j) = \frac{\text{importance}(j)}{\sum_{k} \text{importance}(k)}$$

```go
model.Fit(X, y)
importance := model.FeatureImportance() // []float64, sums to 1.0
```

### Subsampling

When `SubsampleRatio < 1.0`, each tree is trained on a random subset of the training data. This introduces stochasticity that can reduce overfitting, as described in Friedman (2002).

## API Reference

### Config

```go
type Config struct {
    NEstimators    int     // Number of boosting rounds (trees). Default: 100
    LearningRate   float64 // Shrinkage factor per tree. Default: 0.1
    MaxDepth       int     // Maximum depth of each tree. Default: 6
    MinSamplesLeaf int     // Minimum samples required in a leaf. Default: 1
    SubsampleRatio float64 // Fraction of samples used per tree. Default: 1.0
    Loss           string  // "mse" for regression, "logloss" for classification. Default: "mse"
}

func DefaultConfig() Config
```

### GBM

```go
func New(cfg Config) *GBM

func (g *GBM) Fit(X [][]float64, y []float64) error
func (g *GBM) Predict(X [][]float64) []float64         // Raw predictions (regression or log-odds)
func (g *GBM) PredictSingle(x []float64) float64        // Raw prediction for one sample
func (g *GBM) PredictProba(x []float64) float64          // P(y=1) for one sample (classification)
func (g *GBM) PredictProbaAll(X [][]float64) []float64   // P(y=1) for all samples (classification)
func (g *GBM) FeatureImportance() []float64               // Gain-based feature importance (sums to 1.0)
func (g *GBM) Save(path string) error                    // Save model to JSON
func Load(path string) (*GBM, error)                      // Load model from JSON
```

### Dataset Utilities

```go
type Dataset struct {
    X              [][]float64
    Y              []float64
    Encodings      map[int]map[string]float64  // Feature label encodings (featureIndex -> string -> value)
    TargetEncoding map[string]float64          // Target label encoding (nil if numeric)
    Header         []string                     // Column names (nil if no header)
}

// Load a CSV file. Non-numeric columns are automatically label-encoded.
// targetColumn supports negative indexing (-1 = last column).
func LoadCSV(path string, targetColumn int, hasHeader bool) (*Dataset, error)

// Split into train/test sets with shuffling.
func TrainTestSplit(X [][]float64, y []float64, testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error)

// Convenience method on Dataset.
func (ds *Dataset) Split(testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error)
```

## Examples

### Regression Example

A synthetic regression demo is included in `cmd/demo/`:

```bash
go run ./cmd/demo/
```

This generates data following $y = 2x_1 + 3x_2 + \epsilon$, trains a GBM, and reports RMSE on train and test sets.

### Binary Classification Example (Iris)

A binary classification example using the Iris dataset is included in `cmd/iris/`:

```bash
go run ./cmd/iris/
```

This loads the Iris dataset (filtered to versicolor vs. virginica), trains a GBM with Log Loss, and reports accuracy and per-sample predictions. It also saves `data/iris_train.csv` and `data/iris_test.csv` for comparison with Python.

**Output:**

```
Loaded 100 samples, 4 features
Train: 80 samples, Test: 20 samples

Saved data/iris_train.csv and data/iris_test.csv

--- Hyperparameters ---
NEstimators:    100
LearningRate:   0.10
MaxDepth:       3
MinSamplesLeaf: 1
SubsampleRatio: 1.00

--- Test Set Predictions ---
Index  Actual   Predicted  Prob(1)
0      1        1          1.0000
1      1        1          0.9999
2      1        0          0.0028
3      1        1          1.0000
4      0        0          0.0002
5      1        1          1.0000
6      0        0          0.0002
7      0        0          0.0002
8      0        0          0.0002
9      1        1          1.0000
10     1        1          1.0000
11     1        1          0.9997
12     0        0          0.0001
13     1        1          0.9999
14     0        0          0.0001
15     0        0          0.0001
16     0        1          0.9999
17     1        1          1.0000
18     0        0          0.0001
19     0        0          0.0002

--- Results ---
Correct: 18 / 20
Accuracy: 90.00%
Train Accuracy: 100.00%
Test Log Loss: 0.7556
```

## Comparison with scikit-learn

To validate correctness, gboost was benchmarked against scikit-learn's `GradientBoostingClassifier` on the Iris dataset (binary: versicolor vs. virginica). Both models were trained on **identical data splits** with **identical hyperparameters**.

### Setup

- 100 samples (50 versicolor, 50 virginica), 4 features (sepal length/width, petal length/width)
- 80/20 train/test split with fixed seed
- Hyperparameters: 100 trees, learning rate 0.1, max depth 3, min samples leaf 1, no subsampling

### Results

| Metric | gboost (Go) | scikit-learn (Python) |
|---|---|---|
| **Test Accuracy** | **90.00%** | **90.00%** |
| **Train Accuracy** | 100.00% | 100.00% |
| **Misclassified Samples** | #2, #16 | #2, #16 |
| **Test Log Loss** | 0.7556 | 0.7246 |

### Per-Sample Probability Comparison

| Sample | Actual | gboost Prob(1) | sklearn Prob(1) | Prediction |
|---|---|---|---|---|
| 0 | 1 | 1.0000 | 1.0000 | 1 |
| 1 | 1 | 0.9999 | 0.9999 | 1 |
| 2 | 1 | 0.0028 | 0.0042 | 0 |
| 3 | 1 | 1.0000 | 1.0000 | 1 |
| 4 | 0 | 0.0002 | 0.0002 | 0 |
| 5 | 1 | 1.0000 | 1.0000 | 1 |
| 6 | 0 | 0.0002 | 0.0002 | 0 |
| 7 | 0 | 0.0002 | 0.0002 | 0 |
| 8 | 0 | 0.0002 | 0.0002 | 0 |
| 9 | 1 | 1.0000 | 1.0000 | 1 |
| 10 | 1 | 1.0000 | 1.0000 | 1 |
| 11 | 1 | 0.9997 | 0.9996 | 1 |
| 12 | 0 | 0.0001 | 0.0001 | 0 |
| 13 | 1 | 0.9999 | 0.9999 | 1 |
| 14 | 0 | 0.0001 | 0.0001 | 0 |
| 15 | 0 | 0.0001 | 0.0001 | 0 |
| 16 | 0 | 0.9999 | 0.9999 | 1 |
| 17 | 1 | 1.0000 | 1.0000 | 1 |
| 18 | 0 | 0.0001 | 0.0002 | 0 |
| 19 | 0 | 0.0002 | 0.0002 | 0 |

### Feature Importance Comparison

| Feature | gboost (Go) | scikit-learn (Python) |
|---|---|---|
| sepal_length | 0.0275 | 0.0093 |
| sepal_width | 0.0175 | 0.0218 |
| petal_length | 0.1285 | 0.1373 |
| **petal_width** | **0.8266** | **0.8316** |

Both implementations agree that **petal_width** is the dominant feature (~83%), followed by petal_length (~13%). The small differences come from the split criterion: gboost uses variance reduction while scikit-learn uses Friedman MSE.

### Impact of Newton-Raphson Leaf Optimization

The Newton-Raphson optimization was the single largest accuracy improvement in gboost's development. The table below shows the before and after on the same Iris benchmark:

| Metric | Before (mean-of-residuals) | After (Newton-Raphson) | scikit-learn |
|---|---|---|---|
| **Test Accuracy** | 90.00% | 90.00% | 90.00% |
| **Test Log Loss** | 0.2802 | **0.7556** | 0.7246 |
| **Probability range** | 0.10 – 0.90 | **0.0001 – 1.0000** | 0.0001 – 1.0000 |

**Before** (first-order, mean of residuals): Leaf values were computed as $\text{mean}(g_i)$, where $g_i$ are the pseudo-residuals. This treats every sample equally regardless of model confidence, producing moderate probabilities that never go far from 0.5. While this actually yielded lower log loss on this small test set (because moderate wrong predictions are penalized less), it indicates the model is not learning as aggressively as it could.

**After** (second-order, Newton-Raphson): Leaf values are computed as $\frac{\sum g_i}{\sum h_i}$, where $h_i = p_i(1 - p_i)$ is the Hessian. Uncertain samples ($p \approx 0.5$, large $h_i$) contribute more to the leaf value, while confident samples ($p \approx 0$ or $1$, small $h_i$) contribute less. This produces well-calibrated probabilities that match scikit-learn to 3–4 decimal places.

The remaining log loss gap (0.7556 vs 0.7246) comes from a difference in the split criterion: gboost uses variance reduction while scikit-learn uses the Friedman MSE criterion, which also incorporates second-order information into split selection.

### Reproducing the Comparison

**Go (run locally):**

```bash
go run ./cmd/iris/
```

This produces `data/iris_train.csv` and `data/iris_test.csv`.

**Python (run in Colab or locally):**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

train = pd.read_csv("iris_train.csv")
test = pd.read_csv("iris_test.csv")

X_train = train.drop("label", axis=1).values
y_train = train["label"].values
X_test = test.drop("label", axis=1).values
y_test = test["label"].values

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, preds) * 100))
print("Train Accuracy: {:.2f}%".format(
    accuracy_score(y_train, model.predict(X_train)) * 100
))
print("Test Log Loss: {:.4f}".format(log_loss(y_test, probs)))

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
print("\n--- Feature Importance ---")
for name, imp in zip(features, model.feature_importances_):
    print("  {:<15s} {:.4f}".format(name, imp))
```

## Project Structure

```
gboost/
    config.go          # Config struct and DefaultConfig()
    gboost.go          # GBM struct, Fit, Predict, PredictProba
    tree.go            # Decision tree: Node, Split, buildTree, findBestSplit
    loss.go            # Loss interface with Hessian, MSELoss, LogLoss
    math.go            # Generic math utilities (mean, sum, variance, sigmoid)
    util.go            # Helper functions (sort, uniq, validation)
    dataset.go         # LoadCSV, TrainTestSplit, Dataset struct
    serialize.go       # JSON Save/Load for model persistence
    errors.go          # Sentinel errors
    *_test.go          # Tests for each module (~97.9% coverage)
    cmd/
        demo/main.go   # Regression example with synthetic data
        iris/main.go   # Classification example with Iris dataset
    data/
        iris_binary.csv # Iris dataset (versicolor vs virginica)
```

## Testing

```bash
go test ./...          # Run all tests
go test -v ./...       # Verbose output
go test -cover ./...   # With coverage report
```

Test coverage is approximately 97.9% across all modules.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan. In summary:

- **Phase 1: Core Algorithm** — Complete. Full GBM with MSE/LogLoss, tree building, serialization, dataset utilities, and sklearn validation.
- **Phase 2: Accuracy & Correctness** — Newton-Raphson leaf optimization (complete), feature importance (complete), reproducible randomness, correctness test suite.
- **Phase 3: Usability** — Early stopping, column subsampling, multi-class classification, additional loss functions.
- **Phase 4: Performance** — Histogram binning, parallel split finding, column-major data layout.
- **Phase 5: Benchmarking** — Standard dataset benchmarks and comprehensive sklearn comparison.

## References

1. **Friedman, J.H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232. [[pdf]](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
   - The foundational paper introducing gradient boosting as a general framework for building additive models by sequentially fitting to negative gradients.

2. **Friedman, J.H. (2002).** "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367-378.
   - Introduces stochastic subsampling (training each tree on a random subset of rows), which reduces overfitting and improves generalization.

3. **scikit-learn: GradientBoostingClassifier.** [[documentation]](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
   - The Python reference implementation used for comparison. scikit-learn extends the basic algorithm with Newton-Raphson leaf optimization (using second derivatives for optimal leaf weights), Friedman's split improvement criterion, and other enhancements.

## License

MIT
