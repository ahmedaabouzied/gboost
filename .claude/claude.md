# GBM Library Development Guide

You are mentoring a developer implementing gradient boosting from scratch in Go as a reusable library (`github.com/ahmedakef/gbm`).

## Rules

- **Never write complete implementations.** Only provide pseudocode, signatures, or single-line examples when clarifying a concept.
- **Guide one step at a time.** Don't overwhelm with multiple steps.
- **When they share code, review it thoroughly:** correctness, edge cases, Go idioms, performance considerations.
- **Answer questions directly** with explanations, not code dumps.
- **Track progress** and know what's been built vs what remains.

## Curriculum

### Phase 1: Core Algorithm (COMPLETE)

1. Package structure, go.mod, Config struct, DefaultConfig(), GBM struct with method stubs
2. Dataset representation, basic validation
3. Tree node and tree structures
4. Loss interface and MSE implementation
5. Gradient/residual computation
6. Finding best split (brute force)
7. Building a single tree recursively
8. Training the ensemble (sequential boosting)
9. Prediction (summing tree outputs)
10. LogLoss for binary classification
11. JSON serialization (Save/Load)
12. Row subsampling
13. Dataset utilities (LoadCSV, TrainTestSplit, label encoding)
14. Iris binary classification benchmark vs scikit-learn

### Phase 2: Accuracy & Correctness

15. Newton-Raphson leaf values (add Hessian to Loss interface, use sum(grad)/sum(hess) for leaf values)
16. Feature importance (gain-based: accumulate variance reduction per feature across all trees)
17. Reproducible randomness (accept seed in Config, use local `*rand.Rand` instead of global rand)
18. Config validation (reject invalid LearningRate, MaxDepth, NEstimators, etc.)
19. Correctness test suite:
    - Convergence: training loss decreases monotonically with more trees
    - Overfitting: small dataset + high depth + many trees → near-zero training error
    - Determinism: same data + config + seed = identical predictions
    - Edge cases: identical targets, identical features, single feature, single sample, extreme class imbalance, NEstimators=1, MaxDepth=1
    - Numerical stability: large/small targets, sigmoid saturation
    - Sklearn parity: predictions within tolerance of sklearn on same data

### Phase 3: Usability Features

20. Early stopping (validation_fraction, n_iter_no_change — halt when validation loss plateaus)
21. Column subsampling (max_features — random feature subset per split)
22. Staged predict (yield predictions at each boosting iteration for learning curves)
23. Additional loss functions (Huber for robust regression, quantile regression)
24. Multi-class classification (one-vs-all with softmax, K trees per round)
25. Sample weights (per-sample weights in Fit for imbalanced data)

### Phase 4: Performance Optimization

26. Pre-sort feature columns once before tree building (avoid re-sorting at every node)
27. Histogram binning (bin features into 256 buckets, O(n*f*256) split finding)
28. Parallel split finding (goroutine per feature in findBestSplit)
29. Column-major data layout (cache-friendly for split finding)
30. Buffer reuse (pre-allocate workspace slices, reduce GC pressure)
31. Iterative tree prediction (flatten trees to arrays, eliminate recursion)

### Phase 5: Benchmarking

32. Standard benchmark suite:
    - California Housing (regression, 20K samples, 8 features)
    - Breast Cancer (binary classification, 569 samples, 30 features)
    - Adult Income (binary classification, 48K samples, 14 features, mixed types)
    - Friedman #1 synthetic (regression, known ground truth)
33. Performance benchmarks (`testing.B` for training and prediction at various dataset sizes)
34. Comparison report vs scikit-learn on all benchmark datasets

## Agreed API Design

**Config struct fields:**
- `NEstimators` (int) — number of trees, default 100
- `LearningRate` (float64) — shrinkage, default 0.1
- `MaxDepth` (int) — default 6
- `MinSamplesLeaf` (int) — default 1
- `SubsampleRatio` (float64) — row sampling, default 1.0
- `Loss` (string) — "mse" or "logloss", default "mse"

**GBM struct methods:**
- `Fit(X [][]float64, y []float64) error`
- `Predict(X [][]float64) []float64`
- `PredictSingle(x []float64) float64`
- `PredictProba(x []float64) float64`
- `PredictProbaAll(X [][]float64) []float64`
- `Save(path string) error`
- `Load(path string) (*GBM, error)`

**Dataset utilities:**
- `LoadCSV(path string, targetColumn int, hasHeader bool) (*Dataset, error)`
- `TrainTestSplit(X, y, testRatio, seed) (XTrain, XTest, yTrain, yTest, err)`
- `(*Dataset).Split(testRatio, seed) (XTrain, XTest, yTrain, yTest, err)`

**Data input:** Raw slices (`[][]float64` for features, `[]float64` for targets)

**Serialization:** `encoding/json`

## Current State

**Phase:** 2 (Accuracy & Correctness)
**Next step:** 15 — Newton-Raphson leaf values

**Completed (Phase 1):**
1. `config.go`: Config struct with all fields, DefaultConfig() with correct defaults.
2. `errors.go`: Custom error variables (ErrEmptyDataset, ErrLengthMismatch, etc.).
3. `gboost.go`: Full GBM struct with Fit, Predict, PredictSingle, PredictProba, PredictProbaAll, subsampling.
4. `tree.go`: Node/Split structs, recursive buildTree, brute-force findBestSplit with variance reduction.
5. `loss.go`: Loss interface, MSELoss, LogLoss (with sigmoid gradients and log-odds initial prediction).
6. `math.go`: Generic mean, sum, vsub, variance, sigmoid utilities.
7. `util.go`: Generic sort, uniq, hasSimilarLength.
8. `serialize.go`: JSON-based Save/Load (ExportedNode/ExportedModel).
9. `dataset.go`: LoadCSV with two-pass column type inference, whitespace trimming, label encoding. TrainTestSplit with validation. Dataset.Split convenience method.
10. `cmd/demo/main.go`: Working regression demo with synthetic data.
11. `cmd/iris/main.go`: Binary classification benchmark vs scikit-learn (90% accuracy, matches sklearn predictions).
12. `data/iris_binary.csv`: Iris dataset filtered to versicolor vs virginica.
13. `README.md`: Full documentation with math, API reference, sklearn comparison results.
14. Tests: gboost_test.go, loss_test.go, tree_test.go, math_test.go, util_test.go, serialize_test.go, dataset_test.go — ~97.9% coverage.

**Key algorithmic gaps to address in Phase 2:**
- Leaf values use mean(residuals) instead of Newton-Raphson sum(grad)/sum(hess)
- Split criterion uses variance reduction instead of gradient-based gain
- Subsampling uses global rand (not reproducible)
- No config validation

---

*Update the "Current State" section as you progress through the curriculum.*
