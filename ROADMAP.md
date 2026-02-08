# Roadmap

This document tracks the development roadmap for gboost. Phase 1 is complete — the library implements the core gradient boosting algorithm with regression and binary classification support, validated against scikit-learn. The following phases focus on closing the gap with production GBM libraries.

## Phase 1: Core Algorithm — Complete

The foundational gradient boosting implementation, covering the full training and prediction pipeline.

- [x] Package structure, configuration, and GBM struct
- [x] Input validation and error handling
- [x] Decision tree building with recursive splitting
- [x] Loss interface with MSE (regression) and Log Loss (binary classification)
- [x] Gradient computation and pseudo-residual fitting
- [x] Brute-force best split finding with variance reduction
- [x] Sequential boosting with learning rate shrinkage
- [x] Prediction: raw scores, probabilities, single and batch
- [x] Row subsampling (stochastic gradient boosting)
- [x] Model serialization (JSON save/load)
- [x] Dataset utilities: CSV loading with automatic label encoding, train/test splitting
- [x] Validated against scikit-learn on Iris dataset (identical predictions, 90% test accuracy)
- [x] Test coverage ~97.9%

## Phase 2: Accuracy & Correctness

Algorithmic improvements to match the accuracy of established implementations and a comprehensive correctness test suite.

- [x] **Newton-Raphson leaf optimization** — Replace mean-of-residuals leaf values with the second-order optimal `sum(gradients) / sum(hessians)`. This is the single largest accuracy improvement, enabling faster convergence and better probability calibration for classification. Requires adding a `Hessian` method to the `Loss` interface.
- [x] **Feature importance** — Compute gain-based feature importance by accumulating the variance reduction each feature contributes across all splits in all trees. Normalize to sum to 1.0. Essential for model interpretability.
- [ ] **Reproducible randomness** — Accept a random seed in `Config` and use a local `*rand.Rand` for subsampling instead of the global source. Required for deterministic training and meaningful benchmarks.
- [ ] **Configuration validation** — Validate all config fields in `Fit` (e.g., reject `LearningRate <= 0`, `MaxDepth < 1`, `NEstimators < 1`, `SubsampleRatio` outside (0, 1]).
- [ ] **Correctness test suite** — Systematic tests beyond unit tests:
  - Training loss decreases monotonically with more boosting rounds
  - Near-zero training error on small datasets with sufficient capacity
  - Deterministic output given same data, config, and seed
  - Edge cases: identical targets, identical features, single feature, extreme class imbalance, depth-1 stumps
  - Numerical stability with very large/small values and sigmoid saturation
  - Prediction parity with scikit-learn within tolerance on shared datasets

## Phase 3: Usability

Features that make the library practical for real-world use.

- [ ] **Early stopping** — Monitor validation loss during training and halt when it stops improving. Configurable via `ValidationFraction` and `NItersNoChange`. The standard way to prevent overfitting without manually tuning `NEstimators`.
- [ ] **Column subsampling** — Randomly sample a fraction of features at each split (`MaxFeatures` parameter). A strong regularizer, especially for high-dimensional data.
- [ ] **Staged prediction** — Yield predictions at each boosting iteration, enabling learning curve analysis and debugging of the training process.
- [ ] **Additional loss functions** — Huber loss for regression robust to outliers, quantile loss for prediction intervals.
- [ ] **Multi-class classification** — One-vs-all approach with softmax. Train K trees per boosting round (one per class), compute gradients from the multinomial cross-entropy loss.
- [ ] **Sample weights** — Support per-sample weights in `Fit` for cost-sensitive learning and handling class imbalance.

## Phase 4: Performance

Optimizations to scale to larger datasets and reduce training time.

- [ ] **Pre-sorted feature indices** — Sort each feature column once before tree building and reuse the sorted order at every node. Eliminates redundant O(n log n) sorts at each split.
- [ ] **Histogram-based split finding** — Bin continuous features into 256 discrete buckets. Reduces split finding from O(n × features × unique_values) to O(n × features × 256). The key optimization used by LightGBM and XGBoost.
- [ ] **Parallel split finding** — Evaluate candidate splits for each feature concurrently using goroutines. Embarrassingly parallel with near-linear speedup on multi-core hardware.
- [ ] **Column-major data layout** — Store features in column-major order for cache-friendly access during split evaluation, which iterates over samples within a single feature.
- [ ] **Buffer reuse** — Pre-allocate workspace slices for gradients, indices, and temporary arrays. Reuse across boosting iterations and tree nodes to reduce garbage collection pressure.
- [ ] **Iterative tree traversal** — Flatten trees into contiguous arrays and replace recursive prediction with iterative traversal for better cache locality and reduced function call overhead.

## Phase 5: Benchmarking

Comprehensive benchmarks against scikit-learn on standard datasets.

- [ ] **Standard benchmark datasets:**
  - California Housing — regression, 20,640 samples, 8 features
  - Breast Cancer — binary classification, 569 samples, 30 features
  - Adult Income — binary classification, 48,842 samples, 14 features (mixed categorical/numeric)
  - Friedman #1 — synthetic regression with known ground truth function
- [ ] **Go benchmarks** — `testing.B` benchmarks for training and prediction at various dataset sizes (1K, 10K, 100K samples) to track performance over time.
- [ ] **scikit-learn comparison report** — Train both implementations with identical hyperparameters on all benchmark datasets. Compare accuracy, log loss, RMSE, and training time. Publish results in documentation.
