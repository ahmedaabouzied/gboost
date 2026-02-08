package gboost

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Dataset holds loaded CSV data with features, target, and any label encodings.
type Dataset struct {
	X              [][]float64
	Y              []float64
	Encodings      map[int]map[string]float64 // featureIndex → (stringValue → numericValue)
	TargetEncoding map[string]float64         // target column encoding, nil if target is numeric
	Header         []string
}

// LoadCSV reads a CSV file into memory and returns a Dataset. The targetColumn
// specifies which column is the target (supports negative indexing, e.g. -1 for
// last column). Column types are inferred per-column: if any value in a column
// is non-numeric, the entire column is label-encoded.
func LoadCSV(path string, targetColumn int, hasHeader bool) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}

	if len(records) == 0 {
		return nil, ErrEmptyDataset
	}

	ds := &Dataset{
		Encodings: make(map[int]map[string]float64),
	}

	startRow := 0
	if hasHeader {
		ds.Header = records[0]
		startRow = 1
	}

	if startRow >= len(records) {
		return nil, ErrEmptyDataset
	}

	nCols := len(records[startRow])
	if nCols < 2 {
		return nil, fmt.Errorf("csv must have at least 2 columns (got %d)", nCols)
	}

	// Resolve negative target column index.
	if targetColumn < 0 {
		targetColumn = nCols + targetColumn
	}
	if targetColumn < 0 || targetColumn >= nCols {
		return nil, fmt.Errorf("target column %d out of range for %d columns", targetColumn, nCols)
	}

	dataRows := records[startRow:]
	nRows := len(dataRows)

	// Trim whitespace from all cells and validate row lengths.
	for i, record := range dataRows {
		if len(record) != nCols {
			return nil, fmt.Errorf("row %d has %d columns, expected %d", i+startRow, len(record), nCols)
		}
		for j := range record {
			dataRows[i][j] = strings.TrimSpace(record[j])
		}
	}

	// Pass 1: check for empty values and determine which columns are string-typed.
	isStringCol := make([]bool, nCols)
	for _, record := range dataRows {
		for col, val := range record {
			if val == "" {
				return nil, fmt.Errorf("empty value at column %d", col)
			}
			if !isStringCol[col] {
				if _, err := strconv.ParseFloat(val, 64); err != nil {
					isStringCol[col] = true
				}
			}
		}
	}

	// Pass 2: build label encodings for string columns.
	colEncodings := make(map[int]map[string]int) // csv col → string → int label
	for col := 0; col < nCols; col++ {
		if !isStringCol[col] {
			continue
		}
		enc := make(map[string]int)
		next := 0
		for _, record := range dataRows {
			if _, ok := enc[record[col]]; !ok {
				enc[record[col]] = next
				next++
			}
		}
		colEncodings[col] = enc
	}

	// Pass 3: parse all data.
	ds.X = make([][]float64, nRows)
	ds.Y = make([]float64, nRows)

	for i, record := range dataRows {
		features := make([]float64, 0, nCols-1)
		for col, val := range record {
			var v float64
			if isStringCol[col] {
				v = float64(colEncodings[col][val])
			} else {
				v, _ = strconv.ParseFloat(val, 64) // already validated in pass 1
			}
			if col == targetColumn {
				ds.Y[i] = v
			} else {
				features = append(features, v)
			}
		}
		ds.X[i] = features
	}

	// Build exported encodings keyed by feature index (not csv column index).
	featureIdx := 0
	for col := 0; col < nCols; col++ {
		if colEncodings[col] == nil {
			if col != targetColumn {
				featureIdx++
			}
			continue
		}
		enc := make(map[string]float64, len(colEncodings[col]))
		for s, i := range colEncodings[col] {
			enc[s] = float64(i)
		}
		if col == targetColumn {
			ds.TargetEncoding = enc
		} else {
			ds.Encodings[featureIdx] = enc
			featureIdx++
		}
	}

	return ds, nil
}

// TrainTestSplit splits features and targets into training and testing sets.
// testRatio is the fraction of data used for testing (must be between 0 and 1
// exclusive). seed controls the random shuffle for reproducibility.
func TrainTestSplit(X [][]float64, y []float64, testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error) {
	n := len(X)
	if n != len(y) {
		return nil, nil, nil, nil, ErrLengthMismatch
	}
	if n < 2 {
		return nil, nil, nil, nil, fmt.Errorf("need at least 2 samples to split, got %d", n)
	}
	if testRatio <= 0 || testRatio >= 1 {
		return nil, nil, nil, nil, fmt.Errorf("testRatio must be between 0 and 1 exclusive, got %f", testRatio)
	}

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	split := int(float64(n) * (1.0 - testRatio))
	if split < 1 {
		split = 1
	}
	if split >= n {
		split = n - 1
	}

	XTrain = make([][]float64, split)
	yTrain = make([]float64, split)
	XTest = make([][]float64, n-split)
	yTest = make([]float64, n-split)

	for i, idx := range indices[:split] {
		XTrain[i] = X[idx]
		yTrain[i] = y[idx]
	}
	for i, idx := range indices[split:] {
		XTest[i] = X[idx]
		yTest[i] = y[idx]
	}

	return XTrain, XTest, yTrain, yTest, nil
}

// Split is a convenience method that calls TrainTestSplit on the Dataset's X and Y.
func (ds *Dataset) Split(testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error) {
	return TrainTestSplit(ds.X, ds.Y, testRatio, seed)
}
