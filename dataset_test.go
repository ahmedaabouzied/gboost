package gboost

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTestCSV(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCSVNumeric(t *testing.T) {
	path := writeTestCSV(t, "numeric.csv", `1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
`)
	ds, err := LoadCSV(path, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.X) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(ds.X))
	}
	if len(ds.X[0]) != 2 {
		t.Fatalf("expected 2 features, got %d", len(ds.X[0]))
	}
	if ds.Y[0] != 3.0 || ds.Y[2] != 9.0 {
		t.Fatalf("unexpected Y values: %v", ds.Y)
	}
	if len(ds.Encodings) != 0 {
		t.Fatalf("expected no encodings for numeric data, got %v", ds.Encodings)
	}
}

func TestLoadCSVWithHeader(t *testing.T) {
	path := writeTestCSV(t, "header.csv", `a,b,target
1.0,2.0,3.0
4.0,5.0,6.0
`)
	ds, err := LoadCSV(path, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.Header) != 3 {
		t.Fatalf("expected 3 header columns, got %d", len(ds.Header))
	}
	if ds.Header[0] != "a" || ds.Header[2] != "target" {
		t.Fatalf("unexpected header: %v", ds.Header)
	}
	if len(ds.X) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(ds.X))
	}
}

func TestLoadCSVWithStringEncoding(t *testing.T) {
	path := writeTestCSV(t, "strings.csv", `5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
5.0,3.4,1.5,0.2,setosa
`)
	ds, err := LoadCSV(path, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.X) != 4 {
		t.Fatalf("expected 4 rows, got %d", len(ds.X))
	}
	if len(ds.X[0]) != 4 {
		t.Fatalf("expected 4 features, got %d", len(ds.X[0]))
	}
	// setosa should be 0, versicolor 1, virginica 2
	if ds.Y[0] != 0.0 || ds.Y[1] != 1.0 || ds.Y[2] != 2.0 {
		t.Fatalf("unexpected encodings: %v", ds.Y)
	}
	// setosa seen again should still be 0
	if ds.Y[3] != 0.0 {
		t.Fatalf("expected repeated label to get same encoding, got %f", ds.Y[3])
	}
	// Target encoding should be in the TargetEncoding field.
	if ds.TargetEncoding == nil {
		t.Fatal("expected TargetEncoding to be set")
	}
	if ds.TargetEncoding["setosa"] != 0.0 || ds.TargetEncoding["versicolor"] != 1.0 || ds.TargetEncoding["virginica"] != 2.0 {
		t.Fatalf("unexpected target encoding map: %v", ds.TargetEncoding)
	}
}

func TestLoadCSVMixedColumnFullyEncoded(t *testing.T) {
	// Column 1 has "3.0" and "NA" — since NA is non-numeric, the entire
	// column should be label-encoded (not partially parsed as float).
	path := writeTestCSV(t, "mixed.csv", `1.0,3.0,10.0
2.0,NA,20.0
3.0,3.0,30.0
`)
	ds, err := LoadCSV(path, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	// Column 1 (feature index 1) should be fully encoded: "3.0"→0, "NA"→1
	enc, ok := ds.Encodings[1]
	if !ok {
		t.Fatal("expected encoding for feature index 1")
	}
	if enc["3.0"] != 0.0 {
		t.Fatalf("expected '3.0' encoded as 0, got %f", enc["3.0"])
	}
	if enc["NA"] != 1.0 {
		t.Fatalf("expected 'NA' encoded as 1, got %f", enc["NA"])
	}
	// Row 0 feature 1 should be 0.0 (encoded "3.0"), not 3.0
	if ds.X[0][1] != 0.0 {
		t.Fatalf("expected encoded value 0.0 for '3.0', got %f", ds.X[0][1])
	}
	// Row 2 feature 1 should also be 0.0 (same string "3.0")
	if ds.X[2][1] != 0.0 {
		t.Fatalf("expected encoded value 0.0 for repeated '3.0', got %f", ds.X[2][1])
	}
}

func TestLoadCSVEncodingsKeyedByFeatureIndex(t *testing.T) {
	// CSV: col0=numeric, col1=string, col2(target)=string, col3=string
	// After removing target: feature 0=col0, feature 1=col1, feature 2=col3
	// Encodings should use feature indices 1 and 2, plus -1 for target.
	path := writeTestCSV(t, "reindex.csv", `1.0,cat,yes,big
2.0,dog,no,small
`)
	ds, err := LoadCSV(path, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if ds.TargetEncoding == nil {
		t.Fatal("expected TargetEncoding to be set")
	}
	if _, ok := ds.Encodings[1]; !ok {
		t.Fatal("expected encoding at feature index 1 (csv col 1)")
	}
	if _, ok := ds.Encodings[2]; !ok {
		t.Fatal("expected encoding at feature index 2 (csv col 3)")
	}
	// Feature 0 is numeric, should have no encoding.
	if _, ok := ds.Encodings[0]; ok {
		t.Fatal("did not expect encoding for numeric feature 0")
	}
}

func TestLoadCSVWhitespaceTrimmed(t *testing.T) {
	path := writeTestCSV(t, "spaces.csv", ` 5.1 , 3.5 , setosa
7.0,3.2, setosa
`)
	ds, err := LoadCSV(path, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	// Numeric values should be parsed correctly after trimming.
	if ds.X[0][0] != 5.1 {
		t.Fatalf("expected 5.1 after trimming, got %f", ds.X[0][0])
	}
	// " setosa" and "setosa" should be the same label after trimming.
	if ds.Y[0] != ds.Y[1] {
		t.Fatalf("expected same encoding for trimmed strings, got %f and %f", ds.Y[0], ds.Y[1])
	}
}

func TestLoadCSVEmptyValue(t *testing.T) {
	path := writeTestCSV(t, "emptyval.csv", `1.0,,3.0
4.0,5.0,6.0
`)
	_, err := LoadCSV(path, -1, false)
	if err == nil {
		t.Fatal("expected error for empty cell value")
	}
}

func TestLoadCSVNegativeIndex(t *testing.T) {
	path := writeTestCSV(t, "neg.csv", `1.0,2.0,3.0
4.0,5.0,6.0
`)
	ds, err := LoadCSV(path, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	if ds.X[0][0] != 1.0 || ds.X[0][1] != 2.0 {
		t.Fatalf("unexpected features: %v", ds.X[0])
	}
	if ds.Y[0] != 3.0 {
		t.Fatalf("unexpected Y: %f", ds.Y[0])
	}
}

func TestLoadCSVEmptyFile(t *testing.T) {
	path := writeTestCSV(t, "empty.csv", "")
	_, err := LoadCSV(path, 0, false)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestLoadCSVHeaderOnly(t *testing.T) {
	path := writeTestCSV(t, "headeronly.csv", "a,b,c\n")
	_, err := LoadCSV(path, 0, true)
	if err == nil {
		t.Fatal("expected error for header-only file")
	}
}

func TestLoadCSVInvalidTargetColumn(t *testing.T) {
	path := writeTestCSV(t, "inv.csv", `1.0,2.0
3.0,4.0
`)
	_, err := LoadCSV(path, 5, false)
	if err == nil {
		t.Fatal("expected error for out-of-range target column")
	}
}

func TestLoadCSVSingleColumn(t *testing.T) {
	path := writeTestCSV(t, "single.csv", `1.0
2.0
`)
	_, err := LoadCSV(path, 0, false)
	if err == nil {
		t.Fatal("expected error for single-column CSV")
	}
}

func TestTrainTestSplit(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	XTrain, XTest, yTrain, yTest, err := TrainTestSplit(X, y, 0.2, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(XTrain) != 8 {
		t.Fatalf("expected 8 train samples, got %d", len(XTrain))
	}
	if len(XTest) != 2 {
		t.Fatalf("expected 2 test samples, got %d", len(XTest))
	}
	if len(yTrain) != 8 || len(yTest) != 2 {
		t.Fatal("y split sizes don't match X split sizes")
	}

	// Verify all original data is present.
	seen := make(map[float64]bool)
	for _, v := range yTrain {
		seen[v] = true
	}
	for _, v := range yTest {
		seen[v] = true
	}
	if len(seen) != 10 {
		t.Fatalf("expected all 10 values present, got %d unique", len(seen))
	}
}

func TestTrainTestSplitReproducible(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}, {4}, {5}}
	y := []float64{1, 2, 3, 4, 5}

	_, _, yTrain1, _, err1 := TrainTestSplit(X, y, 0.4, 99)
	_, _, yTrain2, _, err2 := TrainTestSplit(X, y, 0.4, 99)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}

	for i := range yTrain1 {
		if yTrain1[i] != yTrain2[i] {
			t.Fatal("same seed should produce identical splits")
		}
	}
}

func TestTrainTestSplitShuffles(t *testing.T) {
	X := make([][]float64, 100)
	y := make([]float64, 100)
	for i := range X {
		X[i] = []float64{float64(i)}
		y[i] = float64(i)
	}

	_, _, yTrain, _, err := TrainTestSplit(X, y, 0.2, 42)
	if err != nil {
		t.Fatal(err)
	}

	inOrder := true
	for i := 1; i < len(yTrain); i++ {
		if yTrain[i] < yTrain[i-1] {
			inOrder = false
			break
		}
	}
	if inOrder {
		t.Fatal("expected shuffled data, but got sorted order")
	}
}

func TestTrainTestSplitLengthMismatch(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}}
	y := []float64{1, 2}
	_, _, _, _, err := TrainTestSplit(X, y, 0.3, 42)
	if err == nil {
		t.Fatal("expected error for length mismatch")
	}
}

func TestTrainTestSplitTooFewSamples(t *testing.T) {
	X := [][]float64{{1}}
	y := []float64{1}
	_, _, _, _, err := TrainTestSplit(X, y, 0.5, 42)
	if err == nil {
		t.Fatal("expected error for single sample")
	}
}

func TestTrainTestSplitInvalidRatio(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}}
	y := []float64{1, 2, 3}

	for _, ratio := range []float64{0.0, 1.0, -0.5, 1.5} {
		_, _, _, _, err := TrainTestSplit(X, y, ratio, 42)
		if err == nil {
			t.Fatalf("expected error for testRatio=%f", ratio)
		}
	}
}

func TestDatasetSplit(t *testing.T) {
	ds := &Dataset{
		X: [][]float64{{1}, {2}, {3}, {4}, {5}},
		Y: []float64{1, 2, 3, 4, 5},
	}
	XTrain, XTest, _, _, err := ds.Split(0.4, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(XTrain)+len(XTest) != 5 {
		t.Fatalf("expected 5 total samples, got %d", len(XTrain)+len(XTest))
	}
}
