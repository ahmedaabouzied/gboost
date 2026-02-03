package gboost

func hasSimilarLength(X [][]float64) bool {
	l := len(X[0])
	for _, row := range X {
		if len(row) != l {
			return false
		}
	}
	return true
}
