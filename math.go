package gboost

import (
	"math"

	"golang.org/x/exp/constraints"
)

func mean[T constraints.Float | constraints.Integer](data []T) float64 {
	if len(data) == 0 {
		return 0
	}

	var sum = sum(data)
	return float64(sum) / float64(len(data))
}

func sum[T constraints.Float | constraints.Integer](data []T) T {
	var s T
	for _, d := range data {
		s += d
	}
	return s
}

func vsub[T constraints.Float | constraints.Integer](a, b []T) []T {
	if len(a) != len(b) {
		panic("vsub: mismatched slice lengths")
	}
	result := make([]T, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func variance[T constraints.Float | constraints.Integer](data []T) float64 {
	m := mean(data)

	sd := make([]float64, len(data))
	for i := range data {
		sd[i] = float64(data[i]) - m
	}
	for i := range sd {
		sd[i] = sd[i] * sd[i]
	}

	s := sum(sd)
	return s / float64(len(data))
}

func sigmoid[T constraints.Float | constraints.Integer](x T) float64 {
	// sigmoid(x) = 1 / (1 + e^(-x))
	return 1 / (1 + math.Exp(-float64(x)))
}
