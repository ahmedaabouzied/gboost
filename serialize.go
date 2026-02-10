package gboost

import (
	"encoding/json"
	"os"
)

// ExportedNode is the JSON-serializable representation of a Node
type ExportedNode struct {
	FeatureIndex int           `json:"feature_index"`
	Threshold    float64       `json:"threshold"`
	Value        float64       `json:"value"`
	IsLeaf       bool          `json:"is_leaf"`
	Left         *ExportedNode `json:"left,omitempty"`
	Right        *ExportedNode `json:"right,omitempty"`
}

// ExportedModel is the JSON-serializable representation of a GBM model
type ExportedModel struct {
	Config            Config          `json:"config"`
	InitialPrediction float64         `json:"initial_prediction"`
	Trees             []*ExportedNode `json:"trees"`
}

// toExported converts an internal Node to an ExportedNode
func (n *Node) toExported() *ExportedNode {
	if n == nil {
		return nil
	}

	isLeaf := n.Left == nil && n.Right == nil

	return &ExportedNode{
		FeatureIndex: n.FeatureIndex,
		Threshold:    n.Threshold,
		Value:        n.Value,
		IsLeaf:       isLeaf,
		Left:         n.Left.toExported(),
		Right:        n.Right.toExported(),
	}
}

// nodeFromExported converts an ExportedNode back to an internal Node
func nodeFromExported(e *ExportedNode) *Node {
	if e == nil {
		return nil
	}

	return &Node{
		FeatureIndex: e.FeatureIndex,
		Threshold:    e.Threshold,
		Value:        e.Value,
		Left:         nodeFromExported(e.Left),
		Right:        nodeFromExported(e.Right),
	}
}

// toExported converts a GBM model to an ExportedModel
func (g *GBM) toExported() *ExportedModel {
	trees := make([]*ExportedNode, len(g.trees))
	for i, tree := range g.trees {
		trees[i] = tree.toExported()
	}

	return &ExportedModel{
		Config:            g.Config,
		InitialPrediction: g.initialPrediction,
		Trees:             trees,
	}
}

// fromExported restores a GBM model from an ExportedModel
func fromExported(e *ExportedModel) *GBM {
	trees := make([]*Node, len(e.Trees))
	for i, tree := range e.Trees {
		trees[i] = nodeFromExported(tree)
	}

	return &GBM{
		Config:            e.Config,
		initialPrediction: e.InitialPrediction,
		trees:             trees,
		isFitted:          true,
		loss:              createLossFunction(e.Config),
	}
}

// Save writes the trained model to a JSON file at the given path.
// Returns [ErrModelNotFitted] if the model has not been trained.
// The saved file can be restored with [Load].
func (g *GBM) Save(path string) error {
	if !g.isFitted {
		return ErrModelNotFitted
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(g.toExported())
}

// Load reads a trained model from a JSON file previously written by [GBM.Save].
// The returned model is ready for prediction without retraining.
func Load(path string) (*GBM, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var exported ExportedModel
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&exported); err != nil {
		return nil, err
	}

	return fromExported(&exported), nil
}
