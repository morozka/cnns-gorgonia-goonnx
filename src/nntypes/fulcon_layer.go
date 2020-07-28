package nntypes

import (
	"fmt"

	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type FulconParams struct {
	OutSize    int    `json:"outsize"`
	Activation string `json:"activation"`
	//InputShape   tensor.Shape
	WeightsShape tensor.Shape
	Biased       bool
}

func (fp FulconParams) AppendLayer(net *Net) {
	//reshape of previous node to correct format
	var reshapedPredNode *grg.Node

	if net.Out.Shape().Dims() != 2 || net.Out.Shape()[0] != net.BatchSize {
		var shapeIn tensor.Shape
		shapeIn = append(shapeIn, net.BatchSize)
		shapeIn = append(shapeIn, net.Out.Shape().TotalSize()/net.BatchSize)
		reshapedPredNode = grg.Must(grg.Reshape(net.Out, shapeIn))
	} else {
		reshapedPredNode = net.Out
	}
	//creation of weights for layer
	weights := grg.NewMatrix(net.Graph, tensor.Float64, grg.WithShape(reshapedPredNode.Shape()[1], fp.OutSize), grg.WithInit(grg.GlorotU(1.0)), grg.WithName("fcl"+fmt.Sprint(len(net.Weights))))
	net.Weights = append(net.Weights, weights)
	//multiplication
	net.Out = grg.Must(grg.Mul(reshapedPredNode, weights))
	if fp.Biased {
		net.AddBiases()
	}
	net.ApplyActivation(fp.Activation)

}
