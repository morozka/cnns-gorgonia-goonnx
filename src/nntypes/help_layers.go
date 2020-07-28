package nntypes

import (
	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MaxPoolParams struct {
	KernelSize tensor.Shape
	Padding    []int
	Stride     []int
	PrintOut   bool
}

func (mxp MaxPoolParams) AppendLayer(net *Net) {
	net.Out = grg.Must(grg.MaxPool2D(net.Out, mxp.KernelSize, mxp.Padding, mxp.Stride))
	if mxp.PrintOut {
		net.Cost = grg.Must(grg.Neg(grg.Must(grg.Neg(net.Out))))
	}
}

type DropOutParams struct {
	DropProb float64
}

func (drp DropOutParams) AppendLayer(net *Net) {
	net.Out = grg.Must(grg.Dropout(net.Out, drp.DropProb))
}

type BatchNormParams struct {
	InputShape tensor.Shape
	Scale      *grg.Node
	Bias       *grg.Node
	Gama       *grg.Node
	Beta       *grg.Node
}

func (bnp BatchNormParams) AppendLayer(net *Net) {
	scale := grg.NewTensor(net.Graph, tensor.Float64, 4, grg.WithShape(net.Out.Shape()...), grg.WithInit(grg.GlorotU(1.0)))
	bias := grg.NewTensor(net.Graph, tensor.Float64, 4, grg.WithShape(net.Out.Shape()...), grg.WithInit(grg.GlorotU(1.0)))
	retval, y, b, op, err := grg.BatchNorm(net.Out, scale, bias, 0.9, 0.00001)
	if err != nil {
		panic(err)
	}
	_ = op
	net.Weights = append(net.Weights, y, b)
	bnp.Gama = y
	bnp.Beta = b
	net.Out = retval
}
