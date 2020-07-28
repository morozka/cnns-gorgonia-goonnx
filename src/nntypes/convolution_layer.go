package nntypes

import (
	"fmt"
	"math"

	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type ConvolutionParams struct {
	KernelsShape tensor.Shape
	KernelSize   tensor.Shape
	Stride       []int  `json:"stride"`
	Padding      []int  `json:"pad"`
	Dilation     []int  `json:"dilation"`
	Activation   string `json:"activation"`
	Biased       bool
	Alpha        float64
	BatchNorm    bool
	PrintOut     bool
	Named        bool
}

func (cnv ConvolutionParams) AppendLayer(net *Net) {
	if net.Out.Dims() != 4 {
		panic("Wrong tensor, tensor should be 4d for Conv2d")
	}
	if net.IsPretrained {
		parsedWeights := cnv.ParseWeights(net)
		if cnv.BatchNorm {
			parsedWeights = cnv.DenormalizeWeights(parsedWeights, net)
		}
		lk := tensor.New(tensor.WithShape(cnv.KernelsShape...), tensor.WithBacking(parsedWeights["kernels"]))
		kernels := grg.NewTensor(net.Graph, tensor.Float32, 4, grg.WithShape(cnv.KernelsShape...), grg.WithName("conv"+fmt.Sprint(len(net.Weights))), grg.WithValue(lk))
		net.Weights = append(net.Weights, kernels)
		net.Out = grg.Must(grg.Conv2d(net.Out, kernels, cnv.KernelSize, cnv.Padding, cnv.Stride, cnv.Dilation))
		if cnv.Biased || cnv.BatchNorm {
			net.AddBiasesPretrained(parsedWeights["biases"])
		}
		net.ApplyActivation(cnv.Activation, cnv.Alpha)
		if cnv.PrintOut {
			net.Cost = grg.Must(grg.Neg(grg.Must(grg.Neg(net.Out))))
		}
	} else {
		kernels := grg.NewTensor(net.Graph, tensor.Float64, 4, grg.WithShape(cnv.KernelsShape...), grg.WithInit(grg.GlorotU(1.0)))
		net.Weights = append(net.Weights, kernels)
		net.Out = grg.Must(grg.Conv2d(net.Out, kernels, cnv.KernelSize, cnv.Padding, cnv.Stride, cnv.Dilation))
		net.ApplyActivation(cnv.Activation, cnv.Alpha)
		if cnv.Biased {
			net.AddBiases()
		}
	}
}
func (cnv ConvolutionParams) ParseWeights(net *Net) map[string][]float32 {
	res := make(map[string][]float32)
	if cnv.BatchNorm {
		nb := cnv.KernelsShape[0]
		nk := cnv.KernelsShape.TotalSize()
		res["biases"] = make([]float32, 0)
		res["biases"] = append(res["biases"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nb]...)
		net.LastWeight += nb
		res["gammas"] = make([]float32, 0)
		res["gammas"] = append(res["gammas"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nb]...)
		net.LastWeight += nb
		res["means"] = make([]float32, 0)
		res["means"] = append(res["means"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nb]...)
		net.LastWeight += nb
		res["vars"] = make([]float32, 0)
		res["vars"] = append(res["vars"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nb]...)
		net.LastWeight += nb
		res["kernels"] = make([]float32, 0)
		res["kernels"] = append(res["kernels"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nk]...)
		net.LastWeight += nk
	} else {
		if cnv.Biased {
			nb := cnv.KernelsShape[0]
			nk := cnv.KernelsShape.TotalSize()
			res["biases"] = make([]float32, 0)
			res["biases"] = append(res["biases"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nb]...)
			net.LastWeight += nb
			res["kernels"] = make([]float32, 0)
			res["kernels"] = append(res["kernels"], net.UnparsedWeights[net.LastWeight:net.LastWeight+nk]...)
			net.LastWeight += nk
		}
	}
	return res
}

func (cnv ConvolutionParams) DenormalizeWeights(lw map[string][]float32, net *Net) map[string][]float32 {
	gammas := lw["gammas"]
	vars := lw["vars"]
	means := lw["means"]
	biases := lw["biases"]
	kernel_weights := lw["kernels"]

	for i := 0; i < cnv.KernelsShape[0]; i++ {
		scale := gammas[i] / float32(math.Sqrt(float64(vars[i]+net.Epsilon)))

		biases[i] = (biases[i] - means[i]*scale)

		isize := cnv.KernelsShape[1] * cnv.KernelsShape[2] * cnv.KernelsShape[3]
		for j := 0; j < isize; j++ {

			kernel_weights[isize*i+j] = kernel_weights[isize*i+j] * scale

		}
	}
	lw["biases"] = biases
	lw["kernels"] = kernel_weights
	if cnv.PrintOut {
		fmt.Println(lw["biases"])
		fmt.Println(len(lw["biases"]))
	}

	return lw
}
