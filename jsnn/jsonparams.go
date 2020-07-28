package jsnn

import (
	"fmt"

	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type param interface {
	CreateLayer(*Net, *grg.Node) *grg.Node
}

type fullConParams struct {
	OutSize    int    `json:"outsize"`
	Activation string `json:"activation"`
}

//CreateLayer - fullcon expects 4d or 2d tensor, will panic on 3d and 1d
/*
	Carefull in use, no Vectors please. 2d tensor shape = (batch, width)
*/
func (fc fullConParams) CreateLayer(nn *Net, input *grg.Node) *grg.Node {
	var nbatch int
	var datasize int

	if input.Shape().Dims() == 3 || input.Shape().Dims() == 1 {
		panic("Wrong fullcon tensor shape")
	} else if input.Shape().Dims() == 4 {
		nbatch = input.Shape()[0]
		datasize = input.Shape()[1] * input.Shape()[2] * input.Shape()[3]
		var err error
		input, err = grg.Reshape(input, tensor.Shape{nbatch, datasize})
		if err != nil {
			panic("Reshape to fullcon failed!")
		}
	} else {
		nbatch = input.Shape()[0]
		datasize = input.Shape()[1]
	}
	fmt.Println(fc.OutSize)
	w := grg.NewMatrix(nn.Graph, tensor.Float64, grg.WithShape(datasize, fc.OutSize), grg.WithInit(grg.GlorotN(0.5)), grg.WithName(fmt.Sprintf("weight %v", wi)))
	wi++
	nn.W = append(nn.W, w)
	l := grg.Must(grg.Mul(input, w))
	nn.L = append(nn.L, l)
	out := l
	switch fc.Activation {
	case "sigm":
		out = grg.Must(grg.Sigmoid(out))
		return out
	case "tanh":
		out = grg.Must(grg.Tanh(out))
		return out
	case "softplus":
		return grg.Must(grg.Softplus(out))
	case "softmax":
		return grg.Must(grg.SoftMax(out))
	case "off":
		return out
	default:
		panic("No such activation for fully connected layer, check your json")
	}
}

type convParams struct {
	Kernels    int    `json:"kernels"`
	KernelSize int    `json:"kernelsize"`
	Stride     []int  `json:"stride"`
	Padding    []int  `json:"pad"`
	Dilation   []int  `json:"dilation"`
	Activation string `json:"activation"`
}

func (cnv convParams) CreateLayer(nn *Net, input *grg.Node) *grg.Node {
	if input.Dims() != 4 {
		panic("Wrong tensor, tensor should be 4d for Conv2d")
	}
	kern := grg.NewTensor(nn.Graph, tensor.Float64, 4, grg.WithShape(cnv.Kernels, 1, cnv.KernelSize, cnv.KernelSize), grg.WithInit(grg.GlorotN(0.5)))
	nn.W = append(nn.W, kern)

	con := grg.Must(grg.Conv2d(input, kern, kern.Shape(), cnv.Padding, cnv.Stride, cnv.Dilation))

	switch cnv.Activation {
	case "leaky":
		return grg.Must(grg.LeakyRelu(con, 0.01))
	case "tanh":
		return grg.Must(grg.Tanh(con))
	case "sigm":
		return grg.Must(grg.Sigmoid(con))
	case "softplus":
		return grg.Must(grg.Softplus(con))
	case "softmax":
		return grg.Must(grg.SoftMax(con))
	case "rect":
		return grg.Must(grg.Rectify(con))
	case "off":
		return con
	default:
		panic("No such activation for conv layer, check your json")
	}
}

type poolParams struct {
	KernelSize int   `json:"kernelsize"`
	Padding    []int `json:"pad"`
	Stride     []int `json:"stride"`
}

func (p poolParams) CreateLayer(nn *Net, input *grg.Node) *grg.Node {
	return grg.Must(grg.MaxPool2D(input, tensor.Shape([]int{p.KernelSize, p.KernelSize}), p.Padding, p.Stride))
}

/*


type dropoutParams struct {
	Probability float64 `json:"probability"`
}

func (d dropoutParams) CreateLayer(inSize []int) Layer {
	//return NewDropOutLayer(tensor.DimsLen(inSize), d.Probability)
}
*/
