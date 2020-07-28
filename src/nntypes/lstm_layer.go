package nntypes

import (
	"fmt"

	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type LSTMParams struct {
	OutputSize int
	Cell       *grg.Node
	Hidden     *grg.Node
}

//dont work with batches!
func (lstm LSTMParams) AppendLayer(net *Net) {

	g := net.Graph
	hiddenSize := lstm.OutputSize

	prevSize := net.Out.Shape().TotalSize()
	inputVector := grg.Must(grg.Reshape(net.Out, []int{net.Out.Shape().TotalSize()}))
	Cell := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("cell_"+fmt.Sprint(len(net.Weights))), grg.WithInit(grg.Zeroes()))
	Hidden := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("hidden_"+fmt.Sprint(len(net.Weights))), grg.WithInit(grg.Zeroes()))

	net.LSTMNodes = append(net.LSTMNodes, Cell, Hidden)
	wix := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, prevSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wix_"+fmt.Sprint(len(net.Weights))))
	wih := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, hiddenSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wih_"+fmt.Sprint(len(net.Weights))))
	bias_i := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("bias_i_"+fmt.Sprint(len(net.Weights))), grg.WithInit(grg.Zeroes()))

	// output gate weights

	wox := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, prevSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wfx_"+fmt.Sprint(len(net.Weights))))
	woh := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, hiddenSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wfh_"+fmt.Sprint(len(net.Weights))))
	bias_o := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("bias_f_"+fmt.Sprint(len(net.Weights))), grg.WithInit(grg.Zeroes()))

	// forget gate weights

	wfx := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, prevSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wox_"+fmt.Sprint(len(net.Weights))))
	wfh := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, hiddenSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("woh_"+fmt.Sprint(len(net.Weights))))
	bias_f := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("bias_o_"+fmt.Sprint(len(net.Weights))), grg.WithInit(grg.Zeroes()))

	// cell write

	wcx := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, prevSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wcx_"))
	wch := grg.NewMatrix(g, tensor.Float64, grg.WithShape(hiddenSize, hiddenSize), grg.WithInit(grg.GlorotN(1.0)), grg.WithName("wch_"))
	bias_c := grg.NewVector(g, tensor.Float64, grg.WithShape(hiddenSize), grg.WithName("bias_c_"), grg.WithInit(grg.Zeroes()))
	net.Weights = append(net.Weights, wix, wih, bias_i, wox, woh, bias_o, wfx, wfh, bias_f)
	prevHidden := Hidden
	prevCell := Cell
	var h0, h1, inputGate *grg.Node
	h0 = grg.Must(grg.Mul(wix, inputVector))
	h1 = grg.Must(grg.Mul(wih, prevHidden))
	inputGate = grg.Must(grg.Sigmoid(grg.Must(grg.Add(grg.Must(grg.Add(h0, h1)), bias_i))))

	var h2, h3, forgetGate *grg.Node
	h2 = grg.Must(grg.Mul(wfx, inputVector))
	h3 = grg.Must(grg.Mul(wfh, prevHidden))
	forgetGate = grg.Must(grg.Sigmoid(grg.Must(grg.Add(grg.Must(grg.Add(h2, h3)), bias_f))))

	var h4, h5, outputGate *grg.Node
	h4 = grg.Must(grg.Mul(wox, inputVector))
	h5 = grg.Must(grg.Mul(woh, prevHidden))
	outputGate = grg.Must(grg.Sigmoid(grg.Must(grg.Add(grg.Must(grg.Add(h4, h5)), bias_o))))

	var h6, h7, cellWrite *grg.Node
	h6 = grg.Must(grg.Mul(wcx, inputVector))
	h7 = grg.Must(grg.Mul(wch, prevHidden))
	cellWrite = grg.Must(grg.Tanh(grg.Must(grg.Add(grg.Must(grg.Add(h6, h7)), bias_c))))

	// cell activations
	var retain, write *grg.Node
	retain = grg.Must(grg.HadamardProd(forgetGate, prevCell))
	write = grg.Must(grg.HadamardProd(inputGate, cellWrite))
	cell := grg.Must(grg.Add(retain, write))
	hidden := grg.Must(grg.HadamardProd(outputGate, grg.Must(grg.Tanh(cell))))
	net.LSTMNodes[len(net.LSTMNodes)-1] = cell
	net.LSTMNodes[len(net.LSTMNodes)-2] = hidden
	Cell = cell
	Hidden = hidden
	net.Out = hidden
}
