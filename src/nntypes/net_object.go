package nntypes

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"gorgonia.org/gorgonia"
	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Net struct {
	Graph           *grg.ExprGraph
	Weights         []*grg.Node
	Layers          []*grg.Node
	LSTMNodes       []*grg.Node
	LSTMValues      []*grg.Value
	Solver          grg.Solver
	Machine         grg.VM
	Input           *grg.Node
	Out             *grg.Node
	LearnNode       *grg.Node
	Cost            *grg.Node
	PredictionValue grg.Value
	CostValue       grg.Value
	BatchSize       int
	UnparsedWeights []float32
	IsPretrained    bool
	LastWeight      int
	Epsilon         float32
	TrueNode        *grg.Node
}

//Interface for unique combinations of params of layers
type Params interface {
	AppendLayer(*Net)
}

//GenerateNet - lps is a list of parameters for layers
func GenerateNet(lps []Params, inputShape, answerShape tensor.Shape) Net {
	var net Net
	net.Epsilon = 0.001
	net.BatchSize = 1
	net.Graph = grg.NewGraph()
	net.Input = grg.NewTensor(net.Graph, tensor.Float64, inputShape.Dims(), grg.WithShape(inputShape...))
	net.Out = net.Input
	net.LSTMValues = make([]*grg.Value, 14)
	for i := 0; i < len(lps); i++ {
		lps[i].AppendLayer(&net)
	}
	net.LearnNode = grg.NewTensor(net.Graph, tensor.Float64, answerShape.Dims(), grg.WithShape(answerShape...), grg.WithName("LearnNode"))
	grg.Read(net.Out, &net.PredictionValue)

	net.Out = grg.Must(grg.Reshape(net.Out, answerShape))
	onc := grg.Must(grg.Neg(grg.Must(grg.Neg(net.Out))))
	net.TrueNode = onc
	net.Cost = grg.Must(grg.Neg(grg.Must(grg.Mean(grg.Must(gorgonia.Square(grg.Must(gorgonia.Sub(net.LearnNode, net.Out))))))))
	//net.Cost = grg.Must(grg.Neg(grg.Must(grg.Sum((grg.Must(grg.HadamardProd(net.Out, net.LearnNode)))))))
	grg.Read(net.Cost, &net.CostValue)
	_, err := grg.Grad(net.Cost, net.Weights...)
	prog, locMap, _ := grg.Compile(net.Graph)
	net.Machine = grg.NewTapeMachine(net.Graph, grg.WithPrecompiled(prog, locMap), grg.BindDualValues(net.Weights...))
	net.Solver = gorgonia.NewRMSPropSolver()
	if err != nil {
		fmt.Println(err)
	}
	return net
}
func GenerateNetPretrained(lps []Params, inputShape, answerShape tensor.Shape, path string) Net {
	var net Net
	net.Epsilon = 0.000001
	net.BatchSize = 1
	net.Graph = grg.NewGraph()
	net.Input = grg.NewTensor(net.Graph, tensor.Float32, inputShape.Dims(), grg.WithShape(inputShape...), grg.WithName("input"))
	net.Out = net.Input
	net.IsPretrained = true
	net.LoadWeights(path)
	for i := 0; i < len(lps); i++ {
		lps[i].AppendLayer(&net)
	}
	return net
}

//AddBiases - adds biases to the last node (fulcon only!)
func (net *Net) AddBiases() {
	weights := grg.NewTensor(net.Graph, tensor.Float64, net.Out.Shape().Dims(), grg.WithShape(net.Out.Shape()...), grg.WithInit(grg.Zeroes()), grg.WithName("biases"+fmt.Sprint(len(net.Weights))))
	net.Weights = append(net.Weights, weights)
	net.Out = grg.Must(grg.Add(net.Out, weights))
}
func (net *Net) AddBiasesPretrained(bweights []float32) {
	lb := tensor.New(tensor.WithShape(len(bweights), 1), tensor.WithBacking(bweights))
	vec := grg.NewTensor(net.Graph, tensor.Float32, 2, grg.WithShape(len(bweights), 1), grg.WithValue(lb), grg.WithName("biases"+fmt.Sprint(len(net.Weights))))
	cv := grg.NewTensor(net.Graph, tensor.Float32, 2, grg.WithShape(1, net.Out.Shape().TotalSize()/len(bweights)), grg.WithInit(grg.Ones()))
	weights := grg.Must(grg.Reshape(grg.Must(grg.Mul(vec, cv)), net.Out.Shape()))
	net.Weights = append(net.Weights, vec)
	net.Out = grg.Must(grg.Add(net.Out, weights))
}

//ApplyActivation - function, that uses activation by name to the last layer
func (net *Net) ApplyActivation(activationName string, alpha ...float64) error {
	switch activationName {
	case "sigm":
		net.Out = grg.Must(grg.Sigmoid(net.Out))
		return nil
	case "tanh":
		net.Out = grg.Must(grg.Tanh(net.Out))
		return nil
	case "softplus":
		net.Out = grg.Must(grg.Softplus(net.Out))
		return nil
	case "leaky":
		net.Out = grg.Must(grg.LeakyRelu(net.Out, alpha[0]))
		return nil
	case "softmax":
		net.Out = grg.Must(grg.SoftMax(net.Out))
		return nil
	case "rect":
		net.Out = grg.Must(grg.Rectify(net.Out))
		return nil
	case "off":
		return nil
	default:
		return errors.New("Activation name is incorrect: " + activationName)
	}
}

//ApplyMSE - function, that calculates MSE into Cost node
func (net *Net) ApplyMSE() {
	losses := grg.Must(grg.Sub(net.LearnNode, net.Out))
	losses = grg.Must(grg.Square(losses))
	//There are no information about shape changes in documentation! Be carefull!!!!
	losses = grg.Must(grg.Reshape(losses, tensor.Shape{1, net.Out.Shape().TotalSize()}))
	net.Cost = grg.Must(grg.Mean(losses, 1))
}

//ApplyDropout - function, that uses dropout on the last layer with prob of alpha    COULD BE AN ACTIVFATION FUNC IN FUTURE!
func (net *Net) ApplyDropout(alpha float64) {
	net.Out = grg.Must(grg.Dropout(net.Out, alpha))
}
func (net *Net) LoadWeights(path string) {

	fp, err := os.Open(path)

	if err != nil {
		panic(err)
	}

	defer fp.Close()

	summary := []byte{}
	data := make([]byte, 4096)
	for {
		data = data[:cap(data)]
		n, err := fp.Read(data)
		if err != nil {
			if err == io.EOF {
				break
			}
			fmt.Println(err)
			return
		}
		data = data[:n]
		//fmt.Println(len(data))

		summary = append(summary, data...)
		// break
	}

	weights := 0
	for i := 0; i < len(summary); i += 4 {
		tempSlice := summary[i : i+4]
		tempFloat32 := Float32frombytes(tempSlice)
		net.UnparsedWeights = append(net.UnparsedWeights, tempFloat32)
		weights++
	}
	net.LastWeight = 5
}

func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

func Float32bytes(float float32) []byte {
	bits := math.Float32bits(float)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint32(bytes, bits)
	return bytes
}
