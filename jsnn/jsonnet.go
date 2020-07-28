package jsnn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	grg "gorgonia.org/gorgonia"
)

type lay struct {
	Name   string          `json:"name"`
	Type   string          `json:"type"`
	Params json.RawMessage `json:"params"`
}

type LearnParams struct {
	Learnrate float64 `json:"learnrate"`
	Batch     int     `json:"batch"`
	Solver    string  `json:"solver"`
}

type netData struct {
	Layers []lay       `json:"layers"`
	Insize []int       `json:"insize"`
	Params LearnParams `json:"params"`
	//Mode   string   `json:"mode"` //@TODO: train or work should be implemented
	Net []string `json:"net"`
}

type Net struct {
	Graph     *grg.ExprGraph
	W         []*grg.Node
	L         []*grg.Node
	Solver    grg.Solver
	Machine   grg.VM
	Input     *grg.Node
	Out       *grg.Node
	InsideOut grg.Node
	LearnVal  *grg.Node
	Cost      *grg.Node
	Grad      grg.Nodes
	OutReader grg.Value
}

func createLayerMap(layer lay) param {
	switch layer.Type {
	case "conv":
		var Param convParams
		json.Unmarshal(layer.Params, &Param)
		return Param
	case "pool":
		var Param poolParams
		json.Unmarshal(layer.Params, &Param)
		return Param
	case "fulc":
		var Param fullConParams
		json.Unmarshal(layer.Params, &Param)
		return Param
	/*case "drop":
	var Param dropoutParams
	json.Unmarshal(layer.Params, &Param)
	return Param*/
	default:
		panic(fmt.Sprintln("There is no such type of layer:", layer.Type))
	}
}

var wi int
var li int

//NetFromJSON - returns WholeNet struct with net, described in json
/*
	@TODO: write docs how to write JSONnet file
*/
func NetFromJSON(jsonpath string) (Net, grg.VM, grg.Solver) {
	JSONnet, err := os.Open(jsonpath)
	if err != nil {
		panic(err)
	}
	defer JSONnet.Close()
	wi = 0
	byteValue, _ := ioutil.ReadAll(JSONnet)
	var jnet netData
	json.Unmarshal(byteValue, &jnet)
	//parses predefined layers
	laymap := make(map[string]param)

	for _, v := range jnet.Layers {
		laymap[v.Name] = createLayerMap(v)
	}
	wnet := Net{
		Graph: grg.NewGraph(),
	}
	wnet.Input = grg.NewTensor(wnet.Graph, tensor.Float64, len(jnet.Insize), grg.WithShape(jnet.Insize...), grg.WithName("Input"))
	input := wnet.Input
	for _, userlay := range jnet.Net {
		l, ok := laymap[userlay]
		if !ok {
			panic(fmt.Sprint("Such layer type (", userlay, ") doesn't exist! Check your json."))
		}
		out := l.CreateLayer(&wnet, input)
		wnet.L = append(wnet.L, out) //creating output node for curent layer
		wnet.Out = out
		input = wnet.L[len(wnet.L)-1]
		log.Println(userlay, "created")
	}

	//wnet.Out = &(wnet.L[len(wnet.L)-1])
	wnet.InsideOut = *wnet.Out

	wnet.LearnVal = grg.NewTensor(wnet.Graph, tensor.Float64, 2, grg.WithShape(1, 1), grg.WithName("Right answer"))
	losses := grg.Must(grg.Sub(wnet.LearnVal, wnet.Out))
	losses = grg.Must(grg.Square(losses))
	//losses = grg.Must(grg.Neg(losses))

	wnet.Cost = grg.Must(grg.Sum(grg.Must(grg.Neg(grg.Must(grg.Square(grg.Must(grg.Sub(wnet.LearnVal, wnet.Out))))))))
	fmt.Printf("Net from \"%v\" created!\n", jsonpath)
	fmt.Println(wnet.Cost)
	if wnet.Grad, err = grg.Grad(wnet.Cost, wnet.W...); err != nil {
		panic(err)
	}

	//wnet.LearnVal = yval

	prog, locMap, _ := grg.Compile(wnet.Graph)
	// fmt.Println(prog, locMap)
	vm := gorgonia.NewTapeMachine(wnet.Graph, grg.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(wnet.W...))

	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.01))

	ioutil.WriteFile("./graph.dot", []byte(wnet.Graph.ToDot()), 0644)
	return wnet, vm, solver
}
