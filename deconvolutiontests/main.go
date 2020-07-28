package main

import (
	"fmt"

	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	ein := []float64{1.0, 2.0, 3.0, 4.0} //, 5.0, 6.0, 7.0, 8.0, 9.0}
	ek := []float64{1.0, 1.0, 1.0, 1.0}
	et := []float64{1.0}
	inTen := tensor.New(tensor.WithShape(1, 1, 2, 2), tensor.Of(tensor.Float64), tensor.WithBacking(ein))
	kerTen := tensor.New(tensor.WithShape(1, 1, 2, 2), tensor.Of(tensor.Float64), tensor.WithBacking(ek))
	swapTen := tensor.New(tensor.WithShape(1, 1, 1, 1), tensor.Of(tensor.Float64), tensor.WithBacking(et))
	g := grg.NewGraph()
	input := grg.NewTensor(g, grg.Float64, 4, grg.WithShape(1, 1, 2, 2), grg.WithName("input"))
	kernel := grg.NewTensor(g, grg.Float64, 4, grg.WithShape(1, 1, 2, 2), grg.WithName("kernel"))
	transposer := grg.NewTensor(g, grg.Float64, 4, grg.WithShape(1, 1, 1, 1), grg.WithName("transposer"))
	deconv := grg.Must(grg.Conv2d(transposer, input, tensor.Shape{2, 2}, []int{1, 1}, []int{1, 1}, []int{1, 1})) // Padding : input -1!
	out := grg.Must(grg.Conv2d(kernel, deconv, tensor.Shape{2, 2}, []int{2, 2}, []int{1, 1}, []int{2, 2}))       // Padding : input*2-2!

	vm := grg.NewTapeMachine(g)
	err := grg.Let(input, inTen)
	grg.Let(kernel, kerTen)
	grg.Let(transposer, swapTen)
	vm.RunAll()
	fmt.Println(out.Value(), err)
	vm.Close()
}
