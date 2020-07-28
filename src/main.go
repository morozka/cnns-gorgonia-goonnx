package main

import (
	"fmt"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"time"

	"github.com/nfnt/resize"

	"github.com/cnns-gorgonia-goonnx/src/nntypes"
	"github.com/cnns-gorgonia-goonnx/src/proc"
	grg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	imgpath = "../data/416.jpg"

	gridWidth  = 13
	gridHight  = 13
	blockSize  = 32
	inputWidth = gridWidth * blockSize
	inputHight = gridHight * blockSize
)

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func main() {
	var err error

	var lps []nntypes.Params
	fmt.Printf("Generating net...\n")
	nettime := time.Now()
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{16, 3, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{2, 2},
		Padding:    []int{0, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{32, 16, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{2, 2},
		Padding:    []int{0, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{64, 32, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{2, 2},
		Padding:    []int{0, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{128, 64, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{2, 2},
		Padding:    []int{0, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{256, 128, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{2, 2},
		Padding:    []int{0, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{512, 256, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.MaxPoolParams{
		KernelSize: tensor.Shape{2, 2},
		Stride:     []int{1, 1},
		Padding:    []int{1, 0, 1, 0},
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{1024, 512, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})

	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{512, 1024, 3, 3},
		KernelSize:   tensor.Shape{3, 3},
		Stride:       []int{1, 1},
		Padding:      []int{1, 1},
		Dilation:     []int{1, 1},
		Activation:   "leaky",
		BatchNorm:    true,
		Alpha:        0.1,
	})
	lps = append(lps, nntypes.ConvolutionParams{
		KernelsShape: tensor.Shape{135, 512, 1, 1},
		KernelSize:   tensor.Shape{1, 1},
		Stride:       []int{1, 1},
		Padding:      []int{0, 0},
		Dilation:     []int{1, 1},
		Activation:   "off",
		Biased:       true,
	})

	wnet := nntypes.GenerateNetPretrained(lps, tensor.Shape{1, 3, inputWidth, inputHight}, tensor.Shape{1, 135, gridWidth, gridHight}, "./models/yolo-obj_196000.weights")
	prog, locMap, _ := grg.Compile(wnet.Graph)
	vm := grg.NewTapeMachine(wnet.Graph, grg.WithPrecompiled(prog, locMap))

	fmt.Printf("Net generated in %0.2f ms\n", float64(time.Since(nettime).Nanoseconds())/1000000.)
	nettime = time.Now()
	fmt.Printf("Processing image...\n")
	ifile, err := os.Open(imgpath)
	if err != nil {
		panic(err)
	}
	imgIn, err := jpeg.Decode(ifile)
	if err != nil {
		panic(err)
	}
	proc.ResizeNet(inputWidth, inputHight, gridWidth, gridHight)
	proc.SetOutShape(imgIn.Bounds())
	imgR := resize.Resize(inputWidth, inputHight, imgIn, resize.Bicubic)
	imgData, _ := proc.Image2Float32(imgR)
	imtest := tensor.New(tensor.WithShape(1, 3, inputHight, inputWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgData))

	//imtest := proc.GetInput(imgR)
	fmt.Printf("Image processed in %0.2f ms\n", float64(time.Since(nettime).Nanoseconds())/1000000.)
	//imtest := proc.GetInputFromFile("./input")

	nettime = time.Now()
	fmt.Printf("Forwarding net...\n")
	grg.Let(wnet.Input, imtest)

	if err = vm.RunAll(); err != nil {
		log.Fatalf("Failed at epoch  %d: %v", 1, err)
	}
	fmt.Printf("Net forwarrded in %0.2f ms\n", float64(time.Since(nettime).Nanoseconds())/1000000.)
	nettime = time.Now()
	fmt.Printf("Processing output...\n")
	tensors := make([]tensor.Tensor, 1)
	tensors[0] = wnet.Out.Value().(tensor.Tensor)

	bb := proc.ProcessOutput(tensors[0])
	fmt.Printf("Processed in %0.2f ms\n", float64(time.Since(nettime).Nanoseconds())/1000000.)
	fmt.Println("Car numer:")
	for i := 0; i < len(bb); i++ {
		fmt.Print(bb[i].GetClass())
	}
	fmt.Println("")
	proc.DrawBoxesOn(bb, imgIn)
	vm.Reset()
	vm.Close()
}
