package proc

import (
	"errors"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"reflect"
	"sort"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

const (
	blockSize     = 32
	boxesPerCell  = 5
	numClasses    = 22
	iouTreshold   = 0.2
	scoreTreshold = 0.6
	envConfPrefix = "yolo"
)

var (
	hSize, wSize = 416, 416

	gridH       = 13
	gridW       = 13
	ImgRescaled *image.NRGBA
	model       = "./models/motherfucker_coco.onnx"
	imgF        = "./data/testnumyolo.jpg"
	outputF     = "./prediction.png"
	classes     = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"}
	//anchors     = []float64{1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52}
	anchors     = []float64{0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828}
	scaleFactor = float32(1) // The scale factor to resize the image to hSize*wSize
	outWidth    = 416
	outHight    = 416
)

type Bbox struct {
	conf  float64
	rect  image.Rectangle
	class string
	score float64
}

func (b Bbox) GetClass() string {
	return b.class
}
func ResizeNet(iw, ih, gw, gh int) {
	hSize = ih
	wSize = iw
	gridH = gh
	gridW = gw
}
func SetOutShape(rect image.Rectangle) {
	outWidth = rect.Dx()
	outHight = rect.Dy()
}
func GetInput(img image.Image) tensor.Tensor {
	//img = resize.Resize(hSize, wSize, img, resize.Bicubic)

	inputT := tensor.New(tensor.WithShape(1, 3, hSize, wSize), tensor.Of(tensor.Float32))
	err := imageToBCHW(img, inputT)
	if err != nil {
		log.Fatal(err)
	}
	return inputT
}

func GetInputFromFile(path string) tensor.Tensor {
	file, err := os.Open(path)
	inp := make([]float32, 0)
	if err != nil {
		panic(err)
	}
	for {
		var flt float32
		var n int
		n, err = fmt.Fscanln(file, &flt)
		if n == 0 || err != nil {
			break
		}
		inp = append(inp, flt)
	}
	return tensor.New(tensor.WithShape(1, 3, hSize, wSize), tensor.Of(tensor.Float32), tensor.WithBacking(inp))
}

func MSEdarknet(goout tensor.Tensor, path string) float64 {
	file, err := os.Open(path)
	godata := goout.Data().([]float32)
	sum := float32(0.0)
	if err != nil {
		panic(err)
	}
	for i := 0; ; i++ {
		var drk float32
		var n int
		n, err = fmt.Fscanln(file, &drk)
		if n == 0 || err != nil {
			break
		}
		sum += (drk - godata[i]) * (drk - godata[i])
	}

	return math.Sqrt(float64(sum / 22816.))
}

func ProcessOutput(t tensor.Tensor) []Bbox {
	bb := make([]Bbox, 0)
	dense := t.(*tensor.Dense)
	must(dense.Reshape((numClasses+5)*boxesPerCell, gridH, gridW))
	data, err := native.Tensor3F32(dense)
	if err != nil {
		log.Fatal(err)
	}

	for cx := 0; cx < gridW; cx++ {
		for cy := 0; cy < gridH; cy++ {
			for b := 0; b < boxesPerCell; b++ {
				class := make([]float64, numClasses)
				channel := b * (numClasses + 5)
				tx := data[channel][cx][cy]
				ty := data[channel+1][cx][cy]
				tw := data[channel+2][cx][cy]
				th := data[channel+3][cx][cy]
				tc := data[channel+4][cx][cy]
				for cl := 0; cl < numClasses; cl++ {
					class[cl] = float64(data[channel+5+cl][cx][cy])
				}
				finclass := softmax(class)
				maxprob, maxi := maxIn(finclass)

				rw := float64(outWidth) / float64(wSize)
				rh := float64(outHight) / float64(hSize)
				//x := int((float64(cx) + sigmoid(float32(math.Pow(float64(tx), 2)))) * blockSize)
				//y := int((float64(cy) + sigmoid(float32(math.Pow(float64(ty), 2)))) * blockSize)
				x := int((float64(cy) + sigmoid(tx)) * blockSize * rw)
				y := int((float64(cx) + sigmoid(ty)) * blockSize * rh)
				// The size of the bounding box, tw and th, is predicted relative to
				// the size of an "anchor" box. Here we also transform the width and
				// height into the original 416x416 image space.
				w := int(exp(tw) * anchors[2*b] * blockSize * rw)
				h := int(exp(th) * anchors[2*b+1] * blockSize * rh)

				sigconf := sigmoid(tc)
				// /sort.Float64s(finclass)
				finconf := sigconf * maxprob
				if finconf > scoreTreshold {
					box := Bbox{
						conf:  sigconf,
						rect:  Rectify(x, y, h, w),
						class: classes[maxi],
						score: maxprob,
					}
					bb = append(bb, box)
				}
			}
		}
	}
	bb = nonMaxSupr(bb)
	sort.Sort(xsort(bb))
	return bb
}
func Image2Float32(img image.Image) ([]float32, error) {
	red := []float32{}
	green := []float32{}
	blue := []float32{}
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := img.At(y, x).RGBA()
			rpix, gpix, bpix := float32(r>>8)/float32(255.0), float32(g>>8)/float32(255.0), float32(b>>8)/float32(255.0)
			red = append(red, rpix)
			green = append(green, gpix)
			blue = append(blue, bpix)
		}
	}
	ans := []float32{}
	ans = append(ans, red...)
	ans = append(ans, green...)
	ans = append(ans, blue...)
	return ans, nil
}
func imageToBCHW(img image.Image, dst tensor.Tensor) error {
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("cannot decode image into a non pointer or a nil receiver")
	}
	// check if tensor is compatible with BCHW (4 dimensions)
	if len(dst.Shape()) != 4 {
		return fmt.Errorf("Expected a 4 dimension tensor, but receiver has only %v", len(dst.Shape()))
	}
	// Check the batch size
	if dst.Shape()[0] != 1 {
		return errors.New("only batch size of one is supported")
	}
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	if dst.Shape()[2] != h || dst.Shape()[3] != w {
		return fmt.Errorf("cannot fit image into tensor; image is %v*%v but tensor is %v*%v", h, w, dst.Shape()[2], dst.Shape()[3])
	}
	switch dst.Dtype() {
	case tensor.Float32:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				r, g, b, a := img.At(x, y).RGBA()

				if a != 65535 {
					return errors.New("transparency not handled")
				}
				err := dst.SetAt(float32(r>>8)/255.0, 0, 0, y, x)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(g>>8)/255.0, 0, 1, y, x)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(b>>8)/255.0, 0, 2, y, x)
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("%v not handled yet", dst.Dtype())
	}
	return nil
}

func sigmoid(sum float32) float64 {
	return float64(1.0 / (1.0 + math.Exp(float64(-sum))))
}

func softmax(a []float64) []float64 {
	sum := 0.0
	output := make([]float64, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = math.Exp(a[i])
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

func exp(val float32) float64 {
	return math.Exp(float64(val))
}

func maxIn(cl []float64) (float64, int) {
	max, maxi := -1., -1
	for i := range cl {
		if max < cl[i] {
			max = cl[i]
			maxi = i
		}
	}
	return max, maxi
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

type sorter []Bbox

func (b sorter) Len() int           { return len(b) }
func (b sorter) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b sorter) Less(i, j int) bool { return b[i].conf < b[j].conf }

type xsort []Bbox

func (b xsort) Len() int           { return len(b) }
func (b xsort) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b xsort) Less(i, j int) bool { return b[i].rect.Min.X < b[j].rect.Min.X }

func nonMaxSupr(b []Bbox) []Bbox {
	//sorts boxes by confidence
	sort.Sort(sorter(b))
	nms := make([]Bbox, 0)
	if len(b) == 0 {
		return nms
	}
	nms = append(nms, b[0])

	for i := 1; i < len(b); i++ {
		tocheck, del := len(nms), false
		for j := 0; j < tocheck; j++ {
			currIOU := iou(b[i].rect, nms[j].rect)
			if currIOU > iouTreshold && b[i].class == nms[j].class {
				del = true
				break
			}
		}
		if !del {
			nms = append(nms, b[i])
		}
	}
	return nms
}

func iou(r1, r2 image.Rectangle) float64 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float64(interArea) / float64(r1Area+r2Area-interArea)
}
