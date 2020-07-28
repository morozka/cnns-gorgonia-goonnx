package proc

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func DrawBoxesOn(dets []Bbox, img image.Image) {
	detim := image.NewNRGBA(img.Bounds())

	out, err := os.Create("./dections.png")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	draw.Draw(detim, detim.Bounds(), img, image.ZP, draw.Src)
	for _, b := range dets {
		drawRectangle(detim, b.rect, fmt.Sprintf("%v %2.2f%%", b.class, b.score))
	}

	if err := png.Encode(out, detim); err != nil {
		log.Fatal(err)
	}
}

func drawRectangle(img *image.NRGBA, r image.Rectangle, label string) {
	col := color.RGBA{225, 50, 100, 255} // Red

	// HLine draws a horizontal line
	hLine := func(x1, y, x2 int) {
		for ; x1 <= x2; x1++ {
			img.Set(x1, y, col)
		}
	}

	// VLine draws a veritcal line
	vLine := func(x, y1, y2 int) {
		for ; y1 <= y2; y1++ {
			img.Set(x, y1, col)
		}
	}

	minX := int(float32(r.Min.X) * scaleFactor)
	maxX := int(float32(r.Max.X) * scaleFactor)
	minY := int(float32(r.Min.Y) * scaleFactor)
	maxY := int(float32(r.Max.Y) * scaleFactor)
	// Rect draws a rectangle utilizing HLine() and VLine()
	rect := func(r image.Rectangle) {
		hLine(minX, maxY, maxX)
		hLine(minX, maxY, maxX)
		hLine(minX, minY, maxX)
		vLine(maxX, minY, maxY)
		vLine(minX, minY, maxY)
	}
	addLabel(img, minX+5, minY+15, label)
	rect(r)
}

func addLabel(img *image.NRGBA, x, y int, label string) {
	col := color.NRGBA{100, 255, 100, 255}
	point := fixed.Point26_6{
		X: fixed.Int26_6(x * 64),
		Y: fixed.Int26_6(y * 64),
	}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

func Rectify(x, y, h, w int) image.Rectangle {
	return image.Rect(max(x-w/2, 0), max(y-h/2, 0), min(x+w/2+1, outWidth), min(y+h/2+1, outHight))
}
