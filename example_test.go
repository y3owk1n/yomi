package yomi_test

import (
	"context"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"log"
	"os"

	"github.com/y3owk1n/yomi"
	"github.com/y3owk1n/yomi/models/caucasian"
	"github.com/y3owk1n/yomi/models/cjk"
	"github.com/y3owk1n/yomi/models/indic"
	"github.com/y3owk1n/yomi/models/router"
	"github.com/y3owk1n/yomi/models/rtl"
	"github.com/y3owk1n/yomi/models/sea"
	"github.com/y3owk1n/yomi/models/western"
)

// A screenshot of a Retina display, read with the Western recognizer.
func ExampleEngine_Read() {
	f, err := os.Open("shot.png")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		log.Fatal(err)
	}
	frame := image.NewRGBA(src.Bounds())
	draw.Draw(frame, frame.Rect, src, src.Bounds().Min, draw.Src)

	m, err := western.Model()
	if err != nil {
		log.Fatal(err)
	}
	e := yomi.New(m)
	lines, err := e.Read(context.Background(), frame, 2)
	if err != nil {
		log.Fatal(err)
	}
	for _, l := range lines {
		for _, w := range l.Words {
			fmt.Println(w.Text, w.Box, w.Confidence)
		}
	}
}

// An engine for text in any script: every recognizer, and the router to
// pick one per line.
func ExampleWithRouter() {
	base, err := western.Model()
	if err != nil {
		log.Fatal(err)
	}
	r, err := router.Model()
	if err != nil {
		log.Fatal(err)
	}
	opts := []yomi.Option{yomi.WithRouter(r)}
	for _, load := range []func() (*yomi.Model, error){cjk.Model, rtl.Model, indic.Model, sea.Model, caucasian.Model} {
		m, err := load()
		if err != nil {
			log.Fatal(err)
		}
		opts = append(opts, yomi.WithModel(m))
	}
	e := yomi.New(base, opts...)
	_ = e
}

// A word's box in physical screen pixels, for a frame captured at origin on a
// display at scale. Boxes are in logical pixels from the frame's top-left, so
// a system whose coordinates are logical, such as macOS points, adds the
// origin without scaling.
func ExampleWord() {
	var (
		w      yomi.Word // from Engine.Read
		origin = image.Pt(1440, 0)
		scale  = 1.5
	)
	at := func(v int) int { return int(float64(v) * scale) }
	screen := image.Rect(at(w.Box.Min.X), at(w.Box.Min.Y), at(w.Box.Max.X), at(w.Box.Max.Y)).Add(origin)
	fmt.Println(screen)
}
