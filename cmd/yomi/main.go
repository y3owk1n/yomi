// Command yomi reads the text in PNG screenshots.
//
//	yomi read [-model NAME | -model-dir DIR] [-scale 2] FILE.png...
//	    prints each file's lines, one per line of output
//	yomi bench [-model NAME | -model-dir DIR] [-reps 3] DIR
//	    reads DIR/pages.txt's NAME@1x.png and NAME@2x.png and writes one JSON
//	    line per frame: {"page","scale","first_ms","median_ms",
//	    "words":[[text,x,y,w,h,confidence]...]}, boxes in logical pixels
//
// NAME is a recognizer (latin, western, cjk, rtl, sea, caucasian), or auto:
// every recognizer, with the router choosing each line's.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/y3owk1n/yomi"
	"github.com/y3owk1n/yomi/models/caucasian"
	"github.com/y3owk1n/yomi/models/cjk"
	"github.com/y3owk1n/yomi/models/latin"
	"github.com/y3owk1n/yomi/models/router"
	"github.com/y3owk1n/yomi/models/rtl"
	"github.com/y3owk1n/yomi/models/sea"
	"github.com/y3owk1n/yomi/models/western"
)

var models = map[string]func() (*yomi.Model, error){
	"latin": latin.Model, "western": western.Model, "cjk": cjk.Model,
	"rtl": rtl.Model, "sea": sea.Model, "caucasian": caucasian.Model,
}

// auto returns an engine with every recognizer and the router, Western
// reading any family it lacks.
func auto() (*yomi.Engine, error) {
	opts := []yomi.Option{}
	for _, load := range []func() (*yomi.Model, error){cjk.Model, rtl.Model, sea.Model, caucasian.Model} {
		m, err := load()
		if err != nil {
			return nil, err
		}
		opts = append(opts, yomi.WithModel(m))
	}
	r, err := router.Model()
	if err != nil {
		return nil, err
	}
	m, err := western.Model()
	if err != nil {
		return nil, err
	}
	return yomi.New(m, append(opts, yomi.WithRouter(r))...), nil
}

func load(path string) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	src, err := png.Decode(f)
	_ = f.Close()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	img := image.NewRGBA(src.Bounds())
	draw.Draw(img, img.Rect, src, image.Point{}, draw.Src)
	return img, nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "yomi:", err)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		fail(fmt.Errorf("usage: yomi read|bench"))
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	name := fs.String("model", "western", "recognizer: latin, western, cjk, rtl, sea, caucasian, or auto for all of them")
	dir := fs.String("model-dir", "", "load the recognizer from this directory (model.json, weights.bin) instead")
	scale := fs.Int("scale", 1, "frame pixels per logical pixel (read)")
	reps := fs.Int("reps", 3, "timed reads after the first (bench)")
	_ = fs.Parse(os.Args[2:])
	e, err := engine(*name, *dir)
	if err != nil {
		fail(err)
	}
	switch os.Args[1] {
	case "read":
		for _, path := range fs.Args() {
			img, err := load(path)
			if err != nil {
				fail(err)
			}
			lines, err := e.Read(context.Background(), img, *scale)
			if err != nil {
				fail(err)
			}
			for _, l := range lines {
				words := make([]string, len(l.Words))
				for i, w := range l.Words {
					words[i] = w.Text
				}
				fmt.Println(strings.Join(words, " "))
			}
		}
	case "bench":
		if err := bench(e, fs.Arg(0), *reps); err != nil {
			fail(err)
		}
	default:
		fail(fmt.Errorf("unknown command %q", os.Args[1]))
	}
}

func engine(name, dir string) (*yomi.Engine, error) {
	if dir != "" {
		m, err := yomi.LoadModel(os.DirFS(dir))
		if err != nil {
			return nil, err
		}
		return yomi.New(m), nil
	}
	if name == "auto" {
		return auto()
	}
	model, ok := models[name]
	if !ok {
		return nil, fmt.Errorf("unknown model %q", name)
	}
	m, err := model()
	if err != nil {
		return nil, err
	}
	return yomi.New(m), nil
}

func bench(e *yomi.Engine, dir string, reps int) error {
	raw, err := os.ReadFile(dir + "/pages.txt")
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	ctx := context.Background()
	for _, page := range strings.Fields(string(raw)) {
		for _, scale := range []int{1, 2} {
			img, err := load(fmt.Sprintf("%s/%s@%dx.png", dir, page, scale))
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				return err
			}
			t := time.Now()
			lines, err := e.Read(ctx, img, scale)
			first := time.Since(t)
			if err != nil {
				return err
			}
			times := []time.Duration{first}
			for range reps {
				t := time.Now()
				if _, err := e.Read(ctx, img, scale); err != nil {
					return err
				}
				times = append(times, time.Since(t))
			}
			slices.Sort(times[1:])
			med := times[len(times)/2]
			var words [][]any
			for _, l := range lines {
				for _, w := range l.Words {
					b := w.Box
					words = append(words, []any{w.Text, b.Min.X, b.Min.Y, b.Dx(), b.Dy(), w.Confidence})
				}
			}
			if err := enc.Encode(map[string]any{
				"page": page, "scale": scale, "words": words,
				"first_ms":  float64(first.Microseconds()) / 1000,
				"median_ms": float64(med.Microseconds()) / 1000,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}
