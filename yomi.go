// Package yomi reads text off screen frames: flat colour and anti-aliased
// glyphs, as app windows draw them. It finds the lines of a frame, reads each
// with a small convolutional recognizer, and returns words with their boxes
// and a confidence.
//
// It is pure Go with no cgo, no runtime model downloads and no network. A
// model is a package of its own under models/, with its weights embedded, so a
// program links only the scripts it imports:
//
//	m, err := latin.Model()
//	e := yomi.New(m)
//	lines, err := e.Read(ctx, frame, 2)
//
// Frames and the text read from them are screen content. Nothing here logs
// them, writes them anywhere, or keeps them past the call.
package yomi

import (
	"context"
	"image"
	"io/fs"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/y3owk1n/yomi/internal/linefind"
	"github.com/y3owk1n/yomi/internal/nn"
	"github.com/y3owk1n/yomi/internal/prep"
)

// Model is a loaded recognizer for one set of scripts.
type Model struct{ net *nn.Model }

// LoadModel reads a model (model.json and weights.bin) from fsys. Model
// packages call it on their embedded files.
func LoadModel(fsys fs.FS) (*Model, error) {
	net, err := nn.Load(fsys)
	if err != nil {
		return nil, err
	}
	return &Model{net}, nil
}

// Word is one word read off a frame. Box is in logical pixels (frame pixels
// divided by the scale Read was given). Confidence is 0 to 1: the lowest
// probability the recognizer gave any of the word's characters.
type Word struct {
	Text       string
	Box        image.Rectangle
	Confidence float32
}

// Line is a line of words, left to right, with the box it was read from.
type Line struct {
	Box   image.Rectangle
	Words []Word
}

// Engine reads frames with one model. It is safe for concurrent use.
type Engine struct {
	model   *Model
	workers int
	pool    sync.Pool
}

// Option configures an Engine.
type Option func(*Engine)

// WithWorkers caps how many lines are read at once. The default is one per
// CPU.
func WithWorkers(n int) Option {
	return func(e *Engine) { e.workers = max(1, n) }
}

// New returns an Engine that reads with m.
func New(m *Model, opts ...Option) *Engine {
	e := &Engine{model: m, workers: runtime.NumCPU()}
	e.pool.New = func() any { return &nn.Scratch{} }
	for _, o := range opts {
		o(e)
	}
	return e
}

// Read finds the lines of frame and reads them. scale is frame pixels per
// logical pixel (1, or 2 on a HiDPI display). Lines come back top to bottom.
// Canceling ctx stops lines not yet started, and Read returns ctx's error.
func (e *Engine) Read(ctx context.Context, frame *image.RGBA, scale int) ([]Line, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	scale = max(1, scale)
	boxes := linefind.Find(frame, float64(scale), linefind.DefaultParams())
	lines := make([]Line, len(boxes))
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	// Longest lines first, so no worker is left with a long one at the end.
	sort.Slice(order, func(a, b int) bool { return boxes[order[a]].Dx() > boxes[order[b]].Dx() })
	next := make(chan int, len(order))
	for _, i := range order {
		next <- i
	}
	close(next)
	var wg sync.WaitGroup
	for range max(1, min(e.workers, len(boxes))) {
		wg.Go(func() {
			for i := range next {
				if ctx.Err() != nil {
					return
				}
				lines[i] = Line{Box: boxes[i], Words: e.readBox(frame, boxes[i], scale)}
			}
		})
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(lines, func(a, b int) bool {
		la, lb := lines[a].Box, lines[b].Box
		if la.Min.Y != lb.Min.Y {
			return la.Min.Y < lb.Min.Y
		}
		return la.Min.X < lb.Min.X
	})
	out := lines[:0]
	for _, l := range lines {
		if len(l.Words) > 0 {
			out = append(out, l)
		}
	}
	return out, nil
}

// ReadLine reads box, a region of frame in logical pixels holding one line of
// text, without line finding: a second look at a known line.
func (e *Engine) ReadLine(frame *image.RGBA, box image.Rectangle, scale int) []Word {
	return e.readBox(frame, box, max(1, scale))
}

func (e *Engine) readBox(frame *image.RGBA, box image.Rectangle, scale int) []Word {
	net := e.model.net
	r := image.Rect(box.Min.X*scale, box.Min.Y*scale, box.Max.X*scale, box.Max.Y*scale).
		Add(frame.Rect.Min).Intersect(frame.Rect)
	x, w := prep.Prepare(frame, r, net.Height)
	if x == nil || w < net.Stride {
		return nil
	}
	s := e.pool.Get().(*nn.Scratch)
	steps := net.Steps(x, w, s)
	e.pool.Put(s)
	words := decode(steps, []rune(net.Alphabet), net.Stride, r.Sub(frame.Rect.Min), w, scale)
	settleScripts(words)
	return words
}

// decode is greedy CTC: the best class per step, repeats and blanks dropped,
// spaces splitting words. A word's box spans its first to last step.
func decode(steps []nn.Step, alpha []rune, stride int, r image.Rectangle, w, scale int) []Word {
	sx := float64(r.Dx()) / float64(w)
	var (
		words  []Word
		cur    []rune
		conf   float32
		t0, t1 int
		prev   int
	)
	flush := func() {
		if len(cur) == 0 {
			return
		}
		x0 := float64(r.Min.X) + float64(t0*stride)*sx
		x1 := float64(r.Min.X) + float64((t1+1)*stride)*sx
		words = append(words, Word{
			Text:       string(cur),
			Box:        image.Rect(int(x0)/scale, r.Min.Y/scale, int(math.Ceil(x1))/scale, r.Max.Y/scale),
			Confidence: conf,
		})
		cur = nil
	}
	for j, st := range steps {
		best := st.Class
		if best != prev && best != 0 {
			ch := alpha[best-1]
			if ch == ' ' {
				flush()
			} else {
				p := st.Prob
				if len(cur) == 0 {
					t0, conf = j, p
				}
				conf = min(conf, p)
				cur = append(cur, ch)
				t1 = j
			}
		}
		prev = best
	}
	flush()
	return words
}
