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
	"slices"
	"sort"
	"sync"
	"unicode"

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
// divided by the scale Read was given). Text is in logical order, so an
// Arabic or Hebrew word reads as written, not as drawn. Chars holds one box
// per character of Text, in Text's order. Each spans from halfway to the
// character drawn before it to halfway to the one drawn after, so the boxes
// tile Box. Confidence is 0 to 1, the lowest probability the recognizer gave
// any of the word's characters.
type Word struct {
	Text       string
	Box        image.Rectangle
	Chars      []image.Rectangle
	Confidence float32
}

// Line is a line of words in the order they sit on screen, left to right,
// with the box it was read from.
type Line struct {
	Box   image.Rectangle
	Words []Word
}

// Engine reads frames with the model New was given, and with any added by
// WithModel when a router picks among them. It is safe for concurrent use.
type Engine struct {
	model    *Model
	byFamily map[string]*Model
	router   *Model
	workers  int
	pool     sync.Pool
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
	e := &Engine{model: m, byFamily: map[string]*Model{m.net.Family: m}, workers: runtime.NumCPU()}
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
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	// Longest lines first, so no worker is left with a long one at the end.
	sort.Slice(order, func(a, b int) bool { return boxes[order[a]].Dx() > boxes[order[b]].Dx() })
	// Every line is prepared and routed before any is read, so the scripts
	// the whole page is written in decide each line's recognizer.
	preps := make([]prepared, len(boxes))
	e.each(ctx, order, func(i int) { preps[i] = e.prepare(frame, boxes[i], scale) })
	var page map[string]bool
	if e.router != nil {
		page = pageFamilies(preps)
	}
	lines := make([]Line, len(boxes))
	e.each(ctx, order, func(i int) {
		lines[i] = Line{Box: boxes[i], Words: e.readPrepared(frame, preps[i], scale, page)}
	})
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
// text, without line finding, for a second look at a known line.
func (e *Engine) ReadLine(frame *image.RGBA, box image.Rectangle, scale int) []Word {
	scale = max(1, scale)
	return e.readPrepared(frame, e.prepare(frame, box, scale), scale, nil)
}

// each runs f for each index in order on the engine's workers, and stops
// starting new ones once ctx is done.
func (e *Engine) each(ctx context.Context, order []int, f func(int)) {
	next := make(chan int, len(order))
	for _, i := range order {
		next <- i
	}
	close(next)
	var wg sync.WaitGroup
	for range max(1, min(e.workers, len(order))) {
		wg.Go(func() {
			for i := range next {
				if ctx.Err() != nil {
					return
				}
				f(i)
			}
		})
	}
	wg.Wait()
}

// prepared is a line ready to read: its rectangle in frame pixels, its ink
// map and the router's view of it. x is nil for a line too short to read.
type prepared struct {
	r     image.Rectangle
	x     []float32
	w     int
	route route
}

func (e *Engine) prepare(frame *image.RGBA, box image.Rectangle, scale int) prepared {
	r := image.Rect(box.Min.X*scale, box.Min.Y*scale, box.Max.X*scale, box.Max.Y*scale).
		Add(frame.Rect.Min).Intersect(frame.Rect)
	// Every model reads the same input: an ink map 32 rows tall.
	x, w := prep.Prepare(frame, r, e.model.net.Height)
	if x == nil || w < e.model.net.Stride {
		return prepared{}
	}
	p := prepared{r: r, x: x, w: w}
	if e.router != nil {
		s := e.pool.Get().(*nn.Scratch)
		p.route = e.routeLine(x, w, s)
		e.pool.Put(s)
	}
	return p
}

// readPrepared reads p with the recognizers pick chooses for it on page.
func (e *Engine) readPrepared(frame *image.RGBA, p prepared, scale int, page map[string]bool) []Word {
	if p.x == nil {
		return nil
	}
	s := e.pool.Get().(*nn.Scratch)
	defer e.pool.Put(s)
	var (
		net   *nn.Model
		steps []nn.Step
		sure  float64
	)
	tried := e.pick(p.route, page)
	for _, n := range tried {
		st := n.Steps(p.x, p.w, s)
		if v := sureness(st); net == nil || v > sure {
			net, steps, sure = n, st, v
		}
	}
	if net == nil {
		return nil
	}
	// A line another script's recognizer reads as mostly ASCII is Latin it was
	// sent away from, often for an arrow or icon beside a word, and the base
	// recognizer reads it too. The surer read stays.
	if net != e.model.net && !slices.Contains(tried, e.model.net) && mostlyASCII(steps, []rune(net.Alphabet)) {
		if st := e.model.net.Steps(p.x, p.w, s); sureness(st) >= sure {
			net, steps = e.model.net, st
		}
	}
	words := decode(steps, []rune(net.Alphabet), net.Stride, p.r.Sub(frame.Rect.Min), p.w, scale)
	if net != e.model.net {
		words = slices.DeleteFunc(words, func(w Word) bool { return w.Text == roundIcon })
	}
	settleScripts(words)
	for i := range words {
		logicalOrder(&words[i])
	}
	return words
}

// mostlyASCII reports whether at least half the characters steps read are
// ASCII letters or digits.
func mostlyASCII(steps []nn.Step, alpha []rune) bool {
	ascii, n, prev := 0, 0, 0
	for _, st := range steps {
		if st.Class != prev && st.Class != 0 {
			c := alpha[st.Class-1]
			if c != ' ' {
				n++
				if c < 0x80 && (unicode.IsLetter(c) || unicode.IsDigit(c)) {
					ascii++
				}
			}
		}
		prev = st.Class
	}
	return n > 0 && ascii*2 >= n
}

// loneFloor is the confidence a one-character word needs. On the layout
// scenes it halves the stray words and costs one real word in 6,050 on the UI
// set.
const loneFloor = 0.5

// gapBreak is the gap between two characters, as a share of the line's
// height, past which they are two words even when the recognizer read no space
// between them. Table columns and labels side by side sit further apart than
// any space the models trained on. The line finder's crops keep a glyph's
// top strokes, so they run taller than the ink. At 0.75 Hindi reads 0.4 to
// 0.5 points more words than at 0.8 and CJK 0.2 more characters, with
// Western, right-to-left and the layout scenes level. At 0.7 Korean words
// split into syllables, and at 0.6 Western and right-to-left lose a point.
const gapBreak = 0.75

// decode is greedy CTC: the best class per step, repeats and blanks dropped,
// spaces splitting words. A word's box spans its first to last character's
// step, and each character's box meets its neighbours' halfway between steps.
// It drops a lone character read below loneFloor, since that is nearly always
// an icon, such as a checkbox read as O or a search glass read as Q, and
// breaks a word at a gap wider than gapBreak.
func decode(steps []nn.Step, alpha []rune, stride int, r image.Rectangle, w, scale int) []Word {
	sx := float64(r.Dx()) / float64(w)
	// x is the frame column where step t starts, in logical pixels.
	x := func(t float64) int {
		return int(math.Round((float64(r.Min.X) + t*float64(stride)*sx) / float64(scale)))
	}
	y0, y1 := r.Min.Y/scale, r.Max.Y/scale
	var (
		words []Word
		cur   []rune
		at    []int // the step each character of cur was read at
		conf  float32
		prev  int
	)
	flush := func() {
		if len(cur) == 0 || len(cur) == 1 && conf < loneFloor {
			cur, at = nil, nil
			return
		}
		chars := make([]image.Rectangle, len(cur))
		left := x(float64(at[0]))
		for i := range cur {
			right := x(float64(at[i] + 1))
			if i+1 < len(cur) {
				right = x(float64(at[i]+at[i+1]+1) / 2)
			}
			chars[i] = image.Rect(left, y0, max(right, left+1), y1)
			left = chars[i].Max.X
		}
		words = append(words, Word{
			Text:       string(cur),
			Box:        image.Rect(chars[0].Min.X, y0, chars[len(chars)-1].Max.X, y1),
			Chars:      chars,
			Confidence: conf,
		})
		cur, at = nil, nil
	}
	for j, st := range steps {
		best := st.Class
		if best != prev && best != 0 {
			ch := alpha[best-1]
			if ch == ' ' {
				flush()
			} else {
				if len(cur) > 0 && float64(j-at[len(at)-1])*float64(stride)*sx > gapBreak*float64(r.Dy()) {
					flush()
				}
				if len(cur) == 0 {
					conf = st.Prob
				}
				conf = min(conf, st.Prob)
				cur = append(cur, ch)
				at = append(at, j)
			}
		}
		prev = best
	}
	flush()
	return words
}
