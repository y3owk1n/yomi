// Package linefind finds text lines in a screen frame: flat colour plus glyph
// ink. It works at one pixel per logical pixel, marks ink against flat
// background runs, drops long straight runs (rules, borders, outlines), joins
// ink into blobs, groups glyph-sized blobs into lines, and crops each line
// from its baseline and x-height the way the recognizer was trained.
package linefind

import (
	"image"
	"math"
	"runtime"
	"sort"
	"sync"
)

// Params tune the finder, in logical pixels.
type Params struct {
	Edge      int     // how far ink differs from the background beside it
	RunMax    int     // horizontal ink runs at least this long are rules, not glyphs
	RunMaxV   int     // vertical ink runs at least this long are borders, not glyphs
	MinGlyphH int     // blobs at least this tall seed lines
	MaxGlyphH int     // blobs taller than this are not text
	GapEm     float64 // largest gap inside a line, in line heights
}

// DefaultParams are the values tuned on the development pages.
func DefaultParams() Params {
	return Params{
		Edge:      40,
		RunMax:    24,
		RunMaxV:   16,
		MinGlyphH: 5,
		MaxGlyphH: 48,
		GapEm:     1.6,
	}
}

type blob struct {
	x0, y0, x1, y1 int // inclusive
}

func (b blob) h() int { return b.y1 - b.y0 + 1 }
func (b blob) w() int { return b.x1 - b.x0 + 1 }

// luma1x returns the frame's luma at one pixel per logical pixel, box
// filtering by k = floor(scale) when scale >= 2.
func luma1x(img *image.RGBA, scale float64) ([]uint8, int, int) {
	k := max(1, int(scale))
	b := img.Rect
	w, h := b.Dx()/k, b.Dy()/k
	out := make([]uint8, w*h)
	bands(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < w; x++ {
				sum := 0
				for dy := 0; dy < k; dy++ {
					row := img.Pix[(y*k+dy)*img.Stride:]
					for dx := 0; dx < k; dx++ {
						p := row[(x*k+dx)*4:]
						sum += (299*int(p[0]) + 587*int(p[1]) + 114*int(p[2])) / 1000
					}
				}
				out[y*w+x] = uint8(sum / (k * k))
			}
		}
	})
	return out, w, h
}

func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// flatRun is how long a run of near-equal luma must be to count as
// background, and flatSpread how far its values may spread.
const (
	flatRun    = 6
	flatSpread = 6
)

// bands runs fn over [0, n) split into one band per CPU, in parallel.
func bands(n int, fn func(lo, hi int)) {
	k := min(runtime.NumCPU(), max(1, n/16))
	var wg sync.WaitGroup
	for i := range k {
		lo, hi := n*i/k, n*(i+1)/k
		wg.Go(func() { fn(lo, hi) })
	}
	wg.Wait()
}

// edges marks ink: per row, runs of at least flatRun pixels within
// flatSpread of each other are background, and any other pixel that differs
// by more than p.Edge from the nearest background on each side that has one
// is ink. Then it clears straight runs of at least RunMax (horizontal) or
// RunMaxV (vertical) marked pixels, which rules, borders and outlines make
// and glyphs at text sizes do not. Rows, then columns, run in parallel bands.
func edges(l []uint8, w, h int, p Params) []bool {
	e := make([]bool, w*h)
	clear := make([]bool, w*h)
	bands(h, func(y0, y1 int) {
		flat := make([]bool, w)
		left := make([]int, w)
		right := make([]int, w)
		for y := y0; y < y1; y++ {
			row := l[y*w : y*w+w]
			for x := range flat {
				flat[x] = false
			}
			for x := 0; x < w; {
				lo, hi, s := row[x], row[x], x
				for x < w && max(hi, row[x])-min(lo, row[x]) <= flatSpread {
					lo, hi = min(lo, row[x]), max(hi, row[x])
					x++
				}
				if x-s >= flatRun {
					for i := s; i < x; i++ {
						flat[i] = true
					}
				}
				if x == s {
					x++
				}
			}
			last := -1
			for x := range w {
				if flat[x] {
					last = int(row[x])
				}
				left[x] = last
			}
			last = -1
			for x := w - 1; x >= 0; x-- {
				if flat[x] {
					last = int(row[x])
				}
				right[x] = last
			}
			er := e[y*w : y*w+w]
			for x := range w {
				if flat[x] {
					continue
				}
				v := int(row[x])
				dl := left[x] < 0 || absInt(v-left[x]) > p.Edge
				dr := right[x] < 0 || absInt(v-right[x]) > p.Edge
				if dl && dr && (left[x] >= 0 || right[x] >= 0) {
					er[x] = true
				}
			}
			for x := 0; x < w; {
				if !er[x] {
					x++
					continue
				}
				s := x
				for x < w && er[x] {
					x++
				}
				if x-s >= p.RunMax {
					for i := s; i < x; i++ {
						clear[y*w+i] = true
					}
				}
			}
		}
	})
	bands(w, func(x0, x1 int) {
		for x := x0; x < x1; x++ {
			for y := 0; y < h; {
				if !e[y*w+x] {
					y++
					continue
				}
				s := y
				for y < h && e[y*w+x] {
					y++
				}
				if y-s >= p.RunMaxV {
					for i := s; i < y; i++ {
						clear[i*w+x] = true
					}
				}
			}
		}
	})
	for i := range e {
		if clear[i] {
			e[i] = false
		}
	}
	return e
}

// blobs labels 8-connected marked pixels with a union-find over pixels.
func blobs(e []bool, w, h int) []blob {
	parent := make([]int32, w*h)
	for i := range parent {
		parent[i] = -1
	}
	find := func(a int32) int32 {
		for parent[a] != a {
			parent[a] = parent[parent[a]]
			a = parent[a]
		}
		return a
	}
	union := func(a, b int32) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := int32(y*w + x)
			if !e[i] {
				continue
			}
			parent[i] = i
			if x > 0 && e[i-1] {
				union(i, i-1)
			}
			if y > 0 {
				up := i - int32(w)
				if e[up] {
					union(i, up)
				}
				if x > 0 && e[up-1] {
					union(i, up-1)
				}
				if x+1 < w && e[up+1] {
					union(i, up+1)
				}
			}
		}
	}
	index := map[int32]int{}
	var out []blob
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := int32(y*w + x)
			if !e[i] {
				continue
			}
			r := find(i)
			k, ok := index[r]
			if !ok {
				k = len(out)
				index[r] = k
				out = append(out, blob{x, y, x, y})
			}
			b := &out[k]
			b.x0, b.x1 = min(b.x0, x), max(b.x1, x)
			b.y0, b.y1 = min(b.y0, y), max(b.y1, y)
		}
	}
	return out
}

type line struct {
	x0, y0, x1, y1 int // bounds of everything joined, small marks included
	sy0, sy1, n    int // sums of the big members' tops and bottoms, and their count
	members        []blob
}

func (l line) h() int { return l.y1 - l.y0 + 1 }

func (l line) w() int { return l.x1 - l.x0 + 1 }

// mergeStacked merges lines that lie mostly over one another. A CJK glyph
// stacks separate parts, such as a Hangul syllable's final consonant, and a
// part too low to join its line's core seeds a second line along the bottom
// of the first, which would read the same text again. Lines of text never
// overlap like this, so lines whose spans share over half the smaller one's
// height, and whose bounds share over half its width, are one line.
func mergeStacked(lines []line) []line {
	for merged := true; merged; {
		merged = false
		for i := 0; i < len(lines); i++ {
			for j := i + 1; j < len(lines); j++ {
				a, b := lines[i], lines[j]
				a0, a1 := a.span()
				b0, b1 := b.span()
				if overlap(a0, a1, b0, b1)*2 <= min(a1-a0+1, b1-b0+1) ||
					overlap(a.x0, a.x1, b.x0, b.x1)*2 <= min(a.w(), b.w()) {
					continue
				}
				lines[i] = line{
					min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1),
					a.sy0 + b.sy0, a.sy1 + b.sy1, a.n + b.n, append(a.members, b.members...),
				}
				lines = append(lines[:j], lines[j+1:]...)
				merged = true
				j--
			}
		}
	}
	return lines
}

// span is the vertical extent of the line's big members, leaving out any
// taller than 2.2 core heights. At small sizes with tight leading, the Thai
// marks below one row touch those above the next. The blob they make is 2.4
// core heights or more, it joins one row, and its bounds would overlap the
// next row enough to merge the two. The tallest CJK glyphs reach 1.9.
func (l line) span() (int, int) {
	c0, c1 := l.core()
	ch := c1 - c0 + 1
	y0, y1 := l.y1, l.y0
	for _, b := range l.members {
		if b.h()*5 <= ch*11 {
			y0, y1 = min(y0, b.y0), max(y1, b.y1)
		}
	}
	return y0, y1
}

// core is the line's band for joining: the mean top and bottom of its big
// members. Unlike the union of their bounds, it cannot creep down a row one
// descender at a time.
func (l line) core() (int, int) { return l.sy0 / l.n, l.sy1 / l.n }

func overlap(a0, a1, b0, b1 int) int {
	return max(0, min(a1, b1)-max(a0, b0)+1)
}

// metrics returns a line's baseline (the median blob bottom) and the heights
// of the blobs that sit on it, sorted.
func (l line) metrics() (int, []int) {
	bottoms := make([]int, len(l.members))
	for i, b := range l.members {
		bottoms[i] = b.y1
	}
	sort.Ints(bottoms)
	base := bottoms[len(bottoms)/2]
	var hs []int
	for _, b := range l.members {
		if absInt(b.y1-base) <= 1 {
			hs = append(hs, b.h())
		}
	}
	sort.Ints(hs)
	return base, hs
}

// Crop geometry in x-heights, matching the training crops: the font's line
// box is about 2.3 x-heights, ascent 1.8 and descent 0.5, and training pads
// each side by 10-35% of it. Measured against the ground-truth crops on the
// development pages, 2.3 and 1.0 came out 17% tall, so these are 15% less.
const (
	aboveXH = 1.95
	belowXH = 0.85
	sideXH  = 0.5
)

// Find returns line boxes in logical pixels, sized for the recognizer.
func Find(img *image.RGBA, scale float64, p Params) []image.Rectangle {
	l, w, h := luma1x(img, scale)
	bs := blobs(edges(l, w, h, p), w, h)

	var big, small []blob
	for _, b := range bs {
		switch {
		case b.h() > p.MaxGlyphH:
			continue // a picture, a panel, or a frame around something
		case b.w() <= 3 && b.h() >= 18:
			continue // the side of a highlight, a field or a button
		case b.h() >= p.MinGlyphH:
			big = append(big, b)
		default:
			small = append(small, b)
		}
	}
	// One pass left to right over every blob. Big blobs seed or join lines.
	// Small marks (dots, commas, hyphens, quotes) only join, which extends the
	// line so the next glyph's gap is measured from them, as in "v7.13".
	all := append(append([]blob{}, big...), small...)
	isBig := make([]bool, len(all))
	for i := range big {
		isBig[i] = true
	}
	order := make([]int, len(all))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return all[order[i]].x0 < all[order[j]].x0 })

	var lines []line
	var leftover []blob
	for _, k := range order {
		b := all[k]
		best, bestOv := -1, 0
		for i, ln := range lines {
			c0, c1 := ln.core()
			ch := c1 - c0 + 1
			gap := b.x0 - ln.x1
			if gap > int(p.GapEm*float64(ch)+0.5) || b.x1 < ln.x0 {
				continue
			}
			if isBig[k] {
				ov := overlap(b.y0, b.y1, c0, c1)
				if ov*2 < min(b.h(), ch) {
					continue
				}
				if ov > bestOv {
					best, bestOv = i, ov
				}
				continue
			}
			up, down := ch*2/5, ch*7/10 // underscores and commas sit below the baseline
			cy := (b.y0 + b.y1) / 2
			if cy >= c0-up && cy <= c1+down {
				ov := ch - absInt(cy-(c0+c1)/2)
				if best < 0 || ov > bestOv {
					best, bestOv = i, ov
				}
			}
		}
		if best < 0 {
			if isBig[k] {
				lines = append(lines, line{b.x0, b.y0, b.x1, b.y1, b.y0, b.y1, 1, []blob{b}})
			} else {
				leftover = append(leftover, b)
			}
			continue
		}
		ln := &lines[best]
		ln.x0, ln.x1 = min(ln.x0, b.x0), max(ln.x1, b.x1)
		ln.y0, ln.y1 = min(ln.y0, b.y0), max(ln.y1, b.y1)
		if isBig[k] {
			ln.sy0, ln.sy1, ln.n = ln.sy0+b.y0, ln.sy1+b.y1, ln.n+1
			ln.members = append(ln.members, b)
		}
	}

	lines = mergeStacked(lines)

	// A small mark the pass met before any line could take it, as the tilde
	// of "~/path" or the quote of "'word", joins a line in whose band it sits
	// and that starts within a gap to its right or ends within one to its left.
	for _, b := range leftover {
		cy := (b.y0 + b.y1) / 2
		for i := range lines {
			ln := &lines[i]
			c0, c1 := ln.core()
			ch := c1 - c0 + 1
			g, up, down := int(p.GapEm*float64(ch)+0.5), ch*2/5, ch*7/10
			if cy < c0-up || cy > c1+down {
				continue
			}
			if (ln.x0-b.x1 >= 0 && ln.x0-b.x1 <= g) || (b.x0-ln.x1 >= 0 && b.x0-ln.x1 <= g) {
				ln.x0, ln.x1 = min(ln.x0, b.x0), max(ln.x1, b.x1)
				ln.y0, ln.y1 = min(ln.y0, b.y0), max(ln.y1, b.y1)
				break
			}
		}
	}

	// x-height per line. A line with blobs of two heights on its baseline
	// (lowercase and ascenders or capitals) knows its x-height: the lower
	// quartile. A line of one height (no ascenders, or all capitals) is
	// ambiguous, and takes whichever known x-height or cap height on the
	// page its height is nearest, relatively.
	type known struct{ xh, cap float64 }
	bases := make([]int, len(lines))
	xhs := make([]float64, len(lines))
	ambiguous := make([]float64, len(lines))
	var page []known
	for i, ln := range lines {
		base, hs := ln.metrics()
		bases[i] = base
		if len(hs) == 0 {
			ambiguous[i] = float64(ln.h())
			continue
		}
		lo, hi := float64(hs[len(hs)/4]), float64(hs[len(hs)-1])
		if hi >= 1.25*lo {
			xhs[i] = lo
			page = append(page, known{lo, hi})
		} else {
			ambiguous[i] = float64(hs[len(hs)/2])
		}
	}
	for i, v := range ambiguous {
		if v == 0 {
			continue
		}
		xhs[i] = v // no evidence: read as lowercase
		bestErr := 0.15
		for _, k := range page {
			if e := math.Abs(v-k.xh) / k.xh; e < bestErr {
				xhs[i], bestErr = v, e
			}
			if e := math.Abs(v-k.cap) / k.cap; e < bestErr {
				xhs[i], bestErr = v*k.xh/k.cap, e
			}
		}
	}

	out := make([]image.Rectangle, 0, len(lines))
	for i, ln := range lines {
		xh := xhs[i]
		top := min(ln.y0-1, bases[i]-int(aboveXH*xh+0.5))
		bot := max(ln.y1+2, bases[i]+1+int(belowXH*xh+0.5))
		side := int(sideXH*xh + 0.5)
		out = append(out, image.Rect(ln.x0-side, top, ln.x1+1+side, bot).Intersect(image.Rect(0, 0, w, h)))
	}
	return out
}
