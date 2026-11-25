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
	"slices"
	"sort"
	"sync"
)

// Params tune the finder, in logical pixels.
type Params struct {
	Edge      int     // how far ink differs from the background beside it
	LowEdge   int     // the same, for lines on a plain background found in a second pass
	QuietRing int     // how far a plain background's luma may spread around its line
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
		LowEdge:   15,
		QuietRing: 8,
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

func lum(p []uint8) int { return (299*int(p[0]) + 587*int(p[1]) + 114*int(p[2])) / 1000 }

// luma1x returns the frame's luma at one pixel per logical pixel, box
// filtering by k when scale is a whole k >= 2, and area resampling when it
// falls between.
func luma1x(img *image.RGBA, scale float64) ([]uint8, int, int) {
	r := math.Round(scale)
	if math.Abs(scale-r) > 0.01 {
		return lumaScaled(img, scale)
	}
	k := max(1, int(r))
	b := img.Rect
	w, h := b.Dx()/k, b.Dy()/k
	out := make([]uint8, w*h)
	bands(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			o := out[y*w : y*w+w]
			switch k {
			case 1:
				row := img.Pix[y*img.Stride : y*img.Stride+w*4]
				for x := range o {
					o[x] = uint8(lum(row[x*4 : x*4+3]))
				}
			case 2:
				r0 := img.Pix[2*y*img.Stride : 2*y*img.Stride+w*8]
				r1 := img.Pix[(2*y+1)*img.Stride : (2*y+1)*img.Stride+w*8]
				for x := range o {
					i := x * 8
					o[x] = uint8((lum(r0[i:i+3]) + lum(r0[i+4:i+7]) + lum(r1[i:i+3]) + lum(r1[i+4:i+7])) / 4)
				}
			default:
				for x := range o {
					sum := 0
					for dy := 0; dy < k; dy++ {
						row := img.Pix[(y*k+dy)*img.Stride:]
						for dx := 0; dx < k; dx++ {
							sum += lum(row[(x*k+dx)*4:])
						}
					}
					o[x] = uint8(sum / (k * k))
				}
			}
		}
	})
	return out, w, h
}

// tap is one frame pixel's share of an output pixel.
type tap struct {
	i int
	w float32
}

// taps returns, for each of n output pixels scale frame pixels wide, the
// frame pixels it covers, each with the fraction of the output pixel it makes
// up.
func taps(n, src int, scale float64) [][]tap {
	t := make([][]tap, n)
	for o := range t {
		lo, hi := float64(o)*scale, float64(o+1)*scale
		for i := int(lo); float64(i) < hi && i < src; i++ {
			c := math.Min(hi, float64(i+1)) - math.Max(lo, float64(i))
			t[o] = append(t[o], tap{i, float32(c / scale)})
		}
	}
	return t
}

// lumaScaled returns the frame's luma at one pixel per logical pixel for a
// fractional scale, each pixel the mean of the frame area it covers.
func lumaScaled(img *image.RGBA, scale float64) ([]uint8, int, int) {
	sw, sh := img.Rect.Dx(), img.Rect.Dy()
	w, h := int(float64(sw)/scale), int(float64(sh)/scale)
	xt, yt := taps(w, sw, scale), taps(h, sh, scale)
	rows := make([]float32, w*sh)
	bands(sh, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			src, o := img.Pix[y*img.Stride:], rows[y*w:y*w+w]
			for x, t := range xt {
				var v float32
				for _, p := range t {
					v += p.w * float32(lum(src[p.i*4:]))
				}
				o[x] = v
			}
		}
	})
	out := make([]uint8, w*h)
	bands(h, func(y0, y1 int) {
		acc := make([]float32, w)
		for y := y0; y < y1; y++ {
			clear(acc)
			for _, p := range yt[y] {
				for x, v := range rows[p.i*w : p.i*w+w] {
					acc[x] += p.w * v
				}
			}
			o := out[y*w : y*w+w]
			for x, v := range acc {
				o[x] = uint8(v + 0.5)
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
	// stemWidth is the narrowest ink a tall vertical run sits in for it to
	// count as a glyph's stroke and not a border.
	stemWidth = 3
	// strokeRun is the longest even run that can still be the inside of a
	// glyph's stroke: three flat runs, past a 26 px bold figure's 6.
	strokeRun = 3 * flatRun
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
		type run struct{ s, e, level int }
		var runs []run
		for y := y0; y < y1; y++ {
			row := l[y*w : y*w+w]
			for x := range flat {
				flat[x] = false
			}
			allFlat := true
			runs = runs[:0]
			for x := 0; x < w; {
				lo, hi, s := row[x], row[x], x
				for x < w && max(hi, row[x])-min(lo, row[x]) <= flatSpread {
					lo, hi = min(lo, row[x]), max(hi, row[x])
					x++
				}
				if x-s >= flatRun {
					runs = append(runs, run{s, x, (int(lo) + int(hi)) / 2})
				} else {
					allFlat = false
				}
				if x == s {
					x++
				}
			}
			// A row of background alone marks nothing.
			if allFlat && len(runs) == 1 {
				continue
			}
			// A run is background when it is at the level of the row's longest
			// run or too long to be a stroke.
			var bg run
			for _, r := range runs {
				if r.e-r.s > bg.e-bg.s {
					bg = r
				}
			}
			ground := func(k int) bool {
				return k >= 0 && k < len(runs) &&
					(absInt(runs[k].level-bg.level) <= p.Edge || runs[k].e-runs[k].s > strokeRun)
			}
			for k, r := range runs {
				// A short run whose level differs from the runs on both sides,
				// which agree with each other and are background, is the inside
				// of a stroke too wide to be anything but even, as in large bold
				// text. It is ink, not background. Two strokes beside a gap too
				// wide to be a run do not make the gap a stroke.
				if r.e-r.s <= strokeRun && ground(k-1) && ground(k+1) &&
					absInt(r.level-runs[k-1].level) > p.Edge &&
					absInt(r.level-runs[k+1].level) > p.Edge &&
					absInt(runs[k-1].level-runs[k+1].level) <= p.Edge {
					continue
				}
				for i := r.s; i < r.e; i++ {
					flat[i] = true
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
	// The column pass walks the rows top to bottom with a count per column of
	// the marked run it is in, so it reads memory in order.
	bands(w, func(x0, x1 int) {
		run := make([]int, x1-x0)
		end := func(x, y int) {
			if n := run[x-x0]; n >= p.RunMaxV && !stem(e, w, x, y-n, y) {
				for i := y - n; i < y; i++ {
					clear[i*w+x] = true
				}
			}
			run[x-x0] = 0
		}
		for y := range h {
			er := e[y*w+x0 : y*w+x1]
			for i, m := range er {
				if m {
					run[i]++
				} else if run[i] > 0 {
					end(x0+i, y)
				}
			}
		}
		for x := x0; x < x1; x++ {
			end(x, h)
		}
	})
	for i := range e {
		if clear[i] {
			e[i] = false
		}
	}
	return e
}

// stem reports whether the marked run in column x from row y0 up to y1 is a
// glyph's stroke rather than a border: at least half its rows sit in ink at
// least stemWidth wide, as the stem of large bold text does. A border or the
// side of a field is one or two pixels wide.
func stem(e []bool, w, x, y0, y1 int) bool {
	wide := 0
	for y := y0; y < y1; y++ {
		row := e[y*w : y*w+w]
		a, b := x, x
		for a > 0 && row[a-1] {
			a--
		}
		for b+1 < w && row[b+1] {
			b++
		}
		if b-a+1 >= stemWidth {
			wide++
		}
	}
	return wide*2 >= y1-y0
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

// absorbMarks folds a line of one blob into a line at least twice its height
// that it sits beside, in that line's band, as the first pass folds in small
// marks. The comma of a large figure is too tall to count as a small mark, but
// the figure's digits wait for the tall pass, so the first pass gives the
// comma a line of its own, and below the digits' baseline it overlaps them too
// little for mergeStacked.
func absorbMarks(lines []line, p Params) []line {
	for i := 0; i < len(lines); i++ {
		if len(lines[i].members) != 1 {
			continue
		}
		b := lines[i].members[0]
		cy := (b.y0 + b.y1) / 2
		for j := range lines {
			if j == i || len(lines[j].members) < 2 {
				continue
			}
			ln := &lines[j]
			c0, c1 := ln.core()
			ch := c1 - c0 + 1
			g, up, down := int(p.GapEm*float64(ch)+0.5), ch*2/5, ch*7/10
			if b.h()*2 > ch || cy < c0-up || cy > c1+down || b.x0 < ln.x0-g || b.x1 > ln.x1+g {
				continue
			}
			ln.x0, ln.x1 = min(ln.x0, b.x0), max(ln.x1, b.x1)
			ln.y0, ln.y1 = min(ln.y0, b.y0), max(ln.y1, b.y1)
			lines = append(lines[:i], lines[i+1:]...)
			i--
			break
		}
	}
	return lines
}

// mergeStacked merges lines that lie mostly over one another. A CJK glyph
// stacks separate parts, such as a Hangul syllable's final consonant, and a
// part too low to join its line's core seeds a second line along the bottom
// of the first, which would read the same text again. The upper parts of a
// large heading's glyphs can seed one that overlaps its row by half. Lines of
// text never overlap like this, so lines whose spans share over two fifths of
// the smaller one's height, and whose bounds share over half its width, are
// one line.
func mergeStacked(lines []line) []line {
	for merged := true; merged; {
		merged = false
		for i := 0; i < len(lines); i++ {
			for j := i + 1; j < len(lines); j++ {
				a, b := lines[i], lines[j]
				a0, a1 := a.span()
				b0, b1 := b.span()
				if overlap(a0, a1, b0, b1)*5 <= min(a1-a0+1, b1-b0+1)*2 ||
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

// oneRow reports whether every line in hits has a core overlapping best's by
// at least half the shorter of the two.
func oneRow(lines []line, hits []int, best int) bool {
	b0, b1 := lines[best].core()
	for _, i := range hits {
		c0, c1 := lines[i].core()
		if overlap(c0, c1, b0, b1)*2 < min(c1-c0+1, b1-b0+1) {
			return false
		}
	}
	return true
}

// mergeLines folds the lines in hits, which are in ascending order, into the
// first of them and drops the rest.
func mergeLines(lines []line, hits []int) []line {
	ln := &lines[hits[0]]
	for _, i := range hits[1:] {
		o := lines[i]
		ln.x0, ln.x1 = min(ln.x0, o.x0), max(ln.x1, o.x1)
		ln.y0, ln.y1 = min(ln.y0, o.y0), max(ln.y1, o.y1)
		ln.sy0, ln.sy1, ln.n = ln.sy0+o.sy0, ln.sy1+o.sy1, ln.n+o.n
		ln.members = append(ln.members, o.members...)
	}
	for k := len(hits) - 1; k > 0; k-- {
		lines = slices.Delete(lines, hits[k], hits[k]+1)
	}
	return lines
}

// within reports whether b lies inside the columns of a taller member, as a
// piece of a large glyph does when its strokes come apart. Its height is not
// the glyph's.
func (l line) within(b blob) bool {
	for _, m := range l.members {
		if m.h() > b.h() && m.x0 <= b.x0 && b.x1 <= m.x1 {
			return true
		}
	}
	return false
}

// metrics returns a line's baseline (the lower median blob bottom) and the
// heights of the blobs that sit on it, sorted. Of an even count the higher of
// the middle two wins, so a comma or descender among two members never sets
// the baseline below the glyph beside it.
func (l line) metrics() (int, []int) {
	bottoms := make([]int, len(l.members))
	for i, b := range l.members {
		bottoms[i] = b.y1
	}
	sort.Ints(bottoms)
	base := bottoms[(len(bottoms)-1)/2]
	var hs []int
	for _, b := range l.members {
		if absInt(b.y1-base) <= 1 && !l.within(b) {
			hs = append(hs, b.h())
		}
	}
	sort.Ints(hs)
	return base, hs
}

// noBaseline is how far above the bottom of a line's ink, as a share of its
// height, the median bottom of its glyphs may sit and still be its baseline.
// Latin descenders reach about a quarter, the split CJK heading that set it
// 0.58.
const noBaseline = 0.55

// Crop geometry in x-heights, matching the training crops: the font's line
// box is about 2.3 x-heights, ascent 1.8 and descent 0.5, and training pads
// each side by 10-35% of it. Measured against the ground-truth crops on the
// development pages, 2.3 and 1.0 came out 17% tall, so these are 15% less.
const (
	aboveXH = 1.95
	belowXH = 0.85
	sideXH  = 0.5
)

// Box is a line found in a frame, in logical pixels. Crop is the region the
// recognizer reads, padded around the text the way it trained. Ink bounds the
// line's marks.
type Box struct {
	Crop, Ink image.Rectangle
}

// Find returns the lines in img.
func Find(img *image.RGBA, scale float64, p Params) []Box {
	l, w, h := luma1x(img, scale)
	if p.LowEdge >= p.Edge {
		return find(l, w, h, p)
	}
	// Faint text, as grey on white or white on a light bar, differs from its
	// background by less than Edge. A second pass at LowEdge, run alongside
	// the first, adds the lines it finds where the first found none and the
	// background around the line is plain, which leaves out the texture and
	// photographs a low edge marks.
	lp := p
	lp.Edge = p.LowEdge
	var faint []Box
	var wg sync.WaitGroup
	wg.Go(func() { faint = find(l, w, h, lp) })
	out := find(l, w, h, p)
	wg.Wait()
	first := len(out)
	for _, b := range faint {
		if !overlapsAny(b.Ink, out[:first]) && quiet(l, w, h, b.Ink, p.QuietRing) {
			out = append(out, b)
		}
	}
	return out
}

func overlapsAny(r image.Rectangle, boxes []Box) bool {
	for _, b := range boxes {
		if r.Overlaps(b.Ink) {
			return true
		}
	}
	return false
}

// quiet reports whether the ring of pixels around ink, half its height wide,
// spreads over at most spread luma levels, the 5th to the 95th percentile.
func quiet(l []uint8, w, h int, ink image.Rectangle, spread int) bool {
	m := max(2, ink.Dy()/2)
	o := ink.Inset(-m).Intersect(image.Rect(0, 0, w, h))
	var hist [256]int
	n := 0
	for y := o.Min.Y; y < o.Max.Y; y++ {
		for x := o.Min.X; x < o.Max.X; x++ {
			if image.Pt(x, y).In(ink) {
				continue
			}
			hist[l[y*w+x]]++
			n++
		}
	}
	if n == 0 {
		return false
	}
	lo, hi, c := -1, 0, 0
	for v, k := range hist {
		c += k
		if lo < 0 && c*20 > n {
			lo = v
		}
		if c*20 >= n*19 {
			hi = v
			break
		}
	}
	return hi-lo <= spread
}

// find runs the finder over l, the frame's luma at one pixel per logical
// pixel, with p.Edge.
func find(l []uint8, w, h int, p Params) []Box {
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
	// Blobs over 2.2 times the median glyph height wait for a second pass.
	// At small sizes with tight leading, the marks below one row of Thai or
	// Myanmar touch those above the next, and the blob they make spans both
	// rows. Grouped with the rest, it can seed a line whose core covers both,
	// which then takes in the glyphs of both rows.
	var tall []blob
	if len(big) > 0 {
		hs := make([]int, len(big))
		for i, b := range big {
			hs[i] = b.h()
		}
		sort.Ints(hs)
		mh := hs[len(hs)/2]
		kept := big[:0]
		for _, b := range big {
			if b.h()*5 > mh*11 {
				tall = append(tall, b)
			} else {
				kept = append(kept, b)
			}
		}
		big = kept
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

	// A tall blob joins the line whose core it overlaps most, as the first
	// pass would have. A blob overlapping two lines' cores on different rows
	// is marks fused across them, so it widens its line without making it
	// taller. Lines on one row that it spans, as the halves of a heading in
	// large type split at a wide gap, merge with it. A blob overlapping none,
	// such as a heading's glyph, seeds a line.
	sort.Slice(tall, func(i, j int) bool { return tall[i].x0 < tall[j].x0 })
	for _, b := range tall {
		best, bestOv, hits := -1, 0, []int(nil)
		for i, ln := range lines {
			c0, c1 := ln.core()
			ch := c1 - c0 + 1
			g := int(p.GapEm*float64(ch) + 0.5)
			if b.x0-ln.x1 > g || ln.x0-b.x1 > g {
				continue
			}
			ov := overlap(b.y0, b.y1, c0, c1)
			if ov*2 < min(b.h(), ch) {
				continue
			}
			hits = append(hits, i)
			if ov > bestOv {
				best, bestOv = i, ov
			}
		}
		if len(hits) > 1 && oneRow(lines, hits, best) {
			lines = mergeLines(lines, hits)
			best, hits = hits[0], hits[:1]
		}
		switch {
		case best < 0:
			lines = append(lines, line{b.x0, b.y0, b.x1, b.y1, b.y0, b.y1, 1, []blob{b}})
		case len(hits) > 1:
			ln := &lines[best]
			ln.x0, ln.x1 = min(ln.x0, b.x0), max(ln.x1, b.x1)
		default:
			ln := &lines[best]
			ln.x0, ln.x1 = min(ln.x0, b.x0), max(ln.x1, b.x1)
			ln.y0, ln.y1 = min(ln.y0, b.y0), max(ln.y1, b.y1)
			ln.sy0, ln.sy1, ln.n = ln.sy0+b.y0, ln.sy1+b.y1, ln.n+1
			ln.members = append(ln.members, b)
		}
	}

	lines = absorbMarks(lines, p)
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
	// lower marks a line of display type, glyph blobs at least 16 px tall,
	// whose top sits 1.25 to 1.6 times their height above its baseline, as
	// far as capitals reach over lowercase. Large bold type runs its lowercase
	// into blobs of their own and a second pass joins its capitals, so its
	// blobs are lowercase and must not take the page's capital height.
	lower := make([]bool, len(lines))
	var page []known
	for i, ln := range lines {
		base, hs := ln.metrics()
		bases[i] = base
		// The parts of large CJK glyphs end at many heights, and their median
		// bottom can sit mid-glyph, which crops the line to its ink. A baseline
		// that far above the ink's bottom is none, and the line reads as one
		// height standing on its ink.
		if float64(ln.y1-base) > noBaseline*float64(ln.h()) {
			bases[i] = ln.y1
			ambiguous[i] = float64(ln.h())
			continue
		}
		if len(hs) == 0 {
			ambiguous[i] = float64(ln.h())
			continue
		}
		lo, hi := float64(hs[len(hs)/4]), float64(hs[len(hs)-1])
		if hi >= 1.25*lo {
			xhs[i] = lo
			// Fonts set capitals 1.25 to 1.8 times their x-height. A pair
			// further apart is an icon beside small text, and as the page's
			// evidence it would read a large figure as capitals of a far
			// smaller x-height and crop it to the top of its digits.
			if hi <= 1.8*lo {
				page = append(page, known{lo, hi})
			}
		} else {
			ambiguous[i] = float64(hs[len(hs)/2])
			reach := float64(base-ln.y0+1) / hi
			lower[i] = hi >= 16 && reach >= 1.25 && reach <= 1.6
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
			if e := math.Abs(v-k.cap) / k.cap; e < bestErr && !lower[i] {
				xhs[i], bestErr = v*k.xh/k.cap, e
			}
		}
	}

	out := make([]Box, 0, len(lines))
	for i, ln := range lines {
		xh := xhs[i]
		top := min(ln.y0-1, bases[i]-int(aboveXH*xh+0.5))
		bot := max(ln.y1+2, bases[i]+1+int(belowXH*xh+0.5))
		side := int(sideXH*xh + 0.5)
		out = append(out, Box{
			Crop: image.Rect(ln.x0-side, top, ln.x1+1+side, bot).Intersect(image.Rect(0, 0, w, h)),
			Ink:  image.Rect(ln.x0, ln.y0, ln.x1+1, ln.y1+1),
		})
	}
	return out
}
