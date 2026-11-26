// Package prep turns a line crop into the recognizer's input: an ink map
// Height rows tall, the same way train/prep.py does.
package prep

import (
	"image"
	"math"
	"slices"
)

// bgBand is how far either side of a luma the background's band reaches.
const bgBand = 8

// Prepare reproduces train/prep.py: integer luma, ink strength against the
// background under each pixel, underlines erased, integer box downsample when
// the crop is at least twice Height, then bilinear (half-pixel centres, edge
// clamp) to Height rows.
func Prepare(img *image.RGBA, r image.Rectangle, height int) ([]float32, int) {
	return PrepareBlanked(img, r, height, image.Rectangle{}, nil)
}

// PrepareBlanked is Prepare with the pixels in blank, other lines' marks
// that reach into r, set to the background before the ink map is made,
// except those in keep, the line's own marks. The polarity and contrast
// still come from the whole crop. A tight line spacing puts the bottom of the line above
// and the top of the one below inside a line's crop, and the recognizer,
// trained on crops no other word reached into, misreads the line for them.
func PrepareBlanked(img *image.RGBA, r image.Rectangle, height int, keep image.Rectangle, blank []image.Rectangle) ([]float32, int) {
	r = r.Intersect(img.Rect)
	h, w := r.Dy(), r.Dx()
	if h < 2 || w < 2 {
		return nil, 0
	}
	g := make([]uint8, h*w)
	for y := range h {
		row := img.Pix[(r.Min.Y+y-img.Rect.Min.Y)*img.Stride:]
		for x := range w {
			p := row[(r.Min.X+x-img.Rect.Min.X)*4:]
			g[y*w+x] = uint8((299*int32(p[0]) + 587*int32(p[1]) + 114*int32(p[2])) / 1000)
		}
	}
	// Background: the luma most of the crop has. A crop fitted to a line of
	// text is mostly background, and its border can run past the edge of the
	// button or field the text sits in. A background drawn as a gradient, as
	// behind a video's title, spreads over many lumas while solid text sits on
	// one, so the pick is the busiest band of bgBand lumas first, then the
	// commonest luma in it.
	var hist [256]int
	for _, v := range g {
		hist[v]++
	}
	band, best := 0, -1
	for v := range hist {
		n := 0
		for u := max(0, v-bgBand); u <= min(255, v+bgBand); u++ {
			n += hist[u]
		}
		if n > best {
			band, best = v, n
		}
	}
	bg := int32(max(0, band-bgBand))
	for v := bg; v <= int32(min(255, band+bgBand)); v++ {
		if hist[v] > hist[bg] {
			bg = v
		}
	}
	lo, hi := int32(0), int32(255)
	for hist[lo] == 0 {
		lo++
	}
	for hist[hi] == 0 {
		hi--
	}
	far, sign := hi, float32(-1)
	if bg-lo >= hi-bg {
		far, sign = lo, 1
	}
	for _, b := range blank {
		b = b.Intersect(r)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if image.Pt(x, y).In(keep) {
					continue
				}
				i := (y-r.Min.Y)*w + x - r.Min.X
				g[i] = uint8(bg)
			}
		}
	}
	// The background under each pixel is the crop opened, for ink lighter
	// than the background, or closed, for darker, by a rectangle half the
	// crop tall and twice as wide. No stroke is thick enough and no icon wide
	// enough to hold the rectangle, so both stay ink. A bar or gradient under
	// the text, as a grey bar running out of a dark page, holds it and
	// becomes background. On a flat background, as nearly every crop has,
	// the result is bg itself.
	kh, kw := h/4*2+1, h/2*4+1
	// On a flat background the first pass already gives bg everywhere, and
	// so would the second, so the second runs only when the first does not.
	b := morph(g, h, w, kh, kw, sign > 0)
	if slices.ContainsFunc(b, func(v uint8) bool { return int32(v) != bg }) {
		b = morph(b, h, w, kh, kw, sign < 0)
	}
	x := make([]float32, h*w)
	for i, v := range g {
		span := float32(max(32, abs32(int32(b[i])-far)))
		x[i] = min(1, max(0, (float32(b[i])-float32(v))*sign/span))
	}
	eraseRules(x, h, w)

	if k := h / height; k >= 2 {
		nh, nw := h/k, w/k
		d := make([]float32, nh*nw)
		inv := 1 / float32(k*k)
		for y := range nh {
			for xx := range nw {
				var s float32
				for dy := range k {
					for dx := range k {
						s += x[(y*k+dy)*w+xx*k+dx]
					}
				}
				d[y*nw+xx] = s * inv
			}
		}
		x, h, w = d, nh, nw
	}
	ow := max(4, int(math.RoundToEven(float64(w)*float64(height)/float64(h))))
	return bilinear(x, h, w, height, ow), ow
}

// morph returns g, h rows of w, with each pixel the largest, when hi is set,
// or else the smallest of the kh rows by kw columns around it, cut at the
// crop's edges. kh and kw are odd. The smallest is the largest of the
// inverted values, inverted back.
func morph(g []uint8, h, w, kh, kw int, hi bool) []uint8 {
	out := slices.Clone(g)
	if !hi {
		invert(out)
	}
	pre, suf := make([]uint8, w+kw), make([]uint8, w+kw)
	for y := range h {
		dilateRow(out[y*w:y*w+w], pre, suf, kw)
	}
	dilateCols(out, h, w, kh)
	if !hi {
		invert(out)
	}
	return out
}

func invert(g []uint8) {
	for i, v := range g {
		g[i] = 255 - v
	}
}

// dilateRow sets each value of row to the largest within kw/2 of it, by van
// Herk / Gil-Werman. It splits the row into blocks of kw and keeps running
// maxima forward and backward within each. A window's largest is the larger
// of the backward maximum at its start and the forward maximum at its end.
// pre and suf are scratch, at least len(row)+kw long.
func dilateRow(row, pre, suf []uint8, kw int) {
	r, n := kw/2, len(row)
	m := n + 2*r
	pre, suf = pre[:m], suf[:m]
	clear(pre)
	copy(pre[r:], row)
	copy(suf, pre)
	for s := 0; s < m; s += kw {
		e := min(s+kw, m)
		for j := s + 1; j < e; j++ {
			pre[j] = max(pre[j-1], pre[j])
		}
		for j := e - 2; j >= s; j-- {
			suf[j] = max(suf[j+1], suf[j])
		}
	}
	for i := range row {
		row[i] = max(suf[i], pre[i+kw-1])
	}
}

// dilateCols is dilateRow down the columns of g, h rows of w, a row at a time.
func dilateCols(g []uint8, h, w, kh int) {
	r := kh / 2
	m := h + 2*r
	pre, suf := make([]uint8, m*w), make([]uint8, m*w)
	copy(pre[r*w:], g)
	copy(suf, pre)
	for j := 1; j < m; j++ {
		if j%kh != 0 {
			cur, prev := pre[j*w:j*w+w], pre[(j-1)*w:j*w]
			for x := range cur {
				cur[x] = max(prev[x], cur[x])
			}
		}
	}
	for j := m - 2; j >= 0; j-- {
		if j%kh != kh-1 {
			cur, next := suf[j*w:j*w+w], suf[(j+1)*w:(j+2)*w]
			for x := range cur {
				cur[x] = max(next[x], cur[x])
			}
		}
	}
	for y := range h {
		out, a, b := g[y*w:y*w+w], suf[y*w:y*w+w], pre[(y+kh-1)*w:(y+kh)*w]
		for x := range out {
			out[x] = max(a[x], b[x])
		}
	}
}

// eraseRules clears long runs of ink in the crop's bottom quarter, below the
// baseline, where they form a band at most a tenth of the crop thick. A run
// is at least the crop's height long, and either at least half strong or, as
// an underline falling between two pixel rows at 1x leaves it, faint but even
// along its length. No glyph stroke there is that long, so the band is an
// underline, which the recognizers never saw and which fills the gaps between
// words. A band starting on the quarter's top row takes the rows just above
// it with the same runs, up to that thickness. Browsers break an underline at
// each descender, so its pieces can be a few letters long. Higher up, the
// headline of Devanagari and the joins of Arabic run as long as a word. A
// thicker band is the background past a button's edge, and a run reaching the
// crop's side is a table rule. The recognizers saw both in training and read
// them as nothing.
func eraseRules(x []float32, h, w int) {
	// runs adds to out the runs of at least floor ink that a rule could be:
	// at least h long, touching neither side and, when even is set, no more
	// uneven than a drawn line.
	runs := func(out [][2]int, row []float32, floor float32, even bool) [][2]int {
		for i := 0; i < len(row); {
			if row[i] < floor {
				i++
				continue
			}
			s, lo, hi := i, row[i], row[i]
			for i < len(row) && row[i] >= floor {
				lo, hi = min(lo, row[i]), max(hi, row[i])
				i++
			}
			if i-s >= h && s > 0 && i < len(row) && (!even || hi-lo <= 0.15) {
				out = append(out, [2]int{s, i})
			}
		}
		return out
	}
	// A rule is ink at least half strong, or at 1x a line that falls between
	// two rows and leaves each faint but even along its length. The faint edge
	// below Arabic joins varies letter by letter.
	spans := func(y int) [][2]int {
		row := x[y*w : y*w+w]
		return runs(runs(nil, row, 0.5, false), row, 0.25, true)
	}
	long := func(y int) bool { return len(spans(y)) > 0 }
	thin := max(2, h/10)
	top := h * 3 / 4
	for y := top; y < h; {
		if !long(y) {
			y++
			continue
		}
		s := y
		for y < h && long(y) {
			y++
		}
		if y-s > thin {
			continue
		}
		// An underline the quarter's top row cuts through starts above it, in
		// a row with the same runs.
		for s == top && s > 0 && y-s < thin && sameRuns(spans(s-1), spans(s)) {
			s--
			top = s
		}
		for r := s; r < y; r++ {
			for _, sp := range spans(r) {
				clear(x[r*w+sp[0] : r*w+sp[1]])
			}
		}
	}
}

// sameRuns reports whether a and b are the same runs, each end within 3 px.
func sameRuns(a, b [][2]int) bool {
	if len(a) == 0 || len(a) != len(b) {
		return false
	}
	for i := range a {
		if absInt(a[i][0]-b[i][0]) > 3 || absInt(a[i][1]-b[i][1]) > 3 {
			return false
		}
	}
	return true
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

func bilinear(x []float32, ih, iw, oh, ow int) []float32 {
	out := make([]float32, oh*ow)
	x0s, x1s, fxs := make([]int, ow), make([]int, ow), make([]float64, ow)
	for j := range ow {
		xs := min(max((float64(j)+0.5)*float64(iw)/float64(ow)-0.5, 0), float64(iw-1))
		x0 := int(math.Floor(xs))
		x0s[j], x1s[j], fxs[j] = x0, min(x0+1, iw-1), xs-float64(x0)
	}
	for i := range oh {
		ys := min(max((float64(i)+0.5)*float64(ih)/float64(oh)-0.5, 0), float64(ih-1))
		y0 := int(math.Floor(ys))
		y1 := min(y0+1, ih-1)
		fy := ys - float64(y0)
		r0, r1 := x[y0*iw:], x[y1*iw:]
		for j := range ow {
			fx := fxs[j]
			top := float64(r0[x0s[j]])*(1-fx) + float64(r0[x1s[j]])*fx
			bot := float64(r1[x0s[j]])*(1-fx) + float64(r1[x1s[j]])*fx
			out[i*ow+j] = float32(top*(1-fy) + bot*fy)
		}
	}
	return out
}
