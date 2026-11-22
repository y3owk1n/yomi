// Package prep turns a line crop into the recognizer's input: an ink map
// Height rows tall, the same way train/prep.py does.
package prep

import (
	"image"
	"math"
)

// bgBand is how far either side of a luma the background's band reaches.
const bgBand = 8

// Prepare reproduces train/prep.py: integer luma, ink strength against the
// crop's commonest luma, underlines erased, integer box downsample when the
// crop is at least twice Height, then bilinear (half-pixel centres, edge
// clamp) to Height rows.
func Prepare(img *image.RGBA, r image.Rectangle, height int) ([]float32, int) {
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
	span := float32(max(32, abs32(bg-far)))
	x := make([]float32, h*w)
	for i, v := range g {
		x[i] = min(1, max(0, (float32(bg)-float32(v))*sign/span))
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
