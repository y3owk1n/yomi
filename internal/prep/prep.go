// Package prep turns a line crop into the recognizer's input: an ink map
// Height rows tall, the same way train/prep.py does.
package prep

import (
	"image"
	"math"
	"sort"
)

// Prepare reproduces train/prep.py: integer luma, ink strength against the border
// median, rules erased, integer box downsample when the crop is at least
// twice Height, then bilinear (half-pixel centres, edge clamp) to Height rows.
func Prepare(img *image.RGBA, r image.Rectangle, height int) ([]float32, int) {
	r = r.Intersect(img.Rect)
	h, w := r.Dy(), r.Dx()
	if h < 2 || w < 2 {
		return nil, 0
	}
	g := make([]int32, h*w)
	for y := range h {
		row := img.Pix[(r.Min.Y+y-img.Rect.Min.Y)*img.Stride:]
		for x := range w {
			p := row[(r.Min.X+x-img.Rect.Min.X)*4:]
			g[y*w+x] = (299*int32(p[0]) + 587*int32(p[1]) + 114*int32(p[2])) / 1000
		}
	}
	// Border median, as numpy's median: the mean of the two middle values
	// for an even count, then truncated to int.
	border := make([]int32, 0, 2*w+2*h)
	border = append(border, g[:w]...)
	border = append(border, g[(h-1)*w:]...)
	for y := 1; y < h-1; y++ {
		border = append(border, g[y*w], g[y*w+w-1])
	}
	sort.Slice(border, func(i, j int) bool { return border[i] < border[j] })
	n := len(border)
	var bg int32
	if n%2 == 1 {
		bg = border[n/2]
	} else {
		bg = int32((float64(border[n/2-1]) + float64(border[n/2])) / 2)
	}
	lo, hi := g[0], g[0]
	for _, v := range g {
		lo, hi = min(lo, v), max(hi, v)
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

// eraseRules clears runs of ink at least half strong and at least the
// crop's height long in the crop's bottom quarter, below the baseline, where
// they form a band at most a tenth of the crop thick. No glyph stroke there is
// that long, so the band is an underline, which the recognizers never saw and
// which fills the gaps between words. Browsers break an underline at each
// descender, so its pieces can be a few letters long. Higher up, the headline
// of Devanagari and the joins of Arabic run as long as a word. A thicker band
// is the background past a button's edge, and a run reaching the crop's side
// is a table rule. The recognizers saw both in training and read them as
// nothing.
func eraseRules(x []float32, h, w int) {
	long := func(y int, fn func(s, e int)) bool {
		row, found := x[y*w:y*w+w], false
		for i := 0; i < w; {
			if row[i] < 0.5 {
				i++
				continue
			}
			s := i
			for i < w && row[i] >= 0.5 {
				i++
			}
			if i-s >= h && s > 0 && i < w {
				found = true
				if fn != nil {
					fn(s, i)
				}
			}
		}
		return found
	}
	thin := max(2, h/10)
	for y := h * 3 / 4; y < h; {
		if !long(y, nil) {
			y++
			continue
		}
		s := y
		for y < h && long(y, nil) {
			y++
		}
		if y-s > thin {
			continue
		}
		for r := s; r < y; r++ {
			long(r, func(a, b int) { clear(x[r*w+a : r*w+b]) })
		}
	}
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
