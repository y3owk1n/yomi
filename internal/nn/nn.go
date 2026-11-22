// Package nn runs the recognizer network: convolutions with batch norm folded
// in, depthwise and pointwise layers, and line-context steps, in plain Go.
package nn

import (
	"slices"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
)

// layer is one step of the network as export.py writes it: a convolution
// with batch norm folded in, or a line-context step.
type layer struct {
	Op                     string `json:"op"`
	Cin, Cout, Groups      int
	Kh, Kw, Sh, Sw, Ph, Pw int
	Dh, Dw                 int
	Relu                   bool
	C                      int
	WOff                   int `json:"w"`
	BOff                   int `json:"b"`
	Rank, Classes          int // gated head: projection rank and character count
	PWOff                  int `json:"proj_w"`
	PBOff                  int `json:"proj_b"`
	CWOff                  int `json:"char_w"`
	CBOff                  int `json:"char_b"`
	pw, pb, cw, cb         []float32
	w, b                   []float32
	wt                     []float32 // pointwise weights in tiles of 4 output channels: [co/4][ci][4]
	pwt                    []float32 // gated projection weights, tiled as wt
}

// Model is a loaded recognizer.
type Model struct {
	Height   int    `json:"height"`
	Stride   int    `json:"stride"`
	Alphabet string `json:"alphabet"`
	// Family names the scripts a recognizer reads. Labels names a router's
	// classes, the families it tells apart.
	Family string   `json:"family"`
	Labels []string `json:"labels"`
	// DType is how weights.bin stores each weight: "f16", IEEE half
	// precision, or float32 when empty.
	DType  string `json:"dtype"`
	Layers []layer
}

// Load reads model.json and weights.bin from fsys.
func Load(fsys fs.FS) (*Model, error) {
	raw, err := fs.ReadFile(fsys, "model.json")
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("model.json: %w", err)
	}
	bin, err := fs.ReadFile(fsys, "weights.bin")
	if err != nil {
		return nil, err
	}
	var f []float32
	switch m.DType {
	case "":
		f = make([]float32, len(bin)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(bin[i*4:]))
		}
	case "f16":
		f = make([]float32, len(bin)/2)
		for i := range f {
			f[i] = halfToFloat(binary.LittleEndian.Uint16(bin[i*2:]))
		}
	default:
		return nil, fmt.Errorf("model.json: unknown dtype %q", m.DType)
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		var nw, nb int
		switch l.Op {
		case "conv":
			nw, nb = l.Cout*(l.Cin/l.Groups)*l.Kh*l.Kw, l.Cout
		case "linectx":
			nw, nb = l.C*l.C, l.C
		case "gated":
			nw, nb = l.C, 1
			np, nc := l.Rank*l.C, l.Classes*l.Rank
			if l.PWOff+np > len(f) || l.CWOff+nc > len(f) || l.CBOff+l.Classes > len(f) {
				return nil, fmt.Errorf("layer %d reaches past weights.bin", i)
			}
			l.pw, l.pb = f[l.PWOff:l.PWOff+np], f[l.PBOff:l.PBOff+l.Rank]
			l.cw, l.cb = f[l.CWOff:l.CWOff+nc], f[l.CBOff:l.CBOff+l.Classes]
			l.pwt = tile4(l.pw, l.Rank, l.C)
		default:
			return nil, fmt.Errorf("unknown op %q", l.Op)
		}
		if l.WOff+nw > len(f) || l.BOff+nb > len(f) {
			return nil, fmt.Errorf("layer %d reaches past weights.bin", i)
		}
		l.w, l.b = f[l.WOff:l.WOff+nw], f[l.BOff:l.BOff+nb]
		if l.Op == "conv" && l.Kh == 1 && l.Kw == 1 && l.Groups == 1 {
			l.wt = tile4(l.w, l.Cout, l.Cin)
		}
	}
	return &m, nil
}

// halfToFloat widens an IEEE 754 half-precision number.
func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal: normalize it
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (frac&0x3ff)<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
}

// tensor is channel-major: data[c*h*w + y*w + x].
type tensor struct {
	c, h, w int
	data    []float32
}

// Scratch holds the two buffers one forward pass reuses. One per goroutine.
type Scratch struct{ a, b, tile, planes []float32 }

func (s *Scratch) get(which *[]float32, n int) []float32 {
	if cap(*which) < n {
		*which = make([]float32, n)
	}
	return (*which)[:n]
}

// Forward runs the network on an ink map of Height rows and returns the
// logits, T rows of len(Alphabet)+1 classes, T = width / Stride.
func (m *Model) Forward(in []float32, w int, s *Scratch) ([]float32, int, int) {
	x := m.run(in, w, s, len(m.Layers))
	// x is (classes, 1, T): return it as T rows of classes.
	t, c := x.w, x.c
	out := make([]float32, t*c)
	for k := range c {
		for j := range t {
			out[j*c+k] = x.data[k*t+j]
		}
	}
	return out, t, c
}

// Step is one time step's most likely class (0 is the CTC blank) and its
// probability.
type Step struct {
	Class int
	Prob  float32
}

// Steps runs the network and returns each time step's most likely class. A
// gated head computes character logits only at the steps its blank logit
// leaves open.
func (m *Model) Steps(in []float32, w int, s *Scratch) []Step {
	last := &m.Layers[len(m.Layers)-1]
	if last.Op != "gated" {
		// x is (classes, 1, T). Walking it a class row at a time reads memory
		// in order, where a step's logits lie T apart.
		x := m.run(in, w, s, len(m.Layers))
		t, c := x.w, x.c
		steps := make([]Step, t)
		top := slices.Clone(x.data[:t])
		for k := 1; k < c; k++ {
			for j, v := range x.data[k*t : (k+1)*t] {
				if v > top[j] {
					top[j], steps[j].Class = v, k
				}
			}
		}
		// The softmax of each step's best class. Classes more than 20 below
		// it add under exp(-20) each, so the sum skips them.
		sum := make([]float64, t)
		for k := range c {
			for j, v := range x.data[k*t : (k+1)*t] {
				if v > top[j]-20 {
					sum[j] += math.Exp(float64(v - top[j]))
				}
			}
		}
		for j := range steps {
			steps[j].Prob = float32(1 / sum[j])
		}
		return steps
	}
	return gated(last, m.run(in, w, s, len(m.Layers)-1))
}

// gated applies a gated head to features x (C channels, one row, T steps).
// The steps it opens share one pass over the character weights.
func gated(l *layer, x tensor) []Step {
	t, c := x.w, x.c
	steps := make([]Step, t)
	var open []int
	f := make([]float32, c)
	for j := range t {
		for k := range c {
			f[k] = x.data[k*t+j]
		}
		b := l.b[0]
		for k, v := range f {
			b += l.w[k] * v
		}
		if b >= 0 {
			steps[j] = Step{0, sigmoid(b)}
			continue
		}
		steps[j] = Step{0, 1 - sigmoid(b)} // the probability of not blank, for now
		open = append(open, j)
	}
	if len(open) == 0 {
		return steps
	}
	n := len(open)
	z := charLogits(l, project(l, x, open), n)
	arg := make([]int, n)
	sum := make([]float64, n)
	for s := range n {
		row := z[s*l.Classes : (s+1)*l.Classes]
		best := 0
		for ci, v := range row {
			if v > row[best] {
				best = ci
			}
		}
		// Classes more than 20 below the best add under exp(-20) each, under
		// 3e-5 together even for twelve thousand classes, so the softmax sum
		// skips them.
		top, floor := row[best], row[best]-20
		var total float64
		for _, v := range row {
			if v > floor {
				total += math.Exp(float64(v - top))
			}
		}
		arg[s], sum[s] = best, total
	}
	for s, j := range open {
		p := float32(1/sum[s]) * steps[j].Prob // softmax of the best class times P(not blank)
		pBlank := 1 - steps[j].Prob
		if pBlank >= p {
			steps[j] = Step{0, pBlank}
		} else {
			steps[j] = Step{arg[s] + 1, p}
		}
	}
	return steps
}

// charLogits computes the character logits of n open steps, whose
// projections h holds (Rank each): z[s*Classes+c] = cb[c] + cw[c] . h[s]. For
// each run of four classes, charTile works through the steps its register
// tile covers, reading each class's weights once per tile. Steps left over go
// one at a time.
// project returns the projections of x's steps in open, Rank each. The steps
// are the positions of one pointwise convolution, so the projection runs in
// the same kernel.
func project(l *layer, x tensor, open []int) []float32 {
	t, c, n := x.w, x.c, len(open)
	in := make([]float32, c*n)
	for k := range c {
		row := x.data[k*t : (k+1)*t]
		for s, j := range open {
			in[k*n+s] = row[j]
		}
	}
	out := make([]float32, l.Rank*n)
	matmul(l.pwt, l.pb, in, out, c, l.Rank, n, false, nil)
	h := make([]float32, n*l.Rank)
	for r := range l.Rank {
		for s := range n {
			h[s*l.Rank+r] = out[r*n+s]
		}
	}
	return h
}

func charLogits(l *layer, h []float32, n int) []float32 {
	r, c := l.Rank, l.Classes
	// Steps pad to a multiple of four with zero projections, so every step
	// goes through charTile, and the padding's logits are cut off at the end.
	n4 := (n + 3) &^ 3
	if n4 != n {
		h = append(h[:n*r:n*r], make([]float32, (n4-n)*r)...)
	}
	z := make([]float32, n4*c)
	steps := n
	n = n4
	ci := 0
	for ; ci+4 <= c; ci += 4 {
		w0, w1, w2, w3 := l.cw[ci*r:(ci+1)*r], l.cw[(ci+1)*r:(ci+2)*r], l.cw[(ci+2)*r:(ci+3)*r], l.cw[(ci+3)*r:(ci+4)*r]
		s := charTile(w0, w1, w2, w3, l.cb[ci:ci+4], h, r, n, c, z[ci:])
		for ; s < n; s++ {
			hs := h[s*r : (s+1)*r]
			for i, w := range [][]float32{w0, w1, w2, w3} {
				v := l.cb[ci+i]
				for k, x := range hs {
					v += w[k] * x
				}
				z[s*c+ci+i] = v
			}
		}
	}
	for ; ci < c; ci++ {
		w := l.cw[ci*r : (ci+1)*r]
		for s := range n {
			v := l.cb[ci]
			for k, x := range h[s*r : (s+1)*r] {
				v += w[k] * x
			}
			z[s*c+ci] = v
		}
	}
	return z[:steps*c]
}

func sigmoid(v float32) float32 { return float32(1 / (1 + math.Exp(-float64(v)))) }

// run applies the first n layers to an ink map of Height rows.
func (m *Model) run(in []float32, w int, s *Scratch, n int) tensor {
	x := tensor{1, m.Height, w, in}
	useA := true
	for i := range n {
		l := &m.Layers[i]
		if l.Op == "linectx" {
			lineContext(l, x)
			continue
		}
		oh := (x.h+2*l.Ph-l.Dh*(l.Kh-1)-1)/l.Sh + 1
		ow := (x.w+2*l.Pw-l.Dw*(l.Kw-1)-1)/l.Sw + 1
		var buf []float32
		if useA {
			buf = s.get(&s.a, l.Cout*oh*ow)
		} else {
			buf = s.get(&s.b, l.Cout*oh*ow)
		}
		useA = !useA
		y := tensor{l.Cout, oh, ow, buf}
		switch {
		case l.Kh == 1 && l.Kw == 1 && l.Groups == 1 && l.Sh == 1 && l.Sw == 1:
			pointwise(l, x, y, s)
		case l.Groups == l.Cin && l.Cin == l.Cout:
			depthwise(l, x, y, s)
		default:
			direct(l, x, y)
		}
		x = y
	}
	return x
}

func relu(v float32) float32 {
	if v < 0 {
		return 0
	}
	return v
}

// tile4 rearranges a (cout, cin) weight matrix into tiles of four output
// channels, [co/4][ci][4], so the pointwise kernel reads four weights at
// once. The last tile is zero padded.
func tile4(w []float32, cout, cin int) []float32 {
	tiles := (cout + 3) / 4
	t := make([]float32, tiles*cin*4)
	for co := range cout {
		for ci := range cin {
			t[(co/4)*cin*4+ci*4+co%4] = w[co*cin+ci]
		}
	}
	return t
}

// pointwise is a 1x1 convolution: out[co][p] = b[co] + sum over ci of
// w[co][ci] * in[ci][p]. For each tile of four output channels, positions
// works through the positions its register tile covers, summing every input
// channel before one store. Positions left over go one at a time.
func pointwise(l *layer, x, y tensor, s *Scratch) {
	matmul(l.wt, l.b, x.data, y.data, l.Cin, l.Cout, x.h*x.w, l.Relu, s)
}

// matmul writes out[co*n+p] = bias[co] + sum over ci of w[co][ci] * in[ci*n+p]
// for cout channels and n positions, w tiled by tile4, clamped at zero when
// relu is set. s, when not nil, lends the buffer a short last tile runs into.
func matmul(wt, bias, in, out []float32, cin, cout, n int, relu bool, s *Scratch) {
	run := blockRun()
	if run == 0 || n < run || cin == 0 || cout < 4 {
		matmulRows(wt, bias, in, out, cin, cout, n, relu, s)
		return
	}
	// Positions in blocks of run, every tile over one block before the next,
	// keep the block's inputs in L1 where whole rows stream from L2 once per
	// tile. The last block ends at n, overlapping the one before.
	r := 0
	if relu {
		r = 1
	}
	full := cout &^ 3
	for p0 := 0; p0 < n; p0 += run {
		q := min(p0, n-run)
		for co := 0; co < full; co += 4 {
			b := (*[4]float32)(bias[co : co+4])
			runBlock(wt[co*cin:(co+4)*cin], b, in[q:], out[co*n+q:], cin, n, r)
		}
	}
	if full < cout {
		matmulRows(wt[full*cin:], bias[full:], in, out[full*n:], cin, cout-full, n, relu, s)
	}
}

// matmulRows is matmul a tile at a time over whole rows.
func matmulRows(wt, bias, in, out []float32, cin, cout, n int, relu bool, s *Scratch) {
	for tile := 0; tile*4 < cout; tile++ {
		co := tile * 4
		w := wt[tile*cin*4 : (tile+1)*cin*4]
		nco := min(4, cout-co)
		var b [4]float32
		copy(b[:nco], bias[co:co+nco])
		if nco < 4 {
			// The last tile of a width not a multiple of four, as a head's
			// classes often are. Its weights pad with zeros, so the vector
			// kernel runs it whole into four rows and the real ones are kept.
			var tmp []float32
			if s != nil {
				tmp = s.get(&s.tile, 4*n)
			} else {
				tmp = make([]float32, 4*n)
			}
			if positions(w, b, in, tmp, n, 4, relu) == n {
				copy(out[co*n:], tmp[:nco*n])
				continue
			}
		}
		p := positions(w, b, in, out[co*n:], n, nco, relu)
		for ; p < n; p++ {
			for i := range nco {
				v := b[i]
				for ci := range cin {
					v += w[ci*4+i] * in[ci*n+p]
				}
				if relu && v < 0 {
					v = 0
				}
				out[(co+i)*n+p] = v
			}
		}
	}
}

// depthwise convolves each channel with its own kernel. Each kernel tap adds
// a scaled run of an input row over the output columns it reaches, so the
// inner loops carry no bounds tests.
// addScaled adds k times src to dst, as far as the shorter runs.
func addScaled(dst, src []float32, k float32) {
	n := min(len(dst), len(src))
	for i := axpyVec(dst, src, k); i < n; i++ {
		dst[i] += k * src[i]
	}
}

func depthwise(l *layer, x, y tensor, s *Scratch) {
	// At a stride of two, a tap reads every other column. Each channel splits
	// into its even and odd columns first, so every tap reads a run.
	var even, odd []float32
	we, wo := (x.w+1)/2, x.w/2
	if l.Sw == 2 {
		buf := s.get(&s.planes, x.h*(we+wo))
		even, odd = buf[:x.h*we], buf[x.h*we:]
	}
	for c := range l.Cout {
		in := x.data[c*x.h*x.w : (c+1)*x.h*x.w]
		if l.Sw == 2 {
			for iy := range x.h {
				row := in[iy*x.w : (iy+1)*x.w]
				for i := range we {
					even[iy*we+i] = row[2*i]
				}
				for i := range wo {
					odd[iy*wo+i] = row[2*i+1]
				}
			}
		}
		out := y.data[c*y.h*y.w : (c+1)*y.h*y.w]
		k := l.w[c*l.Kh*l.Kw : (c+1)*l.Kh*l.Kw]
		bias := l.b[c]
		for oy := range y.h {
			orow := out[oy*y.w : (oy+1)*y.w]
			for i := range orow {
				orow[i] = bias
			}
			for ky := range l.Kh {
				iy := oy*l.Sh - l.Ph + ky*l.Dh
				if iy < 0 || iy >= x.h {
					continue
				}
				irow := in[iy*x.w : (iy+1)*x.w]
				for kx := range l.Kw {
					kv := k[ky*l.Kw+kx]
					off := kx*l.Dw - l.Pw // ix = ox*Sw + off
					// valid ox: 0 <= ox*Sw+off < x.w
					ox0 := 0
					if off < 0 {
						ox0 = (-off + l.Sw - 1) / l.Sw
					}
					ox1 := min(y.w, (x.w-1-off)/l.Sw+1)
					if ox1 <= ox0 {
						continue
					}
					if l.Sw == 1 {
						src := irow[ox0+off : ox1+off]
						dst := orow[ox0:ox1]
						addScaled(dst, src, kv)
					} else if l.Sw == 2 {
						// ix = 2*(ox+base) + parity, a column of one plane.
						parity := off & 1
						base := (off - parity) / 2
						plane, pw := even, we
						if parity == 1 {
							plane, pw = odd, wo
						}
						src := plane[iy*pw+ox0+base : iy*pw+ox1+base]
						dst := orow[ox0:ox1]
						addScaled(dst, src, kv)
					} else {
						for ox := ox0; ox < ox1; ox++ {
							orow[ox] += kv * irow[ox*l.Sw+off]
						}
					}
				}
			}
			if l.Relu {
				reluRow(orow)
			}
		}
	}
}

// direct is a plain grouped convolution, used by the first layer. For each
// kernel tap it gathers the input row the tap reads, at the layer's stride,
// once, and adds it into every output channel's row scaled by the tap's
// weight, so the inner loop is a run of multiply-adds with no bounds tests.
func direct(l *layer, x, y tensor) {
	cpg := l.Cin / l.Groups
	opg := l.Cout / l.Groups
	tap := make([]float32, y.w)
	for oy := range y.h {
		for co := range l.Cout {
			row := y.data[(co*y.h+oy)*y.w : (co*y.h+oy+1)*y.w]
			for i := range row {
				row[i] = l.b[co]
			}
		}
		for g := range l.Groups {
			for ci := range cpg {
				in := x.data[(g*cpg+ci)*x.h*x.w : (g*cpg+ci+1)*x.h*x.w]
				for ky := range l.Kh {
					iy := oy*l.Sh - l.Ph + ky*l.Dh
					if iy < 0 || iy >= x.h {
						continue
					}
					irow := in[iy*x.w : (iy+1)*x.w]
					for kx := range l.Kw {
						off := kx*l.Dw - l.Pw
						for ox := range tap {
							tap[ox] = 0
							if ix := ox*l.Sw + off; ix >= 0 && ix < x.w {
								tap[ox] = irow[ix]
							}
						}
						for co := g * opg; co < (g+1)*opg; co++ {
							k := l.w[((co*cpg+ci)*l.Kh+ky)*l.Kw+kx]
							row := y.data[(co*y.h+oy)*y.w : (co*y.h+oy+1)*y.w]
							addScaled(row, tap, k)
						}
					}
				}
			}
		}
		if l.Relu {
			for co := range l.Cout {
				reluRow(y.data[(co*y.h+oy)*y.w : (co*y.h+oy+1)*y.w])
			}
		}
	}
}

// lineContext adds relu(W * mean + b) to every step, the mean taken over the
// line (x has one row here).
func lineContext(l *layer, x tensor) {
	n := x.h * x.w
	mean := make([]float32, l.C)
	for c := range l.C {
		var s float32
		for _, v := range x.data[c*n : (c+1)*n] {
			s += v
		}
		mean[c] = s / float32(n)
	}
	for c := range l.C {
		v := l.b[c]
		for k := range l.C {
			v += l.w[c*l.C+k] * mean[k]
		}
		v = relu(v)
		row := x.data[c*n : (c+1)*n]
		for p := range row {
			row[p] += v
		}
	}
}
