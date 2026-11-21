// Package nn runs the recognizer network: convolutions with batch norm folded
// in, depthwise and pointwise layers, and line-context steps, in plain Go.
package nn

import (
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
	w, b                   []float32
	wt                     []float32 // pointwise weights in tiles of 4 output channels: [co/4][ci][4]
}

// Model is a loaded recognizer.
type Model struct {
	Height   int    `json:"height"`
	Stride   int    `json:"stride"`
	Alphabet string `json:"alphabet"`
	Layers   []layer
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
	f := make([]float32, len(bin)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(bin[i*4:]))
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		var nw, nb int
		switch l.Op {
		case "conv":
			nw, nb = l.Cout*(l.Cin/l.Groups)*l.Kh*l.Kw, l.Cout
		case "linectx":
			nw, nb = l.C*l.C, l.C
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

// tensor is channel-major: data[c*h*w + y*w + x].
type tensor struct {
	c, h, w int
	data    []float32
}

// Scratch holds the two buffers one forward pass reuses. One per goroutine.
type Scratch struct{ a, b []float32 }

func (s *Scratch) get(which *[]float32, n int) []float32 {
	if cap(*which) < n {
		*which = make([]float32, n)
	}
	return (*which)[:n]
}

// Forward runs the network on an ink map of Height rows and returns the
// logits, T rows of len(Alphabet)+1 classes, T = width / Stride.
func (m *Model) Forward(in []float32, w int, s *Scratch) ([]float32, int, int) {
	x := tensor{1, m.Height, w, in}
	useA := true
	for i := range m.Layers {
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
			pointwise(l, x, y)
		case l.Groups == l.Cin && l.Cin == l.Cout:
			depthwise(l, x, y)
		default:
			direct(l, x, y)
		}
		x = y
	}
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
// w[co][ci] * in[ci][p]. It works in register tiles of four output channels
// by four positions, summing every input channel before one store.
func pointwise(l *layer, x, y tensor) {
	n := x.h * x.w
	cin := l.Cin
	in := x.data
	for tile := 0; tile*4 < l.Cout; tile++ {
		co := tile * 4
		wt := l.wt[tile*cin*4 : (tile+1)*cin*4]
		nco := min(4, l.Cout-co)
		var b [4]float32
		copy(b[:nco], l.b[co:co+nco])
		p := 0
		for ; p+4 <= n; p += 4 {
			a00, a01, a02, a03 := b[0], b[0], b[0], b[0]
			a10, a11, a12, a13 := b[1], b[1], b[1], b[1]
			a20, a21, a22, a23 := b[2], b[2], b[2], b[2]
			a30, a31, a32, a33 := b[3], b[3], b[3], b[3]
			for ci := range cin {
				w := wt[ci*4 : ci*4+4 : ci*4+4]
				v := in[ci*n+p : ci*n+p+4 : ci*n+p+4]
				w0, w1, w2, w3 := w[0], w[1], w[2], w[3]
				v0, v1, v2, v3 := v[0], v[1], v[2], v[3]
				a00 += w0 * v0
				a01 += w0 * v1
				a02 += w0 * v2
				a03 += w0 * v3
				a10 += w1 * v0
				a11 += w1 * v1
				a12 += w1 * v2
				a13 += w1 * v3
				a20 += w2 * v0
				a21 += w2 * v1
				a22 += w2 * v2
				a23 += w2 * v3
				a30 += w3 * v0
				a31 += w3 * v1
				a32 += w3 * v2
				a33 += w3 * v3
			}
			acc := [4][4]float32{{a00, a01, a02, a03}, {a10, a11, a12, a13}, {a20, a21, a22, a23}, {a30, a31, a32, a33}}
			for i := range nco {
				o := y.data[(co+i)*n+p : (co+i)*n+p+4 : (co+i)*n+p+4]
				for j := range 4 {
					v := acc[i][j]
					if l.Relu && v < 0 {
						v = 0
					}
					o[j] = v
				}
			}
		}
		for ; p < n; p++ {
			for i := range nco {
				v := b[i]
				for ci := range cin {
					v += wt[ci*4+i] * in[ci*n+p]
				}
				if l.Relu && v < 0 {
					v = 0
				}
				y.data[(co+i)*n+p] = v
			}
		}
	}
}

// depthwise convolves each channel with its own kernel. Each kernel tap adds
// a scaled run of an input row over the output columns it reaches, so the
// inner loops carry no bounds tests.
func depthwise(l *layer, x, y tensor) {
	for c := range l.Cout {
		in := x.data[c*x.h*x.w : (c+1)*x.h*x.w]
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
						for i, v := range src {
							dst[i] += kv * v
						}
					} else {
						for ox := ox0; ox < ox1; ox++ {
							orow[ox] += kv * irow[ox*l.Sw+off]
						}
					}
				}
			}
			if l.Relu {
				for i, v := range orow {
					if v < 0 {
						orow[i] = 0
					}
				}
			}
		}
	}
}

// direct is a plain grouped convolution, used by the first layer.
func direct(l *layer, x, y tensor) {
	cpg := l.Cin / l.Groups
	opg := l.Cout / l.Groups
	for co := range l.Cout {
		g := co / opg
		out := y.data[co*y.h*y.w : (co+1)*y.h*y.w]
		for oy := range y.h {
			for ox := range y.w {
				sum := l.b[co]
				for ci := range cpg {
					in := x.data[(g*cpg+ci)*x.h*x.w:]
					k := l.w[(co*cpg+ci)*l.Kh*l.Kw:]
					for ky := range l.Kh {
						iy := oy*l.Sh - l.Ph + ky*l.Dh
						if iy < 0 || iy >= x.h {
							continue
						}
						for kx := range l.Kw {
							ix := ox*l.Sw - l.Pw + kx*l.Dw
							if ix < 0 || ix >= x.w {
								continue
							}
							sum += k[ky*l.Kw+kx] * in[iy*x.w+ix]
						}
					}
				}
				if l.Relu {
					sum = relu(sum)
				}
				out[oy*y.w+ox] = sum
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
