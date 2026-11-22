package nn

// positions runs a tile of four output channels over pairs of positions and
// returns the first position it left. Go leaves 15 vector registers for
// floats on amd64, since X15 holds zero, so this tile keeps eight
// accumulators. The 16 of the four-position tile spill to the stack on every
// input channel there.
func positions(wt []float32, b [4]float32, in, out []float32, n, nco int, relu bool) int {
	cin := len(wt) / 4
	p := 0
	for ; p+2 <= n; p += 2 {
		a00, a01 := b[0], b[0]
		a10, a11 := b[1], b[1]
		a20, a21 := b[2], b[2]
		a30, a31 := b[3], b[3]
		for ci := range cin {
			w := wt[ci*4 : ci*4+4 : ci*4+4]
			v := in[ci*n+p : ci*n+p+2 : ci*n+p+2]
			v0, v1 := v[0], v[1]
			a00 += w[0] * v0
			a01 += w[0] * v1
			a10 += w[1] * v0
			a11 += w[1] * v1
			a20 += w[2] * v0
			a21 += w[2] * v1
			a30 += w[3] * v0
			a31 += w[3] * v1
		}
		acc := [4][2]float32{{a00, a01}, {a10, a11}, {a20, a21}, {a30, a31}}
		for i := range nco {
			o := out[i*n+p : i*n+p+2 : i*n+p+2]
			for j := range 2 {
				v := acc[i][j]
				if relu && v < 0 {
					v = 0
				}
				o[j] = v
			}
		}
	}
	return p
}

// charTile computes the logits of four classes, whose weights w0 to w3 and
// biases b hold, over pairs of steps, and returns the first step it left. z
// starts at the first of the four classes. Like positions, it keeps eight
// accumulators so they stay in registers.
func charTile(w0, w1, w2, w3, b, h []float32, r, n, c int, z []float32) int {
	s := 0
	for ; s+2 <= n; s += 2 {
		h0, h1 := h[s*r:(s+1)*r], h[(s+1)*r:(s+2)*r]
		var a00, a01, a10, a11, a20, a21, a30, a31 float32
		for k := range r {
			x0, x1 := h0[k], h1[k]
			v0, v1, v2, v3 := w0[k], w1[k], w2[k], w3[k]
			a00 += v0 * x0
			a01 += v0 * x1
			a10 += v1 * x0
			a11 += v1 * x1
			a20 += v2 * x0
			a21 += v2 * x1
			a30 += v3 * x0
			a31 += v3 * x1
		}
		z[s*c], z[s*c+1], z[s*c+2], z[s*c+3] = b[0]+a00, b[1]+a10, b[2]+a20, b[3]+a30
		z[(s+1)*c], z[(s+1)*c+1], z[(s+1)*c+2], z[(s+1)*c+3] = b[0]+a01, b[1]+a11, b[2]+a21, b[3]+a31
	}
	return s
}
