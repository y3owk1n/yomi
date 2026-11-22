//go:build !amd64

package nn

// positions4 runs a tile of four output channels over runs of four positions
// from start and returns the first position it left. The 16 accumulators and
// 8 operands fit the 32 vector registers of arm64.
func positions4(wt []float32, b [4]float32, in, out []float32, start, n, nco int, relu bool) int {
	cin := len(wt) / 4
	p := start
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
			o := out[i*n+p : i*n+p+4 : i*n+p+4]
			for j := range 4 {
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

// charTile4 computes the logits of four classes, whose weights w0 to w3 and
// biases b hold, over runs of four steps, and returns the first step it left.
// z starts at the first of the four classes.
func charTile4(w0, w1, w2, w3, b, h []float32, r, n, c int, z []float32) int {
	s := 0
	for ; s+4 <= n; s += 4 {
		h0, h1, h2, h3 := h[s*r:(s+1)*r], h[(s+1)*r:(s+2)*r], h[(s+2)*r:(s+3)*r], h[(s+3)*r:(s+4)*r]
		var a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33 float32
		for k := range r {
			x0, x1, x2, x3 := h0[k], h1[k], h2[k], h3[k]
			v0, v1, v2, v3 := w0[k], w1[k], w2[k], w3[k]
			a00 += v0 * x0
			a01 += v0 * x1
			a02 += v0 * x2
			a03 += v0 * x3
			a10 += v1 * x0
			a11 += v1 * x1
			a12 += v1 * x2
			a13 += v1 * x3
			a20 += v2 * x0
			a21 += v2 * x1
			a22 += v2 * x2
			a23 += v2 * x3
			a30 += v3 * x0
			a31 += v3 * x1
			a32 += v3 * x2
			a33 += v3 * x3
		}
		z[s*c], z[s*c+1], z[s*c+2], z[s*c+3] = b[0]+a00, b[1]+a10, b[2]+a20, b[3]+a30
		z[(s+1)*c], z[(s+1)*c+1], z[(s+1)*c+2], z[(s+1)*c+3] = b[0]+a01, b[1]+a11, b[2]+a21, b[3]+a31
		z[(s+2)*c], z[(s+2)*c+1], z[(s+2)*c+2], z[(s+2)*c+3] = b[0]+a02, b[1]+a12, b[2]+a22, b[3]+a32
		z[(s+3)*c], z[(s+3)*c+1], z[(s+3)*c+2], z[(s+3)*c+3] = b[0]+a03, b[1]+a13, b[2]+a23, b[3]+a33
	}
	return s
}
