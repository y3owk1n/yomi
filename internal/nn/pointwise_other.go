//go:build !amd64

package nn

// positions runs a tile of four output channels over runs of four positions
// and returns the first position it left. The 16 accumulators and 8 operands
// fit the 32 vector registers of arm64.
func positions(wt []float32, b [4]float32, in, out []float32, n, nco int, relu bool) int {
	cin := len(wt) / 4
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
