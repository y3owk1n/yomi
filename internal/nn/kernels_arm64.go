package nn

// positions runs a tile of four output channels over runs of sixteen
// positions with NEON (pointwise4x16, kernels_arm64.s), or eight on a line
// under sixteen (pointwise4x8), four lanes per instruction, and
// returns the first position it left. The last run ends at n, overlapping the
// one before, since a scalar tail costs as much as a third of the vector
// work on a short line. A tile short of four channels, the last of a layer
// whose width is not a multiple of four, or a line under eight positions,
// takes the scalar kernel throughout.
func positions(wt []float32, b [4]float32, in, out []float32, n, nco int, relu bool) int {
	cin := len(wt) / 4
	if nco < 4 || n < 8 || cin == 0 {
		return positions4(wt, b, in, out, 0, n, nco, relu)
	}
	r := 0
	if relu {
		r = 1
	}
	kernel, run := pointwise4x16, 16
	if n < 16 {
		kernel, run = pointwise4x8, 8
	}
	end := n / run * run
	kernel(&wt[0], &in[0], &out[0], &b, cin, n, end, r)
	if end < n {
		kernel(&wt[0], &in[n-run], &out[n-run], &b, cin, n, run, r)
	}
	return n
}

// blockRun is how many positions one vector kernel call covers in matmul's
// blocked loop.
func blockRun() int { return 16 }

// runBlock computes the sixteen positions at in and out of a tile of four
// output channels, rows n apart.
func runBlock(wt []float32, b *[4]float32, in, out []float32, cin, n, relu int) {
	pointwise4x16(&wt[0], &in[0], &out[0], b, cin, n, 16, relu)
}

// pointwise4x16 is pointwise4x8 sixteen positions at a time. Its sixteen
// accumulators keep enough multiply-adds in flight to hide their latency,
// where eight leave the core waiting half the time.
//
//go:noescape
func pointwise4x16(wt, in, out *float32, b *[4]float32, cin, n, end, relu int)

// pointwise4x8 writes out[co*n+p] = b[co] + sum over ci of wt[ci*4+co] *
// in[ci*n+p] for the four channels co and every p below end, eight at a time,
// clamped at zero when relu is 1.
//
//go:noescape
func pointwise4x8(wt, in, out *float32, b *[4]float32, cin, n, end, relu int)

// charTile computes the logits of four classes over runs of four steps with
// NEON (dot4x4), four ranks per instruction, and returns the first step it
// left. z starts at the first of the four classes. A rank not a multiple of
// four takes the scalar kernel.
func charTile(w0, w1, w2, w3, b, h []float32, r, n, c int, z []float32) int {
	if r == 0 || r%4 != 0 {
		return charTile4(w0, w1, w2, w3, b, h, r, n, c, z)
	}
	bias := (*[4]float32)(b)
	s := 0
	for ; s+4 <= n; s += 4 {
		dot4x4(&w0[0], &w1[0], &w2[0], &w3[0], &h[s*r], &h[(s+1)*r], &h[(s+2)*r], &h[(s+3)*r], r, bias,
			&z[s*c:][:4][0], &z[(s+1)*c:][:4][0], &z[(s+2)*c:][:4][0], &z[(s+3)*c:][:4][0])
	}
	return s
}

// dot4x4 writes, for each step row h of length r, the dot products with the
// four weight rows w plus their biases b to the four floats at that step's z.
//
//go:noescape
func dot4x4(w0, w1, w2, w3, h0, h1, h2, h3 *float32, r int, b *[4]float32, z0, z1, z2, z3 *float32)

// axpyVec adds k times src to dst for the leading multiple of four of their
// length with NEON and returns how many it did.
func axpyVec(dst, src []float32, k float32) int {
	m := min(len(dst), len(src)) &^ 3
	if m > 0 {
		axpy(&dst[0], &src[0], m, k)
	}
	return m
}

// axpy adds k times src[:n] to dst[:n], n a multiple of four, sixteen at a
// time and then four.
//
//go:noescape
func axpy(dst, src *float32, n int, k float32)

// reluRow clamps row at zero.
func reluRow(row []float32) {
	m := len(row) &^ 15
	if m > 0 {
		relu16(&row[0], m)
	}
	for i := m; i < len(row); i++ {
		row[i] = max(row[i], 0)
	}
}

// relu16 clamps p[:n] at zero, n a multiple of 16.
//
//go:noescape
func relu16(p *float32, n int)
