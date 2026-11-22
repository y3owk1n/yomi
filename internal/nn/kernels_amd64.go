package nn

// hasAVX2 reports whether this CPU and its OS run AVX2 and FMA, which the
// kernels below need. Without them every kernel takes its scalar version.
var hasAVX2 = cpuHasAVX2()

// positions runs a tile of four output channels over runs of sixteen
// positions with AVX2 (pointwise4x16, kernels_amd64.s), eight lanes per
// instruction, and returns the first position it left. The last run ends at
// n, overlapping the one before, since a scalar tail costs as much as the
// vector work on a short line. A tile short of four channels, the last of a
// layer whose width is not a multiple of four, or a line under sixteen
// positions, takes the scalar kernel throughout.
func positions(wt []float32, b [4]float32, in, out []float32, n, nco int, relu bool) int {
	cin := len(wt) / 4
	end := n &^ 15
	if !hasAVX2 || nco < 4 || end == 0 || cin == 0 {
		return positions2(wt, b, in, out, 0, n, nco, relu)
	}
	r := 0
	if relu {
		r = 1
	}
	pointwise4x16(&wt[0], &in[0], &out[0], &b, cin, n, end, r)
	if end < n {
		pointwise4x16(&wt[0], &in[n-16], &out[n-16], &b, cin, n, 16, r)
	}
	return n
}

// pointwise4x16 writes out[co*n+p] = b[co] + sum over ci of wt[ci*4+co] *
// in[ci*n+p] for the four channels co and every p below end, sixteen at a
// time, clamped at zero when relu is 1.
//
//go:noescape
func pointwise4x16(wt, in, out *float32, b *[4]float32, cin, n, end, relu int)

// charTile computes the logits of four classes over pairs of steps with AVX2
// (dot4x2), eight ranks per instruction, and returns the first step it left.
// z starts at the first of the four classes. A rank not a multiple of eight
// takes the scalar kernel.
func charTile(w0, w1, w2, w3, b, h []float32, r, n, c int, z []float32) int {
	if !hasAVX2 || r == 0 || r%8 != 0 {
		return charTile2(w0, w1, w2, w3, b, h, r, n, c, z)
	}
	bias := (*[4]float32)(b)
	s := 0
	for ; s+2 <= n; s += 2 {
		dot4x2(&w0[0], &w1[0], &w2[0], &w3[0], &h[s*r], &h[(s+1)*r], r, bias, &z[s*c:][:4][0], &z[(s+1)*c:][:4][0])
	}
	return s
}

// dot4x2 writes, for each step row h of length r, the dot products with the
// four weight rows w plus their biases b to the four floats at that step's z.
//
//go:noescape
func dot4x2(w0, w1, w2, w3, h0, h1 *float32, r int, b *[4]float32, z0, z1 *float32)

// axpyVec adds k times src to dst for the leading multiple of eight of their
// length with AVX2 and returns how many it did.
func axpyVec(dst, src []float32, k float32) int {
	m := min(len(dst), len(src)) &^ 7
	if !hasAVX2 || m == 0 {
		return 0
	}
	axpy(&dst[0], &src[0], m, k)
	return m
}

// axpy adds k times src[:n] to dst[:n], n a multiple of eight, 32 at a time
// and then eight.
//
//go:noescape
func axpy(dst, src *float32, n int, k float32)

// reluRow clamps row at zero.
func reluRow(row []float32) {
	m := len(row) &^ 31
	if hasAVX2 && m > 0 {
		relu32(&row[0], m)
	} else {
		m = 0
	}
	for i := m; i < len(row); i++ {
		row[i] = max(row[i], 0)
	}
}

// relu32 clamps p[:n] at zero, n a multiple of 32.
//
//go:noescape
func relu32(p *float32, n int)

// cpuHasAVX2 reads CPUID and XGETBV: AVX2 and FMA in the CPU, and the OS
// saving the YMM registers across a switch.
func cpuHasAVX2() bool {
	_, _, c1, _ := cpuid(1, 0)
	const fma, osxsave, avx = 1 << 12, 1 << 27, 1 << 28
	if c1&fma == 0 || c1&osxsave == 0 || c1&avx == 0 {
		return false
	}
	if lo, _ := xgetbv(); lo&6 != 6 {
		return false
	}
	_, b7, _, _ := cpuid(7, 0)
	return b7&(1<<5) != 0
}

func cpuid(leaf, sub uint32) (a, b, c, d uint32)

func xgetbv() (lo, hi uint32)
