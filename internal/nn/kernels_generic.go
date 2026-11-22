//go:build !amd64 && !arm64

package nn

func positions(wt []float32, b [4]float32, in, out []float32, n, nco int, relu bool) int {
	return positions4(wt, b, in, out, 0, n, nco, relu)
}

func charTile(w0, w1, w2, w3, b, h []float32, r, n, c int, z []float32) int {
	return charTile4(w0, w1, w2, w3, b, h, r, n, c, z)
}
