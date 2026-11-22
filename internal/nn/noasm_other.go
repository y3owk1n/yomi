//go:build !arm64 && !amd64

package nn

// axpy16 adds k times src to dst for the leading multiple of 16 of their
// length and returns how many it did. Only arm64 and amd64 have vector
// kernels for it, so here it does none.
func axpy16(_, _ []float32, _ float32) int { return 0 }

// reluRow clamps row at zero.
func reluRow(row []float32) {
	for i, v := range row {
		row[i] = max(v, 0)
	}
}
