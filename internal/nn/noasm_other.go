//go:build !arm64 && !amd64

package nn

// axpyVec adds k times src to dst for the prefix the vector kernel takes and
// returns how many it did. Only arm64 and amd64 have vector kernels for it,
// so here it does none.
func axpyVec(_, _ []float32, _ float32) int { return 0 }

// reluRow clamps row at zero.
func reluRow(row []float32) {
	for i, v := range row {
		row[i] = max(v, 0)
	}
}
