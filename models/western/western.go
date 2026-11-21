// Package western is yomi's recognizer for Western scripts: Latin with the
// accented and extended letters of the European languages, Turkish and
// Vietnamese, Cyrillic, monotonic Greek, and the signs UI text uses. Its
// weights (0.9 MB) are embedded.
package western

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the Western recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
