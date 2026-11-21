// Package latin is yomi's recognizer for printable ASCII: English letters,
// digits and punctuation. Its weights (0.6 MB) are embedded.
package latin

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the Latin recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
