// Package sea is yomi's recognizer for Thai, Lao, Khmer and Myanmar, and
// ASCII. Its weights (0.9 MB) are embedded.
package sea

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the Southeast Asian recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
