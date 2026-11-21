// Package rtl is yomi's recognizer for Arabic (with Persian and Urdu letters)
// and Hebrew, and ASCII. Its weights (0.7 MB) are embedded.
package rtl

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the right-to-left recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
