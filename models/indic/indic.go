// Package indic is yomi's recognizer for Devanagari (Hindi, Marathi, Nepali),
// Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam and Sinhala,
// and ASCII. Its weights (0.6 MB) are embedded.
package indic

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the Indic recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
