// Package caucasian is yomi's recognizer for Georgian, Armenian and Ethiopic,
// and ASCII. Its weights (1 MB) are embedded.
package caucasian

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the Georgian, Armenian and Ethiopic recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
