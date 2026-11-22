// Package router is yomi's router: a small model that names the script
// family of each line, so an engine given it (yomi.WithRouter) reads the line
// with that family's recognizer. Its weights (0.1 MB) are embedded.
package router

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the router, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
