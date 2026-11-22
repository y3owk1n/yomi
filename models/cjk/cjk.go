// Package cjk is yomi's recognizer for Chinese (Simplified and Traditional),
// Japanese and Korean: the Han characters of GB2312, Big5 level 1 and JIS
// level 1, kana, the 2,350 Hangul syllables of KS X 1001, CJK punctuation, and
// ASCII. Its weights (3.6 MB) are embedded.
package cjk

import (
	"embed"
	"sync"

	"github.com/y3owk1n/yomi"
)

//go:embed model.json weights.bin
var files embed.FS

// Model returns the CJK recognizer, loaded once.
var Model = sync.OnceValues(func() (*yomi.Model, error) { return yomi.LoadModel(files) })
