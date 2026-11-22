package yomi_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// BenchmarkEngine_Read reads each fixture frame with every recognizer and the
// router, as a program reading any script would. CI runs it on x86 and arm64
// runners, so the two architectures compare on the same frames.
func BenchmarkEngine_Read(b *testing.B) {
	e := routedEngine(b)
	frames, err := filepath.Glob("testdata/fixtures/*.png")
	if err != nil || len(frames) == 0 {
		b.Fatalf("no fixture frames: %v", err)
	}
	for _, path := range frames {
		frame := loadFrame(b, path)
		scale := 1
		if strings.Contains(path, "@2x") {
			scale = 2
		}
		b.Run(filepath.Base(path), func(b *testing.B) {
			for b.Loop() {
				if _, err := e.Read(context.Background(), frame, scale); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
