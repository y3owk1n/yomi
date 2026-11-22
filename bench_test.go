package yomi_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/y3owk1n/yomi"
	"github.com/y3owk1n/yomi/models/caucasian"
	"github.com/y3owk1n/yomi/models/cjk"
	"github.com/y3owk1n/yomi/models/indic"
	"github.com/y3owk1n/yomi/models/router"
	"github.com/y3owk1n/yomi/models/rtl"
	"github.com/y3owk1n/yomi/models/sea"
	"github.com/y3owk1n/yomi/models/western"
)

// BenchmarkEngine_Read reads each fixture frame with every recognizer and the
// router, as a program reading any script would. CI runs it on x86 and arm64
// runners, so the two architectures compare on the same frames.
func BenchmarkEngine_Read(b *testing.B) {
	var opts []yomi.Option
	for _, load := range []func() (*yomi.Model, error){cjk.Model, rtl.Model, indic.Model, sea.Model, caucasian.Model} {
		m, err := load()
		if err != nil {
			b.Fatal(err)
		}
		opts = append(opts, yomi.WithModel(m))
	}
	r, err := router.Model()
	if err != nil {
		b.Fatal(err)
	}
	w, err := western.Model()
	if err != nil {
		b.Fatal(err)
	}
	e := yomi.New(w, append(opts, yomi.WithRouter(r))...)
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
