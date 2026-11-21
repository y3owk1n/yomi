package yomi_test

import (
	"context"
	"errors"
	"image"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"testing"

	"github.com/y3owk1n/yomi"
	"github.com/y3owk1n/yomi/models/latin"
	"github.com/y3owk1n/yomi/models/western"
)

func loadFrame(t *testing.T, path string) *image.RGBA {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	src, err := png.Decode(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	img := image.NewRGBA(src.Bounds())
	draw.Draw(img, img.Rect, src, image.Point{}, draw.Src)
	return img
}

func engine(t *testing.T) *yomi.Engine {
	t.Helper()
	m, err := latin.Model()
	if err != nil {
		t.Fatal(err)
	}
	return yomi.New(m)
}

func text(lines []yomi.Line) string {
	var b strings.Builder
	for _, l := range lines {
		for _, w := range l.Words {
			b.WriteString(w.Text)
			b.WriteByte(' ')
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// TestEngine_Read_ReadsUIFixtures reads screen-sized frames of UI text in the
// macOS system font (Neru's OCR fixtures) and expects each phrase word for
// word, on one line. "CLI" at 11 pt still reads as "CLl", so those phrases
// are not listed yet.
func TestEngine_Read_ReadsUIFixtures(t *testing.T) {
	e := engine(t)
	tests := []struct {
		frame   string
		phrases []string
	}{
		{"1080p-13pt.png", []string{"denotes breaking CLI changes.", "Settings Downloads Help"}},
		{"4k-16.5pt.png", []string{"denotes breaking CLI changes.", "Settings Downloads Help"}},
		{"1080p-11pt.png", []string{"Settings Downloads Help"}},
		{"4k-11pt.png", []string{"Settings Downloads Help"}},
	}
	for _, tt := range tests {
		t.Run(tt.frame, func(t *testing.T) {
			lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/"+tt.frame), 1)
			if err != nil {
				t.Fatal(err)
			}
			got := text(lines)
			for _, p := range tt.phrases {
				if !strings.Contains(got, p+" ") {
					t.Errorf("phrase %q not read; read:\n%s", p, got)
				}
			}
		})
	}
}

// TestEngine_Read_WordBoxesCoverTheirText checks that each read word's box
// lies inside the frame, in logical pixels, that its character boxes tile it
// left to right, one per character, and that confidence is a probability.
func TestEngine_Read_WordBoxesCoverTheirText(t *testing.T) {
	frame := loadFrame(t, "testdata/fixtures/4k-16.5pt.png")
	lines, err := engine(t).Read(context.Background(), frame, 2)
	if err != nil {
		t.Fatal(err)
	}
	logical := image.Rect(0, 0, frame.Rect.Dx()/2, frame.Rect.Dy()/2)
	words := 0
	for _, l := range lines {
		for _, w := range l.Words {
			words++
			if !w.Box.In(logical) || w.Box.Empty() {
				t.Errorf("word %q box %v outside %v", w.Text, w.Box, logical)
			}
			if w.Confidence <= 0 || w.Confidence > 1 {
				t.Errorf("word %q confidence %v", w.Text, w.Confidence)
			}
			if len(w.Chars) != len([]rune(w.Text)) {
				t.Fatalf("word %q has %d character boxes", w.Text, len(w.Chars))
			}
			x := w.Box.Min.X
			for i, c := range w.Chars {
				if c.Min.X != x || c.Empty() || !c.In(w.Box) {
					t.Errorf("word %q character %d box %v does not continue at x=%d inside %v", w.Text, i, c, x, w.Box)
				}
				x = c.Max.X
			}
			if x != w.Box.Max.X {
				t.Errorf("word %q character boxes end at %d, word at %d", w.Text, x, w.Box.Max.X)
			}
		}
	}
	if words == 0 {
		t.Fatal("read no words")
	}
}

// TestEngine_Read_CanceledContextReadsNothing checks that a canceled read
// returns the context's error rather than partial lines.
func TestEngine_Read_CanceledContextReadsNothing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	lines, err := engine(t).Read(ctx, loadFrame(t, "testdata/fixtures/1080p-13pt.png"), 1)
	if !errors.Is(err, context.Canceled) || lines != nil {
		t.Fatalf("Read = %d lines, %v; want nil, context.Canceled", len(lines), err)
	}
}

// TestEngine_Read_SkipsCheckboxesBesideLabels reads a table whose rows start
// with a checkbox, which the recognizer reads as a doubtful O.
func TestEngine_Read_SkipsCheckboxesBesideLabels(t *testing.T) {
	m, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/checkboxes@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := text(lines), "release spam \nfilter publish \nfilter release \nspam rename \n"; got != want {
		t.Errorf("read %q, want %q", got, want)
	}
}

// TestEngine_Read_SettlesLookalikesOnTheWordsScript reads Russian, English
// and Greek lines made largely of letters that draw alike across the three
// scripts (С and C, о and o and ο) and expects every letter in its word's own
// script, compared as code points.
func TestEngine_Read_SettlesLookalikesOnTheWordsScript(t *testing.T) {
	m, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/lookalikes@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Сохранить копию", "Copy screen", "Ελληνικά κείμενα"}
	if len(lines) != len(want) {
		t.Fatalf("read %d lines, want %d:\n%s", len(lines), len(want), text(lines))
	}
	for i, l := range lines {
		var words []string
		for _, w := range l.Words {
			words = append(words, w.Text)
		}
		if got := strings.Join(words, " "); got != want[i] {
			t.Errorf("line %d = %q (%U), want %q (%U)", i, got, []rune(got), want[i], []rune(want[i]))
		}
	}
}
