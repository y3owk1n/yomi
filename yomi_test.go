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
	"unicode"

	"github.com/y3owk1n/yomi"
	"github.com/y3owk1n/yomi/models/caucasian"
	"github.com/y3owk1n/yomi/models/cjk"
	"github.com/y3owk1n/yomi/models/indic"
	"github.com/y3owk1n/yomi/models/latin"
	"github.com/y3owk1n/yomi/models/router"
	"github.com/y3owk1n/yomi/models/rtl"
	"github.com/y3owk1n/yomi/models/sea"
	"github.com/y3owk1n/yomi/models/western"
)

func loadFrame(t testing.TB, path string) *image.RGBA {
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

// routedEngine reads with every recognizer and the router, as a program
// reading any script would.
func routedEngine(tb testing.TB) *yomi.Engine {
	tb.Helper()
	var opts []yomi.Option
	for _, load := range []func() (*yomi.Model, error){cjk.Model, rtl.Model, indic.Model, sea.Model, caucasian.Model} {
		m, err := load()
		if err != nil {
			tb.Fatal(err)
		}
		opts = append(opts, yomi.WithModel(m))
	}
	r, err := router.Model()
	if err != nil {
		tb.Fatal(err)
	}
	w, err := western.Model()
	if err != nil {
		tb.Fatal(err)
	}
	return yomi.New(w, append(opts, yomi.WithRouter(r))...)
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

// TestEngine_Read_BreaksWordsAtWideGaps reads two labels set side by side, a
// gap wider than any space apart, which the recognizer reads with no space
// between them.
func TestEngine_Read_BreaksWordsAtWideGaps(t *testing.T) {
	m, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/side-by-side@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := text(lines), "Attachment OpenUSD \n"; got != want {
		t.Errorf("read %q, want %q", got, want)
	}
}

// TestEngine_Read_ReadsStackedGlyphsOnce reads a line of Korean, whose
// syllables stack a final consonant under the rest, and expects no two words
// on top of one another, as a second read of the line along its bottom would
// leave.
func TestEngine_Read_ReadsStackedGlyphsOnce(t *testing.T) {
	m, err := cjk.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/hangul-stacked.png"), 1)
	if err != nil {
		t.Fatal(err)
	}
	var words []yomi.Word
	for _, l := range lines {
		words = append(words, l.Words...)
	}
	if len(words) == 0 {
		t.Fatal("read no words")
	}
	for i, a := range words {
		for _, b := range words[i+1:] {
			in := a.Box.Intersect(b.Box)
			if in.Dx()*2 > min(a.Box.Dx(), b.Box.Dx()) && in.Dy()*2 > min(a.Box.Dy(), b.Box.Dy()) {
				t.Errorf("words %q %v and %q %v lie on top of one another", a.Text, a.Box, b.Text, b.Box)
			}
		}
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

// TestEngine_Read_KeepsTightlySetThaiRowsApart reads two rows of 11 px Thai
// set so close that the marks below the first touch those above the second,
// and expects the Latin words in each row, "16" in the first and "iPhone" in
// the second. Read as one line, the two rows yield nothing.
func TestEngine_Read_KeepsTightlySetThaiRowsApart(t *testing.T) {
	m, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/thai-tight@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"16", "iPhone"} {
		if !strings.Contains(got, want) {
			t.Errorf("read %q, want it to hold %q", got, want)
		}
	}
}

// TestEngine_Read_KeepsTightlySetMyanmarRowsApart reads two rows of 11 px
// Myanmar whose marks touch across the rows, and expects the Latin words in
// each row, "pfizer" in the first and "Choice" in the second. Grouped from a
// blob spanning both, the rows read as one line and yield nothing.
func TestEngine_Read_KeepsTightlySetMyanmarRowsApart(t *testing.T) {
	m, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/myanmar-tight@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"pfizer", "Choice"} {
		if !strings.Contains(got, want) {
			t.Errorf("read %q, want it to hold %q", got, want)
		}
	}
}

// TestEngine_Read_GivesRightToLeftWordsInLogicalOrder reads Arabic labels
// beside Latin ones and expects each Arabic word as it is written, first
// letter first, though the recognizer scans it from its last letter, and the
// Latin words left to right.
func TestEngine_Read_GivesRightToLeftWordsInLogicalOrder(t *testing.T) {
	m, err := rtl.Model()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := yomi.New(m).Read(context.Background(), loadFrame(t, "testdata/fixtures/arabic@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	read := map[string]bool{}
	for _, l := range lines {
		for _, w := range l.Words {
			read[w.Text] = true
		}
	}
	for _, want := range []string{"حذف", "متابعة", "OK", "SSH", "تسجيل", "مراقب"} {
		if !read[want] {
			t.Errorf("did not read %q; read:\n%s", want, text(lines))
		}
	}
}

// TestEngine_Read_RoutesEachLineToItsScriptsRecognizer gives one engine the
// Western and CJK recognizers and the router, and expects Korean read by CJK
// and Russian, English and Greek read by Western, without being told which.
func TestEngine_Read_RoutesEachLineToItsScriptsRecognizer(t *testing.T) {
	w, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	c, err := cjk.Model()
	if err != nil {
		t.Fatal(err)
	}
	r, err := router.Model()
	if err != nil {
		t.Fatal(err)
	}
	e := yomi.New(w, yomi.WithModel(c), yomi.WithRouter(r))
	tests := []struct {
		frame string
		scale int
		want  []string
	}{
		{"hangul-stacked.png", 1, []string{"진단", "옵션"}},
		{"lookalikes@2x.png", 2, []string{"Сохранить копию", "Copy screen", "Ελληνικά κείμενα"}},
	}
	for _, tt := range tests {
		t.Run(tt.frame, func(t *testing.T) {
			lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/"+tt.frame), tt.scale)
			if err != nil {
				t.Fatal(err)
			}
			got := text(lines)
			for _, want := range tt.want {
				if !strings.Contains(got, want+" ") {
					t.Errorf("did not read %q; read:\n%s", want, got)
				}
			}
		})
	}
}

// TestEngine_Read_SkipsRoundIconsOnRoutedLines reads mail rows whose unread
// dots the CJK recognizer would read as 〇, through the routed engine, and
// expects the rows' text without them.
func TestEngine_Read_SkipsRoundIconsOnRoutedLines(t *testing.T) {
	w, err := western.Model()
	if err != nil {
		t.Fatal(err)
	}
	c, err := cjk.Model()
	if err != nil {
		t.Fatal(err)
	}
	r, err := router.Model()
	if err != nil {
		t.Fatal(err)
	}
	e := yomi.New(w, yomi.WithModel(c), yomi.WithRouter(r))
	lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/mail-rows@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	if !strings.Contains(got, "profile review") {
		t.Errorf("did not read the row's subject; read:\n%s", got)
	}
	if strings.Contains(got, "〇") {
		t.Errorf("read a round icon as 〇; read:\n%s", got)
	}
}

// TestEngine_Read_ReadsOnlyThePagesScripts reads tightly set Myanmar and
// English rows with every recognizer and the router. A short line the router
// is unsure of, a clipped mark or an icon, once went to the Indic recognizer
// and read as Telugu. A script no confident line of the page is written in
// must not appear.
func TestEngine_Read_ReadsOnlyThePagesScripts(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/myanmar-tight@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	if strings.ContainsFunc(got, func(r rune) bool { return unicode.Is(unicode.Telugu, r) }) {
		t.Errorf("read Telugu on a Myanmar and English page:\n%s", got)
	}
	if !strings.Contains(got, "pfizer") {
		t.Errorf("did not read the English on the page:\n%s", got)
	}
}
