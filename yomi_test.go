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

// TestEngine_Read_ReadsTextOnAGradient reads white text on a dark gradient,
// as behind a video's title. The gradient spreads over many lumas while the
// text sits on one, and the text, read as the background, once inverted every
// line to nothing.
func TestEngine_Read_ReadsTextOnAGradient(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/gradient-title@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"挑戰世界經典馬拉松", "Weekly highlights from the marathon finish line"} {
		if !strings.Contains(got, want) {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_ReadsTightlySpacedLines reads three lines of Chinese set
// 18 points apart at 14 points, so each line's crop holds the bottom of the
// line above and the top of the one below, which once garbled the middle one.
func TestEngine_Read_ReadsTightlySpacedLines(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/tight-cjk@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"對夜貓子的我來說", "不過一到比賽現場", "大家都在路邊加油"} {
		if !strings.Contains(got, want) {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_ReadsLargeBoldHeadings reads a 48 point bold heading. Its
// strokes are wider than a background run, and the gaps between them can be
// too narrow to be one, which once split it into a dozen lines.
func TestEngine_Read_ReadsLargeBoldHeadings(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/bold-heading@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := text(lines); !strings.Contains(got, "Amazing price") {
		t.Errorf("did not read %q; read:\n%s", "Amazing price", got)
	}
}

// TestEngine_Read_ReadsFaintText reads grey text on an off-white page and
// pale labels on a light grey bar, under 40 luma levels from their
// background, as faded heroes and translucent navigation bars draw them.
func TestEngine_Read_ReadsFaintText(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/faint-text@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"Store", "Support", "Amazing price", "Meet the newest"} {
		if !strings.Contains(got, want) {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_ReadsFractionalScales reads the tight Thai rows captured at
// 1.25 and 1.5 pixels per point, as a display scaled to 125% or 150% draws
// them. Read as a whole 2x frame, the text comes out at 0.625 or 0.75 of its
// size and yields nothing.
func TestEngine_Read_ReadsFractionalScales(t *testing.T) {
	e := routedEngine(t)
	for _, tt := range []struct {
		frame string
		scale float64
	}{
		{"thai-tight@1.25x.png", 1.25},
		{"thai-tight@1.5x.png", 1.5},
	} {
		t.Run(tt.frame, func(t *testing.T) {
			lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/"+tt.frame), tt.scale)
			if err != nil {
				t.Fatal(err)
			}
			got := text(lines)
			for _, want := range []string{"ค้นหาของฉัน", "iPhone"} {
				if !strings.Contains(got, want) {
					t.Errorf("did not read %q; read:\n%s", want, got)
				}
			}
		})
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
		scale float64
		want  []string
	}{
		{"hangul-stacked.png", 1, []string{"진단", "켜면"}},
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

// TestEngine_Read_ReadsLargeFigures reads three dashboard cards, each a label,
// a 26 px bold figure and a caption set close together, and expects every
// figure whole. The figures' strokes are wider than the background runs the
// line finder looks for, and their commas hang into the caption below.
func TestEngine_Read_ReadsLargeFigures(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/figures@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"61,606", "21,780", "62,824"} {
		if !strings.Contains(got, want+" ") {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_ReadsLatinBesideAnArrowAsLatin reads dashboard captions, a
// word and an arrow, which the router sends to the Indic recognizer, and
// expects them in Latin. A recognizer for another script that reads a line as
// mostly ASCII hands it to the base recognizer.
func TestEngine_Read_ReadsLatinBesideAnArrowAsLatin(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/figures@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	if strings.ContainsFunc(got, func(r rune) bool { return unicode.Is(unicode.Malayalam, r) }) {
		t.Errorf("read a Latin caption as Malayalam:\n%s", got)
	}
	if !strings.Contains(got, "gamma →") {
		t.Errorf("did not read the caption %q:\n%s", "gamma →", got)
	}
}

// TestEngine_Read_ReadsLargeType reads a web page's 26 px bold heading and a
// 34 px bold figure at 1x, whose stems are as tall as the field borders the
// line finder clears and whose letters touch.
func TestEngine_Read_ReadsLargeType(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/heading-links.png"), 1)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"Repository settings", "21,780"} {
		if !strings.Contains(got, want) {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_ReadsUnderlinedLinksAsWords reads underlined links, whose
// underline fills the gaps between words, and expects the spaces: a row of
// links, and a link in serif body text whose underline falls across two pixel
// rows at 1x and once read "commit move" as "commit_oxg".
func TestEngine_Read_ReadsUnderlinedLinksAsWords(t *testing.T) {
	e := routedEngine(t)
	for _, tt := range []struct {
		frame string
		scale float64
		want  []string
	}{
		{"heading-links.png", 1, []string{"Pull requests", "Security and quality"}},
		{"serif-links.png", 1, []string{"share commit move rename"}},
		{"serif-links@2x.png", 2, []string{"share commit move rename"}},
	} {
		t.Run(tt.frame, func(t *testing.T) {
			lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/"+tt.frame), tt.scale)
			if err != nil {
				t.Fatal(err)
			}
			got := text(lines)
			for _, want := range tt.want {
				if !strings.Contains(got, want) {
					t.Errorf("did not read %q; read:\n%s", want, got)
				}
			}
		})
	}
}

// TestEngine_Read_FitsWordBoxesToTheInk expects a word's box to span its
// line's ink, not the padded crop the recognizer read, so a short word in
// large type keeps a word's shape: "21,780" in 34 px bold is wider than twice
// its height.
func TestEngine_Read_FitsWordBoxesToTheInk(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/heading-links.png"), 1)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, l := range lines {
		for _, w := range l.Words {
			if !w.Box.In(l.Box) {
				t.Errorf("word %q box %v is outside its line's box %v", w.Text, w.Box, l.Box)
			}
			if w.Text == "21,780" {
				found = true
				if w.Box.Dx() <= 2*w.Box.Dy() {
					t.Errorf("box of %q is %v, no wider than twice its height", w.Text, w.Box)
				}
			}
		}
	}
	if !found {
		t.Fatalf("did not read %q:\n%s", "21,780", text(lines))
	}
}

// TestEngine_Read_ReadsTextOnAButtonItsCropOverruns reads white text on a blue
// button at 2x, where the line's crop runs past the button's bottom and right
// edges so that most of the crop's border is the page, not the button.
func TestEngine_Read_ReadsTextOnAButtonItsCropOverruns(t *testing.T) {
	lines, err := routedEngine(t).Read(context.Background(), loadFrame(t, "testdata/fixtures/button-overrun@2x.png"), 2)
	if err != nil {
		t.Fatal(err)
	}
	got := text(lines)
	for _, want := range []string{"Apple Watch連接充電器，", "iPhone和Apple"} {
		if !strings.Contains(got, want) {
			t.Errorf("did not read %q; read:\n%s", want, got)
		}
	}
}

// TestEngine_Read_SettlesYehAndKafOnThePagesLanguage reads an Arabic page and
// a Persian one. Yeh and kaf draw alike in the two at the start and in the
// middle of a word, and each page should keep its own language's letters
// there: ي and ك in Arabic, ی and ک in Persian.
func TestEngine_Read_SettlesYehAndKafOnThePagesLanguage(t *testing.T) {
	e := routedEngine(t)
	tests := []struct {
		frame      string
		want, lose []string
	}{
		{"arabic-yeh@2x.png", []string{"تعديل", "خلايا"}, []string{"ی", "ک"}},
		{"persian-yeh@2x.png", []string{"جنایتکار", "فیلتر", "کارهایی"}, []string{"ي", "ك"}},
	}
	for _, tt := range tests {
		t.Run(tt.frame, func(t *testing.T) {
			lines, err := e.Read(context.Background(), loadFrame(t, "testdata/fixtures/"+tt.frame), 2)
			if err != nil {
				t.Fatal(err)
			}
			got := text(lines)
			for _, want := range tt.want {
				if !strings.Contains(got, want) {
					t.Errorf("did not read %q; read:\n%s", want, got)
				}
			}
			for _, lose := range tt.lose {
				if strings.Contains(got, lose) {
					t.Errorf("read %q, the other language's letter; read:\n%s", lose, got)
				}
			}
		})
	}
}
