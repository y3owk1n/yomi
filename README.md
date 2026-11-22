# yomi

yomi reads text off screen frames: app windows, menus, terminals and web
pages. It is pure Go with no cgo and no machine learning runtime. Each model
package embeds its weights, nothing downloads, and the same engine runs on
macOS, Linux and Windows. It is built for
[Neru](https://github.com/y3owk1n/neru).

```go
m, err := western.Model() // github.com/y3owk1n/yomi/models/western
if err != nil {
	return err
}
e := yomi.New(m)
lines, err := e.Read(ctx, frame, 2) // frame is an *image.RGBA, 2 on a HiDPI display
for _, l := range lines {
	for _, w := range l.Words {
		fmt.Println(w.Text, w.Box, w.Confidence)
	}
}
```

## Why yomi

Operating systems each ship an OCR engine, and none fits a tool that has to
read any window, in any script, within a keystroke, the same way on macOS,
Linux and Windows. Vision's fast mode reads 37 to 57% of UI words across 17
languages and its accurate mode takes up to 0.7 s a frame. tesseract takes 3
to 12 times as long as yomi and needs a language package per script, and
Windows.Media.Ocr needs an OCR pack per language. yomi reads only screens, so its models are small, and
one pure-Go engine reads the same pixels to the same words everywhere.

In Neru's vim mode, on the same captures, yomi reads 91.7% of words right
against Vision fast's 85.1% and Vision accurate's 88.4%, in 11 to 60 ms a
frame against 22 to 75 ms and 193 to 696 ms. On Linux it reads more words than
tesseract on every script but Hindi, at a fraction of the time.
[docs/why.md](docs/why.md) compares it with every engine, says how it works,
and lists where it is still behind.

## Reading a frame

Read returns lines top to bottom. Each word has a box and one box per
character, in logical pixels. Its confidence is the lowest probability the
model gave any of its characters. An Engine is safe for concurrent use, and
canceling the context stops lines not yet read. Frames and the text read from
them are screen content, so yomi never logs them, writes them anywhere or
keeps them past the call.

## Models

A program links only the models it imports.

| Package | Scripts | Size | yomi (1x / 2x) | Vision accurate (1x / 2x) |
|---|---|---|---|---|
| `models/western` | Latin with the European, Turkish and Vietnamese letters, Cyrillic, Greek | 0.5 MB | 94.6% / 94.9% of UI words | 86.3% / 86.3% |
| `models/cjk` | Simplified and Traditional Chinese, Japanese, Korean | 3.6 MB | 91.9% / 92.2% of UI characters | 86.8% / 86.0% |
| `models/rtl` | Arabic with Persian and Urdu letters, Hebrew | 0.8 MB | Arabic 82.3% / 84.3%, Hebrew 84.0% / 84.7% of UI words | Arabic 83.6% / 89.5%, Hebrew 2.3% |
| `models/indic` | Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala | 0.6 MB | Hindi 84.3% / 85.1% of UI words | 92.1% / 92.4% |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.5 MB | Thai 96.6% / 96.9% of UI characters | 94.5% / 94.2% |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 0.5 MB | 91.6%, 80.9% and 69.2% of held-out words at 1x | 15%, 13% and 20% |
| `models/latin` | printable ASCII | 0.6 MB | English only | |
| `models/router` | names each line's script family | 0.1 MB | | |

UI pages hold Apple's own interface strings, rendered by WebKit at 11 to 16
pixels in light and dark. No training source draws from them. Held-out pages
use training-style text in fonts training never saw. Vision accurate is
macOS's best mode, with automatic language detection. It reads Greek at 10%,
where Western reads 95%, and Japanese at 78%, where CJK reads 92%.

To read text without knowing its script, give one engine several
recognizers and the router:

```go
e := yomi.New(westernModel, yomi.WithModel(cjkModel), yomi.WithModel(rtlModel), yomi.WithRouter(routerModel))
```

The router names each line's family, and that family's recognizer reads it.
A line in a family the engine has no recognizer for goes to the model passed
to New.

## Try it

```sh
oku exec -- go build -o bin/yomi ./cmd/yomi
screencapture -i -x /tmp/shot.png                  # drag a box around some text
./bin/yomi read -model western -scale 2 /tmp/shot.png
./bin/yomi read -model cjk -scale 2 /tmp/shot.png
./bin/yomi read -model auto -scale 2 /tmp/shot.png   # every recognizer, routed
```

`-scale 2` is for a Retina capture, `-scale 1` for a standard display.
`yomi bench DIR` times every page of an evaluation set.

## Status

Western and CJK clear their bars against Vision, and Thai, Lao, Khmer and
Georgian clear theirs. Arabic, Hebrew, the Indic scripts, Myanmar, Armenian
and Ethiopic are packaged but below theirs. [docs/design.md](docs/design.md)
holds the plan, the bar each model must clear and the measurements against
Vision, PaddleOCR, tesseract and Windows.Media.Ocr.

`train/` holds the training pipeline. It renders synthetic lines from open
fonts and trains with PyTorch.

MIT licensed.
