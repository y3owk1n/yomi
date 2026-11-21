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
| `models/western` | Latin with the European, Turkish and Vietnamese letters, Cyrillic, Greek | 0.9 MB | 94.6% / 94.9% of UI words | 86.3% / 86.3% |
| `models/cjk` | Simplified and Traditional Chinese, Japanese, Korean | 7.2 MB | 91.9% / 92.2% of UI characters | 86.8% / 86.0% |
| `models/rtl` | Arabic with Persian and Urdu letters, Hebrew | 0.7 MB | Arabic 77.2% / 81.6%, Hebrew 81.7% / 82.0% of UI words | Arabic 83.6% / 89.5%, Hebrew 2.3% |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.9 MB | Thai 92.2% / 93.2% of UI characters | 94.5% / 94.2% |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 1 MB | 91.6%, 80.9% and 69.2% of held-out words at 1x | 15%, 13% and 20% |
| `models/latin` | printable ASCII | 0.6 MB | English only | |
| `models/router` | names each line's script family | 0.2 MB | | |

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

Western and CJK clear their bars against Vision. Right-to-left, Southeast
Asian, and Georgian, Armenian and Ethiopic are packaged but below theirs.
Indic is training, and has no package yet. [docs/design.md](docs/design.md)
holds the plan, the bar each model must clear and the measurements against
Vision, PaddleOCR, tesseract and Windows.Media.Ocr.

`train/` holds the training pipeline. It renders synthetic lines from open
fonts and trains with PyTorch.

MIT licensed.
