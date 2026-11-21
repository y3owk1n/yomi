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

| Package | Scripts | Size | Mac UI text, yomi (1x / 2x) | Vision accurate (1x / 2x) |
|---|---|---|---|---|
| `models/western` | Latin with the European, Turkish and Vietnamese letters, Cyrillic, Greek | 0.9 MB | 94.4% / 94.8% of words | 86.3% / 86.3% |
| `models/cjk` | Simplified and Traditional Chinese, Japanese, Korean | 7.2 MB | 90.4% / 90.8% of characters | 86.8% / 86.0% |
| `models/latin` | printable ASCII | 0.6 MB | English only | |

The test pages hold Apple's own interface strings, 17 languages for Western
and 4 for CJK, rendered by WebKit at 11 to 16 pixels in light and dark. No
training source draws from them. Vision accurate is macOS's best mode, with
automatic language detection. It reads Greek at 10%, where Western reads 95%,
and Japanese at 78%, where CJK reads 91%.

## Try it

```sh
oku exec -- go build -o bin/yomi ./cmd/yomi
screencapture -i -x /tmp/shot.png                  # drag a box around some text
./bin/yomi read -model western -scale 2 /tmp/shot.png
./bin/yomi read -model cjk -scale 2 /tmp/shot.png
```

`-scale 2` is for a Retina capture, `-scale 1` for a standard display.
`yomi bench DIR` times every page of an evaluation set.

## Status

Right-to-left (Arabic and Hebrew) is trained but below its bar. Indic,
Southeast Asian, and Georgian, Armenian and Ethiopic are training. A small
router model will then pick the model for each line, so a caller need not
know the language. [docs/design.md](docs/design.md) holds the plan, the bar each
model must clear and the measurements against Vision, tesseract and
Windows.Media.Ocr.

`train/` holds the training pipeline. It renders synthetic lines from open
fonts and trains with PyTorch.

MIT licensed.
