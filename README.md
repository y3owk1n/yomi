# yomi

yomi is an OCR engine for screens. Give it a capture of an app window, a
menu, a terminal or a web page, and it returns each line of text with its
words, a box for every word and every character, and a confidence per word.

It is pure Go, with no cgo and no machine learning runtime. Each model is a
Go package with its weights embedded, so nothing downloads, a program links
only the scripts it imports, and it builds for macOS, Linux and Windows with
`CGO_ENABLED=0`. The same pixels read to the same words on all three.

yomi was built for [Neru](https://github.com/y3owk1n/neru), a keyboard tool
that puts a hint on anything you can click and moves a cursor through any
window's text with vim keys.

## What it is for

yomi fits a program that needs to know what text is on screen and where it
is, between a key press and the response to it:

- clicking, selecting or jumping to on-screen text from the keyboard, as
  Neru does
- finding or acting on text in a window whose accessibility tree has none
- reading interface text in any of 24 scripts without asking the user which

It reads what interfaces draw: text on flat colour, gradients and bars,
headings over the plain part of a picture, faint grey labels, code, tables
and web pages, at 1x, 2x and the scales between.

It is not for photos, scanned documents, handwriting, or rotated or curved
text. Text inside pictures, and text dimmed behind a dialog, often goes
unread. It returns lines, not paragraphs or reading order.

## Quick start

```sh
go get github.com/y3owk1n/yomi
```

```go
m, err := western.Model() // github.com/y3owk1n/yomi/models/western
if err != nil {
	return err
}
e := yomi.New(m)
lines, err := e.Read(ctx, frame, 2) // frame is an *image.RGBA, 2 on a Retina display
if err != nil {
	return err
}
for _, l := range lines {
	for _, w := range l.Words {
		fmt.Println(w.Text, w.Box, w.Confidence)
	}
}
```

To read text without knowing its script, give one engine every recognizer
and the router. The router names each line's script and that script's
recognizer reads it.

```go
e := yomi.New(westernModel,
	yomi.WithModel(cjkModel),
	yomi.WithModel(rtlModel),
	yomi.WithModel(indicModel),
	yomi.WithModel(seaModel),
	yomi.WithModel(caucasianModel),
	yomi.WithRouter(routerModel))
```

[docs/usage.md](docs/usage.md) covers choosing models, getting the scale
right, what each field of a result means, concurrency, and the command line.

## How well it reads

Against Apple Vision's accurate mode, the best OCR macOS ships, with
automatic language detection. 1x / 2x where both were measured.

| Text | yomi | Vision accurate |
|---|---|---|
| Apple's own UI strings, 17 Western languages, words | **95.4% / 95.8%** | 86.3% / 86.3% |
| Chinese, Japanese and Korean UI, characters | **96.6% / 96.9%** | 94.1% / 93.7% |
| Thai UI, characters | **97.8% / 98.0%** | 94.7% / 94.7% |
| Hebrew UI, words | **86.1% / 86.5%** | 2.2% / 2.3% |
| Arabic UI, words | **81.8%** / 84.5% | 81.6% / **87.3%** |
| Hindi UI, words | 90.2% / 91.0% | **92.1% / 92.4%** |
| Lao, Khmer, Myanmar, held-out text, characters at 1x | **84%, 91%, 87%** | 22%, 28%, 7% |
| Georgian, Armenian, Amharic, held-out text, words at 1x | **94%, 84%, 79%** | 15%, 12%, 20% |
| 28 live captures of apps and web pages in 15 languages, their accessibility text | **80.8%** | 65.5% |
| Time per window or page | **30 to 40 ms** | 0.1 to 1.4 s |

No training text comes from the UI strings, and the held-out text is a part
of the word lists training never saw. On the 23 captures in scripts Vision
reads, yomi reads 83.7% to Vision's 84.3%. On the five it does not (Bengali,
Tamil, Hebrew, Georgian, Armenian), yomi reads 76.7%. On 77 websites
held out from all development, yomi reads 89% of the text Vision reads, and
81.0% / 81.3% of the pages' own text to Vision's 81.5% / 81.3%.

[docs/why.md](docs/why.md) compares yomi with Vision, PaddleOCR, tesseract
and Windows.Media.Ocr, says how it works, and lists where it is behind. The
open evaluation set, 127 pages in 42 languages from Firefox's interface
strings, lets anyone score yomi on any platform
([docs/contributing.md](docs/contributing.md)).

## Models

| Package | Scripts | Size |
|---|---|---|
| `models/western` | Latin with European, Turkish and Vietnamese letters, Cyrillic, Greek | 0.5 MB |
| `models/cjk` | Simplified and Traditional Chinese, Japanese, Korean | 3.6 MB |
| `models/rtl` | Arabic with Persian and Urdu letters, Hebrew | 0.8 MB |
| `models/indic` | Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala | 1.1 MB |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.5 MB |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 0.5 MB |
| `models/router` | names each line's script family | 0.1 MB |
| `models/latin` | printable ASCII only | 0.6 MB |

Every recognizer also reads ASCII, so a Russian or Thai interface with
English words in it reads with one model. Every family's recognizer and the
router come to 7.1 MB.

## Try it

```sh
go install github.com/y3owk1n/yomi/cmd/yomi@latest
screencapture -i -x /tmp/shot.png            # drag a box around some text
yomi read -model auto -scale 2 /tmp/shot.png # every recognizer, routed
yomi read -model western -scale 2 /tmp/shot.png
```

`-scale 2` is for a Retina capture, `-scale 1` for a standard display, and
`-scale 1.25` or `-scale 1.5` for a display scaled between.

## Status

Against Vision accurate, tesseract and Windows.Media.Ocr, measured on
2026-11-26, yomi reads every evaluation set as well or better, with fewer
stray words and in a fraction of the time, with three exceptions. Arabic at
2x trails Windows.Media.Ocr by 0.8 points and Vision by 2.8. Hindi trails
Vision by 1.9 / 1.4. Held-out Southeast Asian text at 2x trails tesseract by
0.2. It reads the same pixels to the same
words on macOS, Linux and Windows.

Each script family has a bar set before its model trained
([docs/design.md](docs/design.md)). Western, CJK, Hebrew, Thai, Khmer,
Myanmar and Georgian clear theirs, as do Marathi, Nepali, Bengali, Gujarati
and Punjabi. Arabic clears its bar at 1x and is 2.8 points short at 2x.
Hindi trails Vision by 1.4 to 1.9 points. Lao and Armenian read 84%,
Amharic 79%, and Kannada, Telugu, Tamil, Malayalam and Sinhala 55 to 77% at
1x. Urdu in Nastaliq reads 37%.

## Repository

- [docs/usage.md](docs/usage.md): the guide to the API.
- [docs/why.md](docs/why.md): why yomi exists and how it compares.
- [docs/design.md](docs/design.md): the plan, the bars, and every
  measurement and rejected idea, with dates.
- [docs/contributing.md](docs/contributing.md): reporting misreads, adding
  fonts and words, and training a model to send back.
- `train/`: the training pipeline. It renders lines and web-like pages from
  open fonts and trains with PyTorch. [train/README.md](train/README.md)
  covers it.

yomi makes no network calls and writes nothing. Frames and the text read
from them exist only for the call.

MIT licensed.
