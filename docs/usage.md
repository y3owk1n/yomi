# Using yomi

yomi is a Go library. You give it a screen frame and it returns the frame's
lines, each line's words, and a box for every character. This page covers
installing it, choosing models, getting a frame and its scale right, reading
the results, and what yomi does not do. The examples in `example_test.go`
compile with every test run, so they match the current API.

## Install

```sh
go get github.com/y3owk1n/yomi
```

yomi needs Go 1.27 or later. It has no cgo and no dependencies outside the
standard library, so it builds for every target Go supports with
`CGO_ENABLED=0`. The models are Go packages with their weights embedded.
Nothing downloads at build or run time.

## Read a frame

```go
m, err := western.Model() // github.com/y3owk1n/yomi/models/western
if err != nil {
	return err
}
e := yomi.New(m)

lines, err := e.Read(ctx, frame, 2)
if err != nil {
	return err // only ctx's error
}
for _, l := range lines {
	for _, w := range l.Words {
		fmt.Println(w.Text, w.Box, w.Confidence)
	}
}
```

`frame` is an `*image.RGBA`. `png.Decode` returns one for some files and an
`*image.NRGBA` or another type for others. Any `image.Image` converts with
`draw.Draw`, as `ExampleEngine_Read` does. A sub-image from `SubImage` works too, and its
boxes come back relative to its own top-left corner.

## Choose models

Each script family is its own package, and a program links only the ones it
imports.

| Package | Scripts | Weights |
|---|---|---|
| `models/western` | Latin with European, Turkish and Vietnamese letters, Cyrillic, Greek | 0.5 MB |
| `models/cjk` | Simplified and Traditional Chinese, Japanese, Korean | 3.6 MB |
| `models/rtl` | Arabic with Persian and Urdu letters, Hebrew | 0.8 MB |
| `models/indic` | Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala | 1.1 MB |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.5 MB |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 0.5 MB |
| `models/latin` | printable ASCII only | 0.6 MB |
| `models/router` | names each line's script family | 0.1 MB |

Every recognizer also reads ASCII, so a Russian or Thai interface with
English words in it reads with one model.

**One script.** If you know the text's script, pass its recognizer to `New`
and nothing else. This is the fastest engine.

**Any script.** Pass a base recognizer to `New`, add the others with
`WithModel`, and add the router with `WithRouter`. `ExampleWithRouter` builds
this engine.

```go
e := yomi.New(westernModel,
	yomi.WithModel(cjkModel),
	yomi.WithModel(rtlModel),
	yomi.WithRouter(routerModel))
```

`Read` routes every line of a frame before it reads any of them. A script
counts as on the page only once a line of real length shows it, so an icon on
an English page reads with the Western recognizer rather than the Chinese or
Arabic one. A line in a family the engine has no recognizer for goes to the
model passed to `New`. Without a router, the model passed to `New` reads every
line and the others go unused.

Each `Model` function loads its package's weights once and returns the same
`*yomi.Model` after that. The only error it can return is a corrupt embedded
model, and yomi's tests load every one.

## Scale

`scale` is frame pixels per logical pixel, the factor the display draws at.

| Display | scale |
|---|---|
| Standard density | 1 |
| macOS Retina, at any "looks like" resolution | 2 |
| Windows at 125%, 150%, 175% | 1.25, 1.5, 1.75 |
| Linux at an output scale of 1.5 | 1.5 |

When you know the region you captured in logical units, the scale is the
frame's height divided by the region's height. On Windows, a per-monitor DPI
aware process captures in physical pixels, and the scale is the monitor's DPI
scale.

The line finder uses the scale to know how big text is meant to be. A 1.25x
frame read as 2 puts the text at 0.625 of its size, and small text reads as
nothing. Fractional scales read within 2 points of 1x and 2x.

## Results

`Read` returns `[]yomi.Line` top to bottom, and each line's words left to
right as they sit on screen.

- `Line.Box` is the region the line was read from.
- `Word.Text` is in logical order, so an Arabic or Hebrew word reads as
  written, not as drawn.
- `Word.Box` spans the word from its first to its last character, and from
  the top to the bottom of its line's ink, as an app's own text bounds do.
- `Word.Chars` holds one box per character of `Text`, in `Text`'s order. Each
  reaches halfway to its neighbours, so together they tile `Box`. A cursor
  between two characters goes at the shared edge.
- `Word.Confidence` is from 0 to 1, the lowest probability the recognizer gave
  any of the word's characters. A lone character below 0.5 is dropped before
  you see it, since that is nearly always an icon read as a letter.

Every box is in logical pixels from the frame's top-left corner. To get frame
pixels, multiply by the scale. To get screen coordinates, add the capture's
origin, in pixels or in logical units, whichever your platform uses.
`ExampleWord` shows both.

## Read one line

`ReadLine(frame, box, scale)` reads a region you already know holds one line,
with no line finding. `box` is in logical pixels from the frame's top-left.
With a router, it routes the line on its own, without the page-wide vote
`Read` takes. Its word boxes span the region you gave, not the line's ink.

## Concurrency and cancellation

An `Engine` is safe for concurrent use. Build one and keep it, since it pools
its scratch memory between reads. `Read` reads lines on one goroutine per CPU
by default, and `WithWorkers(n)` caps that.

Canceling `ctx` stops lines that have not started, and `Read` returns
`ctx.Err()`. That is the only error `Read` returns. `ReadLine` reads one line
and takes no context.

## Speed

On an Apple M1 runner a 4K window reads in about 48 ms with every recognizer
and the router, and on an AMD runner in 28 ms. A single-script engine is
faster, and so is a smaller frame. Crop to the region you need before calling
`Read`. The hot loops run in NEON on arm64 and in AVX2 with FMA on amd64, and
fall back to plain Go elsewhere. [why.md](why.md) has the measurements.

## What yomi does not do

- Photos, scanned documents, handwriting, and rotated or curved text. It is
  trained on screen text only.
- Paragraphs, columns or reading order across a page. It returns lines and
  leaves layout to you.
- Telling you a line's language. The router picks a recognizer, and it does
  not report its choice.
- Text inside pictures, and text dimmed behind a dialog, often goes unread. A
  heading over a plain part of a picture reads, and one across a busy part
  may not.
- Text only a few pixels tall often goes unread.

## Privacy

yomi makes no network calls and writes nothing. Frames and the text read from
them exist only for the call. Anything your program logs or stores from the
results is screen content.

## Command line

`cmd/yomi` reads PNG files from the terminal:

```sh
go install github.com/y3owk1n/yomi/cmd/yomi@latest
yomi read -model auto -scale 2 shot.png    # every recognizer, routed
yomi read -model western -scale 1.5 shot.png
yomi bench -model auto DIR                 # times every page of an evaluation set
```

## Your own model

`yomi.LoadModel(fsys)` loads any model directory holding `model.json` and
`weights.bin`, such as `os.DirFS("path/to/model")`. `train/README.md` covers
training and exporting one.
