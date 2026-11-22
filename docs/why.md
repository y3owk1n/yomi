# Why yomi

yomi reads the text on a screen. Hand it a frame from an app window, a menu,
a terminal or a web page, and it returns each line and word with a box per
character. Neru, a keyboard tool, puts a hint on anything you can click and
moves through any window's text with vim keys. It needs that answer on macOS,
Linux and Windows within a keystroke, and no engine it could ship gave it.

This page says what was wrong with the alternatives, what yomi does instead,
and how it measures against them, including where it is still behind.
[design.md](design.md) holds the full plan and every measurement.

## What Neru needs

- Words with boxes, and a box per character, since vim mode puts its cursor
  between two letters.
- Every script a user might have on screen, without asking which.
- An answer in tens of milliseconds, since a read sits between a key press
  and the hints appearing.
- The same results on all three platforms, so a fix or a test on one holds on
  the others.
- Nothing that leaves the machine, logs the screen or keeps it.
- Nothing extra to install.

## What was there

| Engine | Platform | What goes wrong for Neru |
|---|---|---|
| Vision (fast) | macOS | Reads Western UI text at 37% at 1x and 57% at 2x, since fast mode skips language correction |
| Vision (accurate) | macOS | 0.2 to 0.7 s a frame, compiles its models for up to a minute the first time it meets a script, reads Greek at 10% and Hebrew at 2% |
| tesseract | Linux | 3 to 12 times slower than yomi on the same machine, a cgo library and a language data package per script, and the user picks the languages |
| Windows.Media.Ocr | Windows | Needs an OCR pack per language, gives no per-word confidence, reads only the languages the account lists |
| PaddleOCR | anywhere | The most accurate on Chinese, but needs ONNX Runtime or ncnn through cgo, and takes 4 to 10 s a page |
| Vision-language models | anywhere | Hundreds of millions of parameters, where each yomi recognizer has 0.2 to 1.8 million |

Each platform also answered differently, so Neru carried three integrations,
three sets of quirks and three answers to "why did this read wrong".

## What yomi does instead

- **Screens only.** No documents, photos, handwriting or rotated text. Screen
  text is flat colour over a flat background, anti-aliased, upright, at a
  handful of sizes. A model built for that alone can be small.
- **Pure Go.** No cgo, no inference runtime, no system library. It builds for
  every Go target with `CGO_ENABLED=0`.
- **Models are packages.** Each script family is a Go package with its
  weights embedded, from 0.2 MB to 7 MB. A program links only the ones it
  imports, and nothing downloads.
- **One engine, every platform.** The same code reads the same pixels to the
  same words on macOS, Linux and Windows. Only capture stays native.
- **No language setting.** A small router names each line's script, and the
  recognizer for that script reads it. Read routes every line of a frame
  first, so a script counts as on the page only once a line of real length
  shows it. An icon on an English page then reads with the Western
  recognizer instead of sending a Chinese or Arabic one after it.
- **Trained on the screen.** Lines are rendered from open fonts the way
  screens draw them: hinted and unhinted, at 1x and 2x, light, dark, dimmed
  and coloured themes, with subpixel fringes, neighbouring lines and controls
  in the margins. Text comes from word lists and real interface strings, and
  the evaluation pages use Apple's own UI strings, which training never sees.
- **Small and fast.** Each recognizer is a convolutional network with CTC
  decoding. On arm64 the hot loops run in NEON assembly, on amd64 in
  register tiles sized for its 16 vector registers.
- **Private by construction.** No network, no logs. Frames and text live only
  for the call.

## How it reads a frame

```
frame (RGBA, scale)
  -> line finder: luma at 1x, ink by flat-background runs, rules and borders
     cleared, blobs grouped into lines
  -> per line, on every core: an ink map 32 rows tall
  -> router: each line's script, then the scripts the page is written in
  -> that script's recognizer: convolutions, line context, CTC over its alphabet
  -> words split on spaces and wide gaps, a box per character, confidence the
     least sure character's probability, right-to-left words in reading order
```

## How it compares

### On macOS, against Vision and PaddleOCR

Word accuracy (character accuracy for CJK and Thai) on pages no training text
came from, at 1x and 2x.

| Set | yomi | Vision accurate | PaddleOCR |
|---|---|---|---|
| Western UI, 17 languages | **94.6% / 94.9%** | 86.3% / 86.3% | 85.9% / 86.8% |
| CJK UI | 92.5% / 92.7% | 86.8% / 86.0% | **96.3% / 96.8%** |
| Arabic UI | 82.3% / 84.3% | **83.6% / 89.5%** | 65.6% / 80.9% |
| Hebrew UI | **84.0% / 84.7%** | 2.3% / 2.3% | no recognizer |
| Hindi UI | 84.3% / 85.1% | **92.1% / 92.4%** | 83.1% / 86.7% |
| Thai UI | **96.6% / 96.9%** | 94.5% / 94.2% | 96.2% / 92.2% |
| Lao, Khmer, Myanmar, held out, 1x | **86.7%, 90.5%, 78.7%** | 22.5%, 29.2%, 7.4% | no recognizer |
| Georgian, Armenian, Amharic, held out, 1x | **91.6%, 80.9%, 69.2%** | 15.3%, 12.5%, 19.9% | no recognizer |

### In Neru's vim mode, against Vision

The same captures through Neru's line-reading path, on one M5 with nothing
else running. Accuracy is over 11 app scenes (mail, chat, editor, Finder,
settings, tables, a dashboard, an article), and the times cover those and a
GitHub page.

| | Vision fast | yomi | Vision accurate |
|---|---|---|---|
| Words read right | 85.1% | **91.7%** | 88.4% |
| Stray words | 53 | **33** | 71 |
| Time per 1280x800 frame | 22 to 75 ms | **11 to 60 ms** | 193 to 696 ms |

yomi reads code in an editor at 98.7%, where Vision reads 62 to 66%, and
tables at 87 to 88%, where Vision reads 76 to 81%.

### On Linux, against tesseract

Both read the same pages in the same Linux VM. tesseract runs as Neru ran it,
LSTM only and sparse text, with each page's own language data, its best case.
The Mac hosting the VM was busy while they ran, so the times compare the two
engines rather than measure either.

| Set | yomi | tesseract | Time per page, yomi / tesseract |
|---|---|---|---|
| English | **98.2% / 98.2%** | 93.3% / 97.5% | 0.25 s / 1.3 to 2.2 s |
| Western UI | **94.3% / 94.6%** | 85.9% / 91.0% | 0.27 s / 1.7 to 2.6 s |
| CJK UI | **91.8% / 92.1%** | 87.3% / 90.3% | 1.1 s / 9 to 12.5 s |
| Right-to-left UI | **82.5% / 84.0%** | 67.2% / 80.3% | |
| Thai UI | **96.3% / 96.6%** | 89.6% / 91.2% | |
| Hindi UI | 83.9% / 84.9% | **85.0% / 90.6%** | 0.35 s / 2.1 to 3.0 s |
| App layouts, stray words | **58 / 58** | 114 / 105 | 0.2 s / 0.7 to 1.2 s |

### On Windows, against Windows.Media.Ocr

On fresh English text in a Windows VM, Windows.Media.Ocr reads 88.9% at 1x
and 98.1% at 2x, and yomi 97.7% at both. The full comparison across scripts
is still running.

### Speed on x86

CI times a frame read with every recognizer and the router on GitHub's hosted
runners. A 4K window reads in 44 ms on a 4-core arm64 runner and 60 to 110 ms
on 4-core x86 ones, and the densest test frame, small Thai at 2x, in 15 to
21 ms on arm64 and 94 to 121 ms on x86. x86 has no vector kernels yet.

### Size

| Package | Scripts | Weights |
|---|---|---|
| `models/western` | Latin, Cyrillic, Greek | 0.9 MB |
| `models/cjk` | Chinese, Japanese, Korean | 7.2 MB |
| `models/rtl` | Arabic, Persian, Urdu, Hebrew | 1.6 MB |
| `models/indic` | nine Indic scripts | 1.1 MB |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.9 MB |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 1.0 MB |
| `models/router` | which script a line is in | 0.2 MB |

Every script together is 13 MB.

## Where yomi is behind

- **Low-contrast text.** Light grey on dark grey, as in an unfocused window or
  a disabled control, can read as noise. A model trained with dimmed and faint
  text is training now.
- **Large display text and some symbols.** Figures at about 28 pt on a
  dashboard misread, and an arrow beside a word can read as a letter of
  another script.
- **Long prose.** Vision accurate reads an article at 99.6%, yomi at 94.5%.
- **Hindi and Arabic** trail Vision by 5 to 8 points at 2x, and Myanmar,
  Armenian and Amharic sit below 80%.
- **CJK** trails PaddleOCR by about 4 points.
- **Fractional scales.** A capture at 1.25 pixels per point reads about 2
  points below the same text at 1x.

## Checking the numbers

The evaluation pages, the baselines and the scripts that score them are in
`train/`: `evalgen.py` builds the pages, `render.swift` renders them,
`vision.swift`, `paddle.py`, `tesseract.c` and `winocr.ps1` read them with
each engine, `yomi bench` reads them with yomi, and `pagescore.py` scores every
run the same way.
