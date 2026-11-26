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
  weights embedded, from 0.1 MB to 3.6 MB. A program links only the ones it
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
  decoding. On arm64 the hot loops run in NEON assembly, on amd64 in AVX2
  with FMA, and in plain Go on a CPU without them.
- **Private by construction.** No network, no logs. Frames and text live only
  for the call.

## How it reads a frame

```
frame (RGBA, scale)
  -> line finder: luma at 1x, ink by flat-background runs, rules and borders
     cleared, blobs grouped into lines, and a second pass at low contrast
     that keeps faint lines on plain ground
  -> per line, on every core: an ink map 32 rows tall, measured against the
     background around each pixel, so text on a bar or a gradient reads
  -> router: each line's script, then the scripts the page is written in
  -> that script's recognizer: convolutions, line context, CTC over its alphabet
  -> words split on spaces and wide gaps, a box per character, confidence the
     least sure character's probability, right-to-left words in reading order
```

## How it compares

### On macOS, against Vision and PaddleOCR

Word accuracy (character accuracy for CJK and Thai) on pages no training text
came from, at 1x and 2x. The UI sets hold Apple's own interface strings,
rendered by WebKit at 11 to 16 pixels in light and dark. The held-out sets
hold the fifth of the training word lists that training never saw. The
answer keys count a token that wraps, as a CJK or Thai sentence with no
spaces does, line by line. yomi's numbers are from 2026-11-26, the other
engines' from 2026-11-22.

| Set | yomi | Vision accurate | PaddleOCR |
|---|---|---|---|
| Western UI, 17 languages | **95.4% / 95.8%** | 86.3% / 86.3% | 85.9% / 86.8% |
| CJK UI | 96.6% / 96.9% | 94.1% / 93.7% | **97.8% / 98.1%** |
| Arabic UI | **81.8%** / 84.5% | 81.6% / **87.3%** | 64.1% / 78.9% |
| Hebrew UI | **86.1% / 86.5%** | 2.2% / 2.3% | no recognizer |
| Hindi UI | 90.6% / 90.0% | **92.1% / 92.4%** | 83.1% / 86.7% |
| Thai UI | **97.8% / 98.0%** | 94.7% / 94.7% | 97.2% / 93.3% |
| Lao, Khmer, Myanmar, held out, 1x | **83.8%, 90.6%, 86.8%** | 22.1%, 28.2%, 7.2% | no recognizer |
| Georgian, Armenian, Amharic, held out, 1x | **93.7%, 84.4%, 79.3%** | 15.2%, 12.4%, 19.9% | no recognizer |
| Bengali, Tamil, Telugu, held out, 1x | **82.3%, 70.1%, 71.6%** | 18.8%, 12.5%, 15.4% | not run |
| App layouts, 11 scenes | **94.5% / 94.9%** | 90.2% / 92.4% | not run |

### On real websites and live captures

The UI sets are clean text on plain ground. Real pages put text on photos,
bars and gradients, dim it behind dialogs and set headlines in large bold
type. Two sets measure that, both read with every recognizer and the router.

- **Websites.** 77 sites, rendered at 1x and 2x and held out from all
  development. Their reads are scored, and the pages are never looked at.
- **Live captures.** 28 captures of apps and web pages in 15 languages, taken
  from a real screen, scored against each element's accessibility text.

| | yomi | Vision accurate |
|---|---|---|
| Websites: share of the text Vision reads that yomi reads, 1x / 2x | 89.4% / 89.2% | |
| Websites: the pages' own text, 1x / 2x | 81.0% / 81.3% | **81.5%** / 81.3% |
| Websites: time per page | **31 to 33 ms** | 505 to 566 ms |
| Live captures, all 28 | **80.7%** | 65.5% |
| Live captures in scripts Vision reads, 23 | 83.5% | **84.3%** |
| Live captures in Bengali, Tamil, Hebrew, Georgian and Armenian | **76.7%** | 0.6% |

What Vision reads and yomi does not is mostly text inside pictures, text
dimmed behind a cookie or sign-in dialog, and text a few pixels tall.

### In Neru's vim mode, against Vision

### In Neru's vim mode, against Vision

Measured 2026-11-22. The same captures through Neru's line-reading path, on one M5 with nothing
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
Measured 2026-11-22 with that day's build, which read the same pixels to the
same words on every platform. The Mac hosting the VM was busy while they ran,
so the times compare the two engines rather than measure either.

| Set | yomi | tesseract | Time per page, yomi / tesseract |
|---|---|---|---|
| English, 1x | **98.8%** | 93.0% | 0.25 s / 1.3 to 2.2 s |
| Western UI | **95.3% / 95.5%** | 85.9% / 91.0% | 0.27 s / 1.7 to 2.6 s |
| CJK UI | **95.4% / 95.5%** | 88.2% / 91.9% | 1.1 s / 9 to 12.5 s |
| Right-to-left UI | **84.8% / 86.6%** | 65.2% / 77.9% | |
| Thai UI | **97.3% / 97.5%** | 89.8% / 91.2% | |
| Hindi UI | **89.5%** / 89.3% | 85.0% / **90.6%** | 0.35 s / 2.1 to 3.0 s |
| App layouts, stray words | **50 / 46** | 114 / 105 | 0.2 s / 0.7 to 1.2 s |

### On Windows, against Windows.Media.Ocr

Both read the same pages in a Windows 11 VM on 2026-11-22, Windows.Media.Ocr
with each page's own language pack. Windows has no pack for Hebrew, Ukrainian or
Vietnamese, so those pages are left out.

| Set | yomi | Windows.Media.Ocr | Time per page at 2x, yomi / Windows |
|---|---|---|---|
| Code | **98.6% / 98.7%** | 88.9% / 98.1% | 138 / 192 ms |
| Western UI, 15 languages | **95.6% / 95.9%** | 86.8% / 93.5% | 94 / 289 ms |
| CJK UI | **95.4% / 95.5%** | 87.4% / 94.6% | 417 / 1,006 ms |
| Arabic UI | **83.1% / 86.0%** | 54.9% / 85.3% | 133 / 1,692 ms |
| App layouts, stray words | **45 / 43** | 70 / 96 | 100 / 252 ms |

### Speed on x86

On amd64 the hot loops run in AVX2 with FMA when the CPU has them, chosen at
run time. On GitHub's hosted x86 runners that reads the test frames 3.1
times as fast as plain Go at the geometric mean. Later kernel work made
reads another 1.2 to 1.3 times as fast on x86 and arm64 alike: a 4K window
reads in 28 ms on an AMD runner and small Thai at 2x in 10 ms.

### Size

| Package | Scripts | Weights |
|---|---|---|
| `models/western` | Latin, Cyrillic, Greek | 0.5 MB |
| `models/cjk` | Chinese, Japanese, Korean | 3.6 MB |
| `models/rtl` | Arabic, Persian, Urdu, Hebrew | 0.8 MB |
| `models/indic` | nine Indic scripts | 1.1 MB |
| `models/sea` | Thai, Lao, Khmer, Myanmar | 0.5 MB |
| `models/caucasian` | Georgian, Armenian, Ethiopic | 0.5 MB |
| `models/router` | which script a line is in | 0.1 MB |

Every script together is 7.1 MB. The weights ship as half floats, which
read every evaluation set within 0.2 points of float32 and halve the size.

## Where yomi is behind

- **Live pages in scripts Vision reads.** yomi averages 0.8 points behind.
  The largest gaps are BBC Thai (72.0% against 81.9%, a bold headline row
  whose crop comes out a few pixels short), Arabic Wikipedia (61.4% against
  67.0%), YouTube (77.5% against 82.3%) and apple.com (84.2% against 88.0%,
  faint navigation at low contrast).
- **Websites.** yomi reads 89% of the text Vision reads. Text in pictures,
  behind dialogs and at a few pixels tall makes up most of the rest.
- **Hindi** trails Vision by 1.5 to 2.4 points and **Arabic** by 2.8 at 2x.
- **CJK** trails PaddleOCR by about 1.2 points.
- **Lao, Armenian and Amharic** read 79 to 84%, and **Kannada, Telugu,
  Tamil, Malayalam and Sinhala** 54 to 77% at 1x.
- **Urdu** in Nastaliq reads about 37%.

## Checking the numbers

The evaluation pages, the baselines and the scripts that score them are in
`train/`: `evalgen.py` builds the pages, `render.swift` renders them,
`vision.swift`, `paddle.py`, `tesseract.c` and `winocr.ps1` read them with
each engine, `yomi bench` reads them with yomi, and `pagescore.py` scores every
run the same way. `urlrender.swift` renders the sites listed in
`urls-dev.txt` and `urls-heldout.txt`, and `webscore.py` scores the reads.
The renders and the live captures stay out of the repository, since the
sites are other people's pages and the captures are of a personal screen.
