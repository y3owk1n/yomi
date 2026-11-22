# yomi design

yomi is an OCR library for screen frames, written for Neru. It is pure Go
with no cgo, ships its weights embedded, and runs the same engine on macOS,
Linux and Windows. Neru imports it as a Go module.

## Goals

- Read UI text (app windows, 11-28 px, light and dark, any colour) word for
  word, with word boxes and a confidence, in every script Neru reads today.
- Faster than the native engine on each platform, on CPU alone.
- Lightweight: a few MB per script family, nothing to download or install.
- One code path everywhere. No Vision, no tesseract, no Windows.Media.Ocr.

yomi does not read documents, photos, handwriting, or rotated or curved
text, and it does no layout analysis beyond finding lines.

## Why build it (measured, 2026-11-21)

Whole-page reads of the same pages on each platform, word-exact accuracy,
median ms per page. Linux and Windows are UTM VMs on an M5 (4 vCPU each), so
compare times within a platform. The full tables and harnesses are in Neru's
`research/ocr-engine/README.md` (Benchmark) and `proto/bench/`.

| Platform | Engine | Fresh text 1x | Fresh text 2x | Windows render | Fixtures | ms / page 1x, 2x |
|---|---|---|---|---|---|---|
| macOS | Vision fast (hints today) | 60.8% | 95.9% | 43.9% | 2/8 | 21, 28 |
| macOS | Vision accurate (vim today) | 97.4% | 97.2% | 89.2% | 4/8 | 70, 79 |
| macOS | PP-OCRv6 tiny | 98.6% | 98.7% | 90.9% | 6/8 | 695, 913 |
| macOS | **yomi (latin)** | 97.7% | 97.7% | 88.4% | 6/8 | **22, 24** |
| Linux | tesseract (Neru today) | 93.3% | 97.5% | 80.7% | 8/8 | 348, 603 |
| Linux | **yomi (latin)** | **97.7%** | 97.7% | **88.4%** | 6/8 | **34, 36** |
| Windows | Windows.Media.Ocr (Neru today) | 88.9% | 98.1% | 78.0% | 8/8 | 44, 88 |
| Windows | **yomi (latin)** | **97.7%** | 97.7% | **88.4%** | 6/8 | **36, 39** |

yomi starts ahead of every native engine at 1x and level with them at 2x. It
is about one point behind PP-OCRv6 tiny and 25-40x faster. It reads printable
ASCII only, and two fixture phrases fail because "CLI" at 11 pt reads as "CLl".

The research README cites the sources for the third-party engines that were
ruled out. PP-OCR is the most accurate, but it needs ONNX Runtime or ncnn
through cgo and takes 0.6-3 s a page. ocrs reads Latin only and calls itself
an early preview. Tesseract is slow and weak at 1x. No pure-Go engine exists.
Vision-language models have hundreds of millions of parameters.

## Architecture

```
frame (RGBA, scale)
  -> linefind: luma at 1x, ink by flat-background runs, rules cleared,
     blobs, lines by core band and gap, crop from baseline and x-height
  -> per line, in parallel: prep (luma, ink vs border median, height 32)
  -> nn: conv 3x3, depthwise-separable blocks, row collapse, line context,
     dilated context, 1x1 head, CTC over the model's alphabet
  -> decode: greedy CTC, words split on spaces, boxes from time steps,
     confidence = lowest character probability
  -> []Line{Box, []Word{Text, Box, Confidence}}
```

- `yomi` (root): `Engine`, `New`, `Read`, `ReadLine`, `LoadModel`, `Line`,
  `Word`. Context cancellation between lines, a worker cap.
- `internal/linefind`, `internal/prep`, `internal/nn`: no dependencies
  beyond the standard library.
- `models/<family>`: one package per script family, weights embedded with
  `go:embed`, loaded once. A program imports only the families it reads.
- `train/`: the Python pipeline (PyTorch, Pillow with raqm for shaping) that
  renders synthetic lines from open fonts, trains, and exports
  `model.json` and `weights.bin`. `fetch_fonts.sh` downloads the fonts, so
  the repository holds none. `train/fonts/LICENSES.md` records each licence.

The Go engine reads the same text as the PyTorch model on 3,054 of 3,054
test crops, and `TestModel_Forward_MatchesTrainingReference` pins the logits.
A page takes 24 ms on 10 M5 cores, 42 ms on 4 and 130 ms on 1.

## What Neru needs from it

Neru's `internal/ports/vision.go` and vim mode need these from an engine:

- words with boxes, in reading order, grouped into lines. Neru's
  `line_pieces.go` derives per-character cells, background and borders from
  the frame itself.
- a fast and an accurate level. One read can answer both.
- a confidence per word, for `hints.vision.minimum_confidence`.
- cancellation.
- CJK, which vim's romanized search and initials work on.

## Milestones

Each milestone fixes its pass criteria before training starts. Its test set
uses text that no training source draws from, and its report compares yomi
with the native engines on all three platforms through the same harness.

0. **Repo and port.** Done. The library API, the latin model embedded, the
   reference and fixture tests, and cgo-free builds for six targets.
1. **Western scripts at their best.** One model over Latin with accents,
   Cyrillic and Greek, about 350 classes. Training adds targeted data for
   I/l/1 and acronyms, labels non-text crops as empty, and the line finder
   gets more layout templates. Stray words on icons and photos are the main
   open cost, 60-170 per 11 layout pages for every engine. The gate: Neru's
   fixture test reads 8 of 8, fresh-text accuracy is no worse than today,
   stray words halve, and a set of accented, Cyrillic and Greek text reads at
   least as well as Vision accurate. Comparing ASCII accuracy with the current
   model decides between one model per family and one shared model.
2. **CJK.** zh-Hans, zh-Hant, ja, ko (about 12,000 classes, a factorized
   head keeps its cost bounded). The line finder gets checked on CJK first,
   since CJK has no word spaces and no Latin baseline. The metric is
   character accuracy. Training uses open fonts (Noto Sans and Serif CJK,
   LXGW WenKai, Nanum, M PLUS), and system fonts serve only for evaluation.
   The gate is at least Vision accurate on macOS, and above tesseract and
   Windows.Media.Ocr on their platforms.
3. **Right to left.** Arabic, Persian, Urdu and Hebrew. Training shapes
   the text, labels follow logical order, and lines may mix directions.
4. **South and Southeast Asian scripts.** The nine Indic scripts, Thai, Lao,
   Khmer and Myanmar (stacked marks, and some write no spaces), plus
   Georgian, Armenian and Ethiopic. That covers the rest of Neru's tesseract
   locale map.
5. **Script routing.** A tiny classifier per line picks the model, so a page
   mixing scripts reads without locale configuration.
6. **Neru integration.** A VisionPort adapter on yomi keeps native capture.
   Then tesseract, the WinRT OCR code and Vision text recognition go. This
   needed x86 timing on real hardware first, recorded under x86 timing
   below.

## x86 timing (measured, 2026-11-22)

CI's `bench` job runs `BenchmarkEngine_Read` on GitHub's hosted runners. It
reads each frame with every recognizer and the router on all of the runner's
cores. The table gives the median of five, in ms per frame.

| Frame | Apple M1, 3 vCPU | Linux arm64, 4 vCPU | EPYC 7763 (Zen 3), 4 vCPU | EPYC 9V45 (Zen 5), 4 vCPU |
|---|---|---|---|---|
| 1080p window, 11 pt | 18 | 18 | 43 | 24 |
| 4K window, 11 pt | 45 | 48 | 108 | 62 |
| Thai, dense, 2x | 49 | 41 | 127 | 76 |
| Myanmar, dense, 2x | 33 | 32 | 97 | 58 |
| Korean, 1x | 24 | 31 | 88 | 58 |

A Zen 3 runner takes 2.4 to 3.7 times as long as the M1 and a Zen 5 one 1.3
to 2.4 times. The slowest frame, dense Thai on Zen 3, reads in 0.13 s. In the
first table tesseract takes 348 and 603 ms a page, on the Linux VM.

amd64 leaves 15 vector registers for floats against 32 on arm64, so the four
by four pointwise tile spilled about 40 stack accesses per input channel
there. amd64 takes a four by two tile with 5 instead. With both builds run in
one job on the same runner, the narrower tile reads 1.22 times as fast on Zen
3 and on Zen 5, and up to 1.5 times as fast on the dense frames. It makes
arm64 8 to 25% slower, so arm64 keeps the wider tile. The gated head's
character tile, which CJK reads with, spilled the same way. Two steps per tile
instead of four read the Korean frame 1.22 times as fast on an Intel Xeon
6975P and 1.36 times on Zen 3, and leave the other frames as they were.

AVX2 with FMA now runs the same four kernels as NEON on amd64, chosen at run
time from CPUID, with the scalar tiles for an older CPU. With both builds
timed in one job on the same runner, the fixture frames read 3.1 times as
fast at the geometric mean on Zen 5 and on Zen 4, and dense Thai and Myanmar 5 to 7 times
as fast: small Thai at 2x goes from 85 to 12 ms, a 4K window from 42 to 30 ms.

## Gradient backgrounds (measured 2026-11-22)

A video page in a dark theme draws its title in white over a soft gradient,
YouTube's glow behind the player. The gradient spread over lumas 30 to 45
while the text sat on 239, so the commonest single luma was the text's. The
crop read inverted, the router called it no text, and the title read as
nothing. The background is now the busiest band of 17 lumas, then the
commonest luma in it. Every evaluation set reads exactly as before, and the
title now goes to the CJK recognizer.

That recognizer still reads only half the title's characters right. The title
is bold at about 20 px, beyond training's 10 to 18 px, and it reads perfectly
only from a crop with a fifth of its height below the ink. A minimum of that
much costs Thai UI pages 0.9 points and right-to-left 0.6, so the crop stays
and the fix belongs in training, with larger and bolder type.

## Underlined links in body text (measured 2026-11-22)

The underline eraser looked only at the bottom quarter of a line's crop and
only at ink at least half strong. Under serif body text a link's underline
sits on the quarter's top row. At 2x the eraser cleared the lower of its two
rows and left the upper one, and at 1x the underline falls between two pixel
rows, which leaves each about 40% ink. Either way the underline joined the
words, and "commit move" read as "commit_oxg". A band starting on the
quarter's top row now takes the rows above it whose runs match, and a run
counts when it is faint but even along its length. The faint edge under
Arabic joins varies letter by letter, so it fails that test and stays.

App layouts read 93.7% to 94.1% at 1x and 93.7% to 94.3% at 2x. The article
page goes from 93.3% to 95.7% at 1x and from 93.7% to 96.8% at 2x, and the
settings page gains 0.7 and 2 points. Every other set reads as before. Most of
what the article still misses is its heading, bold serif at 26 px, which the
Western recognizer reads as noise. That needs bold serif display type in
training.

## Fractional scales (measured 2026-11-22)

A display scaled to 125%, 150% or 175% draws text at 1.25, 1.5 or 1.75
pixels per point. Windows laptops ship at those scales, and so do many Linux
desktops. `Read` took a whole scale, and the line finder shrank the frame by
it, so a 1.25x frame read as 2x put text at 0.625 of its size. Thai at 11 px
then read nothing at all. `Read` now takes the scale as a float, and the line
finder resamples a fractional frame to one pixel per point by area. Whole
scales keep their box filters and read exactly as before.

The pages are the evaluation sets' 2x captures resized to each scale with
Lanczos, as macOS and GNOME downsample a scaled display. Each cell gives word
accuracy (characters for CJK and Thai) read as a whole 2x frame, the way Neru
read these frames, and then read at the frame's own scale.

| Set | 1x / 2x | 1.25x | 1.5x | 1.75x |
|---|---|---|---|---|
| Western UI | 95.3% / 95.5% | 93.0% to **95.5%** | 94.9% to **95.2%** | **95.6%** to 95.5% |
| CJK UI | 95.4% / 95.5% | 90.9% to **94.4%** | 93.0% to **95.8%** | 94.4% to **95.4%** |
| Right-to-left UI | 83.1% / 86.0% | 84.8% to **84.9%** | **85.8%** to 85.5% | **86.4%** to 85.4% |
| Hindi UI | 89.5% / 89.3% | 85.9% to **90.3%** | 88.9% to **89.6%** | **90.5%** to 89.7% |
| Thai UI | 97.3% / 97.5% | 88.3% to **96.5%** | **96.8%** to 96.6% | **97.8%** to 96.3% |
| Code | 98.6% / 98.7% | 97.1% to **98.8%** | 98.8% to 98.8% | 98.8% to 98.8% |
| App layouts | 93.7% / 93.7% | 90.1% to **93.4%** | 93.8% to **94.2%** | **93.8%** to 93.5% |

Every set now reads within 2 points of its 1x and 2x scores at every scale,
the bar the Milestone 6 gate set. The largest gains are at 1.25x, the scale
that read worst before. At 1.75x a whole 2x read was already close, and
right-to-left and Thai lose 1 and 1.5 points read at their own scale. A
fractional frame takes about as long to read as the same frame at 1x.

## Answer keys and kernel speed (measured 2026-11-22)

The script that `evalgen.py` puts in each page kept a word only when its text
sat on one line, and skipped any token that wrapped. CJK and Thai write
sentences without spaces, so a wrapped sentence is one token, and every
wrapped sentence dropped out of the key. A wrapped token now counts as one
word per line, from its characters' boxes. Re-rendering every generated set
with the fix gave byte-identical images, so every engine's saved reads were
rescored without reading again. The keys grew by 33% on CJK, 21% on Thai and
4 to 8% on the right-to-left sets, and Western and Hindi stayed the same.
Vision accurate, which reads long sentences well, rose most: CJK from 86.0% to
93.7% at 2x, where yomi rose from 93.7% to 95.5%. Most of what looked like
stray words on the CJK and Thai pages were those sentences.

The kernels gained in the same pass:

- The gated head finishes its sums and adds its bias in assembly, and its
  projection runs through the pointwise kernel. CJK pages read 20 to 35%
  faster with identical output.
- A pointwise layer ends on a vector run overlapping the one before rather
  than a scalar tail, a short last tile runs in the vector kernel into a
  scratch buffer, and the steps of a full head come straight from the
  channel-major output without a transposed copy.
- Depthwise layers run their row tails four or eight wide and their stride-2
  taps over split even and odd columns.
- Pointwise layers run sixteen positions at a time, every tile over one block
  before the next, so the block's inputs stay in L1.

With both builds timed in one job on each GitHub runner, the fixture frames
read 1.19 times as fast at the geometric mean on arm64 Linux, 1.28 on an M1, 1.32
on AMD and 1.34 on Intel, and up to 2 times on dense Arabic and Myanmar.
Allocation per read fell by 40%. Every evaluation set reads the same words at
the same confidences.

## Background and yeh against Windows.Media.Ocr (measured 2026-11-22)

CJK and Arabic trailed Windows.Media.Ocr at 2x. Two faults outside the models
explained most of it.

- Prepare took the background from the median of the crop's border. A line on
  a button whose crop runs past the button's edge has a border that is mostly
  page, so the button's fill became ink and the text background, and the line
  read as nothing. The background is now the luma most of the crop has, since
  a crop fitted to text is mostly background. `train/prep.py` does the same,
  and both give identical inputs on the checked crops.
- Arabic ي and ك draw like Persian ی and ک at the start and in the middle of a
  word, and the right-to-left recognizer often chose the Persian letter. Read
  now settles those positions on the page's word endings, where the letters
  differ and the recognizer reads them right. Letters only Persian and Urdu
  use are no evidence, since the recognizer reads Arabic's dots as them on
  most Arabic pages.

| Set | Before (1x / 2x) | After (1x / 2x) | Windows.Media.Ocr |
|---|---|---|---|
| CJK UI (characters) | 92.5% / 92.2% | 93.6% / 93.7% | 86.1% / 93.4% |
| Arabic UI | 83.3% / 86.2% | 85.2% / 88.2% | 56.3% / 87.4% |
| Hebrew UI | 89.5% / 90.4% | 89.5% / 90.4% | no OCR pack |
| Western UI | 95.1% / 95.3% | 95.3% / 95.5% | 86.8% / 93.5% |
| Persian, held out | 84.2% / 86.5% | 84.8% / 87.6% | |

The CJK "before" column predates the large-type fixes. Hindi, Thai, code and
the held-out pages hold or gain. yomi now passes Windows.Media.Ocr on every
set at both scales.

Of the CJK errors left, the commonest are full-width and half-width
punctuation read for each other, which Windows.Media.Ocr shares. The worst
lines in the CJK set hold text the page draws and its ground truth leaves out.
Of the Arabic errors left, 28 at 2x are words split where the recognizer reads
a space at the gap after a letter that does not join leftward. That one needs
training. Urdu, which WebKit sets in Nastaliq, reads at 38% on the held-out
pages.

The layout scenes' strays are mostly not mistakes. Some are labels the page
draws and its ground truth omits, read at confidence 1.0, and the rest are
icons read as Q, ☆, + or ⇧, which Neru wants as hint targets. No confidence
floor separates them from real words: a floor of 0.6 drops 13 strays and 104
correct words.

## Indic, wider (measured 2026-11-22)

indic-c is indic-b's recognizer at width 1.25 instead of 0.8, 571k parameters
where indic-b has 293k, trained 40,000 steps. It validates at 81.5% of words
where indic-b reached 74.5%. Read through the routed engine, it costs Indic
pages 20 ms a page on the M5 and changes no other script's.

| Set | indic-b (1x / 2x) | indic-c (1x / 2x) | Median ms, b / c |
|---|---|---|---|
| Hindi UI | 84.2% / 85.9% | 89.4% / 89.2% | 28 / 48 |
| Indic, held out | 66.7% / 70.7% | 72.8% / 76.5% | 35 / 54 |
| Layout scenes, Western UI | unchanged | unchanged | |

The Indic weights grow from 0.6 MB to 1.1 MB.

## Large type and underlined links (measured 2026-11-22)

A live run of Neru in the Linux VM read a web page's 26 px bold heading as
noise, its 34 px bold figure as two lines and its underlined links without
spaces. Each traced to one fault.

- The border pass cleared any marked column run 16 px or longer, and the stems
  of large bold text are that tall. It now keeps a run when at least half its
  rows sit in ink three or more pixels wide, which no border or field side is.
- A heading in large type split into two lines at a gap wider than its core
  allowed, and a tall blob that spans both counted as marks fused across two
  rows. When every line a tall blob overlaps shares a row with the best one,
  the lines merge.
- Large letters fuse into a few wide blobs, and a piece inside one gave the
  line its x-height. A baseline blob inside a taller member's columns no longer
  counts.
- Browsers draw an underline under the spaces between linked words. Prepare
  now erases a run of ink at least the crop's height long in the crop's bottom
  quarter, below the baseline, when it forms a band at most a tenth of the
  crop thick and stays clear of the crop's sides. The Devanagari headline and
  Arabic joins run higher, a button's background below its text is thicker,
  and a table rule reaches the sides.

| Set | Before (1x / 2x) | After (1x / 2x) |
|---|---|---|
| Layout scenes | 93.3% / 93.0% | 93.5% / 93.6% |
| Right-to-left UI | 83.8% / 85.1% | 86.1% / 87.6% |
| CJK UI (characters) | 92.5% / 92.2% | 93.1% / 92.9% |
| Thai UI (characters) | 96.3% / 96.6% | 96.5% / 96.8% |

Code, Western, Hindi and the held-out pages read the same. The layout scenes
read 50 and 45 stray words where they read 45 and 43. On the Korean fixture
통계 now reads right and 옵션 reads as 옵선.

## Half-precision weights (measured 2026-11-22)

The models ship their weights as IEEE half floats, widened to float32 once at
load, so every script together drops from 13 MB to 6.5 MB and the arithmetic
is unchanged. `export.py` rounds the folded weights in the PyTorch model
before it computes the check crops, so the Go reference test still holds to
1e-3. Against the float32 weights, every evaluation set reads within 0.2
points at 1x and 2x, the layout scenes and code identically.

## Line finder fixes (measured 2026-11-22)

Large figures on a dashboard read as noise, and following them turned up
three faults in the line finder that touched every page. A stroke wider than
six pixels, as in 26 px bold digits or the long horizontal strokes of Han
characters, counted as background, since background is any even run that
long. Glyphs broke into pieces, and a crop could start below a character's
top stroke, so 不 read as 个 and 正 as 止. A short even run whose level
differs from the runs on both sides, which agree with each other, now counts
as ink. A large figure's comma, too tall for a small mark, became a line of its
own, and with two members a line's baseline was the comma's. A one-blob line
at most half the height of a line it sits in now joins it, and the baseline
is the lower median. Crops that keep a glyph's top strokes run taller than its
ink, so the gap that breaks a word fell from 0.8 to 0.75 line heights.

| Set | Before (1x / 2x) | After (1x / 2x) |
|---|---|---|
| Layout scenes | 92.3% / 91.7% | 93.3% / 92.7% |
| Code | 98.0% / 98.1% | 98.6% / 98.7% |
| Western UI | 94.6% / 94.9% | 95.0% / 95.2% |
| Right-to-left UI | 82.2% / 83.8% | 83.8% / 85.1% |
| Hindi UI | 84.0% / 84.8% | 84.5% / 85.8% |
| CJK UI (characters) | 91.8% / 92.1% | 92.5% / 92.2% |
| Georgian, Armenian, Amharic, held out | 79.3% / 78.2% | 79.8% / 79.9% |

Thai and the held-out Southeast Asian pages hold or gain. The layout scenes
read 44 stray words at 2x where they read 33.

## NEON kernels (measured 2026-11-22)

On arm64 the pointwise convolution, the depthwise rows, ReLU and the gated
head's character dot products run in NEON assembly (`internal/nn/kernels_arm64.s`),
four lanes per instruction, with the scalar Go kernels left for the tails.
The first layer gathers each kernel tap's strided row once and adds it into
every output channel, the gated head pads its steps to a multiple of four so
none takes the scalar path, and the per-step softmax skips classes more than
20 below the best. Those three are plain Go and help every architecture. One
line on one M5 core, training running on the GPU:

| Recognizer | Scalar Go | Now |
|---|---|---|
| Western | 17.5 ms | 5.5 ms |
| CJK | 21 ms | 9.5 ms |
| Router | 3.5 ms | 1.2 ms |

A whole frame through the routed engine on all cores reads in half the time.
The GitHub window takes 155 ms against 315 ms, and the layout scenes 88 ms at
the median against 160 ms. No score changes.

## Milestone 1 gate (fixed 2026-11-21, before training)

The western model replaces the latin model only if all of these hold. The
sets live in `train/data/`, and `evalgen.py` rebuilds them. The UI-string set
uses text that no training source draws from.

| Criterion | Set | Baseline | Bar |
|---|---|---|---|
| Neru's fixtures, word for word | `testdata/fixtures`, 4 frames | 6 of 8 phrases | 8 of 8 |
| Western UI text | `ui-western`: 34 pages, 17 languages, 6,050 words | Vision accurate 86.3% / 86.3% (1x / 2x), latin model 56.7% / 56.5% | at least Vision accurate overall at both scales, and no language more than 3 points below Vision accurate |
| English is no worse | `code`, 4,655 words | 97.7% / 97.7% | at least 97.2% at both scales |
| Linux rendering is no worse | `linux`, 4,653 words | 97.9% | at least 97.4% |
| Fewer stray words | `layout`, 11 app scenes | 140 / 127 extra words | at most 70 at both scales |
| Speed | `code` at 2x, all cores, this M5 | 22.6 ms median | at most 30 ms |

Vision accurate reads only about 10% of the Greek pages: it does not read
Greek even with automatic language detection.

## Milestone 2 to 5 gates (fixed 2026-11-21, before those models finished training)

Vision accurate, with automatic language detection, is the baseline where it
reads a script at all. Where it does not, the bar is absolute. `ui-*` sets
are the Mac's own UI strings, never trained on. `ho-*` sets are the held-out
fifth of the training word lists, a weaker test because their text comes from
the same source as training. Scripts without spaces (CJK, Thai, Lao, Khmer,
Myanmar) are scored per character, the rest per word.

| Family | Set | Vision accurate (1x / 2x) | Bar |
|---|---|---|---|
| CJK | `ui-cjk`, 16 pages, 19,845 characters | 86.8% / 86.0% (zh_CN 88.5, zh_TW 88.6, ja 78.3, ko 95.8 at 1x) | at least Vision overall at both scales, no language more than 3 points below it |
| Right to left | `ui-rtl`, Arabic and Hebrew | Arabic 83.6% / 89.5%, Hebrew 2.3% | Arabic at least Vision, Hebrew at least 85% |
| Indic | `ui-indic` (Hindi), `ho-indic` (11 languages) | Hindi 92.1% / 92.4%, Marathi and Nepali 71-80%, the other scripts 8-25% | Hindi, Marathi and Nepali at least Vision, every other language at least 80% |
| Southeast Asian | `ui-sea` (Thai), `ho-sea` | Thai 94.5% / 94.2%, Lao, Khmer and Myanmar 7-31% | Thai at least Vision, Lao, Khmer and Myanmar at least 85% |
| Georgian, Armenian, Ethiopic | `ho-cau` | 12-21% | each language at least 90% |
| Router | every set above, read through the routed engine | n/a | routed accuracy within 1 point of reading each set with its own family's model |

Vision accurate takes 0.6 to 1.9 s per page on these sets.

## Milestone 6 gate (fixed 2026-11-22, before the Neru adapter)

Each platform switches on its own, Linux first, then Windows, then macOS.
yomi replaces a platform's native engine in Neru only if all of these hold
there. Removing the native engine's code and packaging is a separate change
after that.

Neru reads with one engine on every machine. It has every recognizer and the
router, as `yomi -model auto` builds it. The router picks each line's script,
so Neru drops the locale lookups it makes today to choose tesseract's
languages, Windows' OCR engines and Vision's languages.

| Criterion | Set | Baseline | Bar |
|---|---|---|---|
| Neru's own tests | its fixture and integration tests for text, on the platform | the native engine passes them | they pass, changed only where they name the native engine |
| Accuracy | every `eval-*` set the native engine has language data for, read on the platform | the native engine with the page's own language data, its best case | at least the native engine on each set at both scales |
| Stray words | `layout` | the native engine | at most the native engine and at most 70, at both scales |
| Speed | the same sets, the same machine | the native engine's median ms per page | at most the native engine's |
| Fractional scales | a page captured at 1.25x and at 1.5x | the same page at 1x and 2x | within 2 points |
| x86 speed (Windows) | `BenchmarkEngine_Read` frames on `windows-latest` | Windows.Media.Ocr on the same runner | at most Windows.Media.Ocr's |

Vision has two baselines on macOS. Vision fast is what hints read with and
Vision accurate is what vim reads with.

## Against Windows.Media.Ocr (measured 2026-11-22)

The evaluation sets read in the Windows 11 arm64 VM by `train/winocr.ps1`,
which gives each page an engine for its own language, its best case.
Windows has no OCR pack for Hebrew, Ukrainian or Vietnamese, so those pages
are left out for both engines. Accuracy is the current yomi's over the same
pages. Times are each engine's median ms per page in the VM.

| Set | yomi | Windows.Media.Ocr | yomi, Windows ms at 1x | at 2x |
|---|---|---|---|---|
| Code | **98.6% / 98.7%** | 88.9% / 98.1% | 121, 104 | 138, 192 |
| English (Linux pages) | **98.8%** | 91.5% | 95, 113 | |
| Layout scenes | **93.3% / 93.0%** | 86.2% / 90.8% | 105, 115 | 100, 252 |
| Layout stray words | **45 / 43** | 70 / 96 | | |
| Western UI, 15 languages | **95.4% / 95.6%** | 86.8% / 93.5% | 94, 171 | 94, 289 |
| CJK UI (characters) | **92.5%** / 92.2% | 86.1% / **93.4%** | 420, 591 | 417, 1,006 |
| Arabic UI | **83.5%** / 85.4% | 56.3% / **87.4%** | 118, 1,672 | 133, 1,692 |

Against the milestone 6 gate, Windows passes stray words and speed and
misses accuracy on two sets at 2x. Windows.Media.Ocr reads CJK 1.2 points
better and Arabic 2 points better there. At 1x yomi leads both by 6 and 27
points.

## Against PaddleOCR (measured 2026-11-22)

The same Mac UI and held-out pages, read by three engines: Vision accurate,
PaddleOCR and yomi. PaddleOCR runs through `train/paddle.py`, which uses
rapidocr 3.10 on ONNX Runtime. It reads Chinese and Japanese with PP-OCRv6
small and every other language with its PP-OCRv5 or v4 recognizer, chosen
from the page's language. yomi has to work the language out itself.
Western, right-to-left and Indic count words, CJK and Thai characters. CJK
counts yomi's words as one box per character, as PaddleOCR returns them.
Times are per page on the M5, with training running for yomi's.

| Set | yomi | PaddleOCR | Vision accurate | yomi, PaddleOCR, Vision ms |
|---|---|---|---|---|
| Western, 17 languages | **94.6% / 94.9%** | 85.9% / 86.8% | 86.3% / 86.3% | 130, 4,200, 90 |
| CJK | 92.5% / 92.7% | **96.3% / 96.8%** | 86.8% / 86.0% | 550, 9,600, 1,600 |
| Arabic | 82.3% / 84.3% | 65.6% / 80.9% | **83.6% / 89.5%** | 220, 4,500, 570 |
| Hebrew | **84.0% / 84.7%** | no recognizer | 2.3% / 2.3% | |
| Hindi UI | 84.3% / 85.1% | 83.1% / 86.7% | **92.1% / 92.4%** | 250, 4,500, 600 |
| Thai UI | **96.6% / 96.9%** | 96.2% / 92.2% | 94.5% / 94.2% | 90, 8,400, 1,700 |
| Lao, Khmer, Myanmar, held out, 1x | **86.7%, 90.5%, 78.7%** | no recognizer | 22.5%, 29.2%, 7.4% | 70, n/a, 1,000 |
| Georgian, Armenian, Amharic, held out, 1x | **91.6%, 80.9%, 69.2%** | no recognizer | 15.3%, 12.5%, 19.9% | 42, n/a, 720 |

PaddleOCR's Latin recognizer reads Vietnamese at 40%, Romanian at 67% and
Turkish at 71%, where yomi reads 89 to 97%, and its CJK lead is 8 points on
Simplified Chinese. Of the 11 held-out Indic languages it has recognizers for
6. It reads Tamil and Telugu better than yomi, 78% and 76% at 2x against 71%
and 49%, and Kannada at 18%, where yomi reads 68%. It has none for Hebrew, Bengali,
Gujarati, Punjabi, Malayalam, Sinhala, Lao, Khmer, Myanmar, Georgian,
Armenian or Amharic. That makes it a teacher for CJK, Tamil and Telugu, and
not for Arabic.

The right-to-left model is 1.25 times as wide as the first, at 1.6 MB, and
trained from scratch on all 900,000 right-to-left lines. It reads Arabic 5.1
and 2.7 points better and Hebrew 2.3 and 2.7 points better, at 1x and 2x,
and a page takes about twice as long. Arabic is still 1.3 and 5.2 points
behind Vision, and Hebrew 1 and 0.3 points short of its bar.

Thai, Lao and Khmer pass their bars, Myanmar does not. The first Thai model
read monospace text at 61%, where Vision reads 98%. Training had no
monospace Thai font, and with the TLWG fonts it reads 96%. The line finder
also merged rows of small, tightly set Thai and Myanmar, whose marks touch
across rows. Grouping the fused blobs after the rest lifted Thai from 93% to
97% and Myanmar from 63% to 79%. Myanmar trained on three font families
and is held out on a fourth, a serif. Of the Caucasian-Ethiopic languages,
Georgian passes. Armenian reads 81% and Amharic 69%. The held-out serif draws some of
their letters unlike any training font.

The held-out Georgian pages were regenerated on 2026-11-22. The generator had
capitalized the first letter of Georgian words as it does Latin, which
Georgian never does, and a third of the ground-truth words began with a
Mtavruli capital. The words and layout are the same. Vision's score did not
change.

## Routed reads (measured 2026-11-22)

One engine with Western as its base, the CJK, right-to-left, Indic,
Southeast Asian and Caucasian-Ethiopic recognizers, and router-3. The router gate allows a
1-point loss against each set's own model.

| Set | Own model (1x / 2x) | Routed (1x / 2x) |
|---|---|---|
| Western UI | 94.6% / 94.9% | 94.3% / 94.6% |
| CJK UI | 91.9% / 92.2% | 91.8% / 92.1% |
| Right-to-left UI | 83.1% / 84.5% | 82.5% / 84.0% |
| Hindi UI | 84.3% / 85.1% | 83.9% / 84.8% |
| Indic, held out | 66.1% / 69.5% | 65.7% / 69.1% |
| Thai UI | 96.6% / 96.9% | 96.3% / 96.6% |
| Southeast Asian, held out | 87.6% / 85.9% | 86.1% / 84.0% |
| Caucasian-Ethiopic, held out | 80.0% / 78.6% | 79.4% / 78.3% |
| Code | 98.0% / 98.1%, 5 extra words | 98.2% / 98.2%, 13 extra words |
| Layout scenes | 92.1% / 91.9%, 41 extra words | 92.2% / 92.0%, 72 extra words |

Southeast Asian misses the gate. Router-3 sends some Myanmar, Lao and
Khmer lines in the held-out serif fonts to Western, and finds no text in a
few Khmer ones. A line whose runner-up family comes within 2 nats per step
of the first is read by both recognizers, and the surer read kept. That
recovers 0.7 points on those pages and reads about 4% of Western UI lines
twice. The router sends some icons to CJK, which reads a round one as 〇
and others as Han or Hangul characters where Western reads nothing. That
nearly doubled the stray words on the layout scenes. A routed read now drops
a word that is a lone 〇 when a recognizer other than the base read it, which
takes the layout scenes from 75 / 77 stray words to 58 / 60 and changes no
score on any set, since no page has a 〇 to read. Mail's unread dots were 16
of them at each scale. Dropping every lone
symbol took the stray words to 42 / 43 but lost the ☆ that mail draws as
text, so ✓, ✕ and the rest stay.

The router also sent icons and the short tokens beside them on an English
page to the other recognizers: 45 of the 135 lines of a GitHub page, plus 25
second reads, each a recognizer run that read garbage. Its first choice leads
the runner-up by 20 to 35 nats per step on a line written in a script it
knows, and by under 16 on those. So Read now routes every line before it
reads any, and a family is on the page only once it leads a line at least two
em wide by 12 or more. A line routed to a family not on the page reads with
the base recognizer alone. The layout scenes drop from 58 / 58 stray words to
42 / 33, as few as Western alone gives, code from 11 / 14 to 5 / 6, and every
UI and held-out set stays within 0.6 points. The GitHub page reads in about
30 to 50% less time.

That rule also fixed what looked like a weakness on low-contrast text. A
dimmed banner in an unfocused window read as mixed Latin, Greek and Cyrillic
garbage, because the router sent its faint lines to other scripts'
recognizers. Read on its own, the Western recognizer reads it word for word.
A Western model fine-tuned on dimmed and faint text, 18 to 40 px figures and
symbols beside words read every Western set the same, and the layout scenes
with twice the stray words, 76 / 74, so it was not shipped. Figures at about
28 pt on a dashboard still misread with either model.

Two attempts at the Southeast Asian gap changed nothing. Router-4,
fine-tuned from router-3 on lines with the current fonts, scored within 0.3
points of it on every set. Reading a line with the runner-up family too
whenever the first read's mean character probability fell under 0.85 to
0.96 moved held-out Southeast Asian by at most 0.3 points and added 30 to 40
stray words on the layout scenes. A recognizer reading a script it never
trained on can still be sure of its characters, so comparing two reads by
that measure does not tell which recognizer was right.

## How the later milestones work

- **Large alphabets (CJK).** The CJK family has 12,098 characters, from
  GB2312, Big5 level 1, JIS X 0208 level 1 and KS X 1001, plus kana,
  punctuation and what the Mac's own CJK UI strings use (over 99.9% of
  their characters). A full output layer would cost more than the rest of the
  network, so the CJK model uses a gated head. A one-output layer gives each
  step's blank logit, and a low-rank pair of layers gives the character
  logits only where a character can win (`internal/nn`, `train/model.py`).
  Training is ordinary CTC, and the Go decision is the exact argmax.
- **Shaping.** Arabic, Hebrew, the Indic scripts, Thai, Lao, Khmer and
  Myanmar render through HarfBuzz (uharfbuzz) and FreeType (freetype-py),
  both pip wheels, so nothing comes from a system package manager. Lines lay
  out in visual order and their labels follow it. The engine turns
  right-to-left words back into logical order.
- **Lookalikes.** Latin, Cyrillic and Greek letters that draw the same take
  the script of the rest of their word, or of their line.
- **Routing.** A router is the recognizer's network with one class per
  family plus none. Every step learns its line's family, and the engine
  averages the steps and reads the line with that family's recognizer.
- **Character boxes.** Each word carries one box per character, meeting its
  neighbours halfway between the steps they were read at, for vim mode.

## What training uses, and its licences

None of this ships. The library holds only the trained weights.

| Input | Licence | Use |
|---|---|---|
| Open fonts from google/fonts, DejaVu, GNU FreeFont | OFL-1.1, UFL-1.0, Bitstream Vera, GPL-3.0 with the font exception | rendering training lines, listed in `train/fonts/LICENSES.md` |
| wordfreq word lists | code Apache-2.0, data CC-BY-SA 4.0 | training words in 42 languages |
| Wikipedia article intros (`wikitext.py`) | CC BY-SA 4.0 | words for the languages wordfreq lacks |
| OpenCC | Apache-2.0 | Simplified to Traditional Chinese |
| uharfbuzz, freetype-py | Apache-2.0, BSD | shaping and rasterizing |
| PyTorch, NumPy, Pillow, fontTools | BSD, BSD, MIT-CMU, MIT | training |
| The Mac's system app strings (`uistrings.py`) | Apple's | evaluation only, never committed or trained on |

## Known risks

- CJK strokes merge at 1x and 11-13 px, so the gap to Vision will be
  widest there.
- The line finder is tuned on one family of layouts. Real screens (icons,
  KPI numerals, ellipsized text) cost it about 3 points today.
- Every evaluation page so far is synthetic HTML. Real app captures are
  needed before integration.

## Open decisions

- Name `yomi` (Japanese for reading), module `github.com/y3owk1n/yomi`, local
  repository only, nothing pushed.
- MIT, Go 1.27, one module with model packages inside it.
- Western scripts first, then CJK.
