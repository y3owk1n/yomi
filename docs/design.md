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
   needs x86 timing on real hardware first. All timing so far is Apple
   Silicon, and amd64 has half the vector registers the pointwise kernel
   uses.

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
