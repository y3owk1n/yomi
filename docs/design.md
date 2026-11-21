# yomi design

yomi is an OCR library for screen frames, written for Neru: pure Go, no cgo,
weights embedded, the same engine on macOS, Linux and Windows. Neru pulls it
in as a Go module.

## Goals

- Read UI text (app windows, 11-28 px, light and dark, any colour) word for
  word, with word boxes and a confidence, in every script Neru reads today.
- Faster than the native engine on each platform, on CPU alone.
- Lightweight: a few MB per script family, nothing to download or install.
- One code path everywhere. No Vision, no tesseract, no Windows.Media.Ocr.

Non-goals: documents, photos, handwriting, rotated or curved text, layout
analysis beyond lines.

## Why build it (measured, 2026-11-21)

Whole-page reads of the same pages on each platform, word-exact accuracy,
median ms per page. Linux and Windows are UTM VMs on an M5 (4 vCPU each), so
compare times within a platform. Full tables and harnesses: Neru's
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

So the starting point: ahead of every native engine at 1x, level at 2x, about
one point behind PP-OCRv6 tiny at 25-40x its speed, printable ASCII only, and
two fixture phrases short ("CLI" read as "CLl" at 11 pt).

Third-party engines considered and why not (sources in the research README):
PP-OCR is the most accurate but needs ONNX Runtime or ncnn through cgo and
runs 0.6-3 s a page; ocrs is Latin-only and says "early preview"; tesseract is
slow and weak at 1x; pure-Go engines do not exist; vision-language models are
hundreds of millions of parameters.

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
  `model.json` + `weights.bin`. Fonts are fetched, not committed; their
  licences are recorded in `train/fonts/LICENSES.md`.

Measured properties of the current engine: identical text to the PyTorch
model on 3,054 of 3,054 test crops (`TestModel_Forward_MatchesTrainingReference`
pins the logits), 24 ms a page on 10 M5 cores, 42 ms on 4, 130 ms on 1.

## What Neru needs from it

From `internal/ports/vision.go` and vim mode: words with boxes in reading
order grouped into lines (Neru's `line_pieces.go` derives per-character cells,
background and borders from the frame), a fast and an accurate level (one
read can answer both), confidence for `hints.vision.minimum_confidence`,
cancellation, and CJK, where vim's romanized search and initials apply.

## Milestones

Each milestone freezes its gate before training, on a test set sealed from
text no training source uses, and reports against the native engines on all
three platforms with the same harness.

0. **Repo and port** (this commit). Library API, the latin model embedded,
   reference test, fixtures test, cgo-free builds for six targets. Done.
1. **Western scripts at their best.** One model over Latin with accents,
   Cyrillic and Greek (about 350 classes). Targeted data for I/l/1 and
   acronyms, non-text crops labelled empty (stray words on icons and photos
   are the main open cost: 60-170 per 11 layout pages for every engine), more
   layout templates for the line finder. Gate: Neru's fixture test 8/8,
   fresh-text accuracy no worse than today, extras halved, an accents/Cyrillic/
   Greek set at least as good as Vision accurate. Also decides one model per
   family versus one shared model, by comparing ASCII accuracy against the
   current model.
2. **CJK.** zh-Hans, zh-Hant, ja, ko (about 12,000 classes, a factorized
   head so its cost stays bounded). Line finder checked on CJK first (no
   word spaces, no baseline in the Latin sense). Character accuracy as the
   metric. Open fonts for training (Noto Sans/Serif CJK, LXGW WenKai, Nanum,
   M PLUS), system fonts for evaluation only. Gate: at least Vision accurate
   on macOS, above tesseract and Windows.Media.Ocr on their platforms.
3. **Right to left.** Arabic, Persian, Urdu, Hebrew: shaping in training,
   logical-order labels, mixed-direction lines.
4. **South and Southeast Asian scripts.** The nine Indic scripts, Thai, Lao,
   Khmer, Myanmar (stacked marks, no spaces in some), plus Georgian,
   Armenian and Ethiopic: the rest of what Neru's tesseract locale map reads.
5. **Script routing.** A tiny classifier per line picks the model, so a page
   mixing scripts reads without locale configuration.
6. **Neru integration.** A VisionPort adapter on yomi with native capture
   unchanged, then tesseract, the WinRT OCR code and Vision text recognition
   removed. Requires x86 timing on real hardware first (all timing so far is
   Apple Silicon; amd64 has half the vector registers the pointwise kernel
   uses).

## Known risks

- CJK at 1x and 11-13 px: strokes merge. Expect the widest gap to Vision
  there.
- The line finder is tuned on one family of layouts. Real screens (icons,
  KPI numerals, ellipsized text) cost it about 3 points today.
- Every evaluation page so far is synthetic HTML. Real app captures are
  needed before integration.

## Decisions taken without asking (change any)

- Name `yomi` (Japanese for reading), module `github.com/y3owk1n/yomi`, local
  repository only, nothing pushed.
- MIT, Go 1.27, one module with model packages inside it.
- Western scripts first, then CJK.
