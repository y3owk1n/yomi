# Training

yomi's recognizers and router train in `train/`, in Python with PyTorch, and
export to the files the Go engine embeds. This page covers what training
uses, how a model is made, and how one is scored and shipped.
[CONTRIBUTING.md](../CONTRIBUTING.md) has the short version for sending a
model back.

## What training uses

None of this ships. The library holds only the trained weights.

| Input | Licence | Use |
| :---- | :------ | :-- |
| Open fonts from google/fonts, DejaVu, GNU FreeFont | OFL-1.1, UFL-1.0, Bitstream Vera, GPL-3.0 with the font exception | rendering lines, each listed in `train/fonts/LICENSES.md` |
| wordfreq word lists | code Apache-2.0, data CC BY-SA 4.0 | training words in 42 languages |
| Wikipedia article intros (`wikitext.py`) | CC BY-SA 4.0 | words for languages wordfreq lacks |
| OpenCC | Apache-2.0 | Simplified to Traditional Chinese |
| uharfbuzz, freetype-py | Apache-2.0, BSD | shaping and rasterizing |
| PyTorch, NumPy, Pillow, fontTools | BSD, BSD, MIT-CMU, MIT | training |
| Firefox's translations (`openstrings.py`) | MPL-2.0 | the open evaluation set only |
| The Mac's system app strings (`uistrings.py`) | Apple's | the Mac evaluation sets only, never committed or trained on |

Three rules hold for every model:

- Train only on lines rendered from open fonts and the training words.
  Never train or tune on an evaluation set, the held-out words, or the
  held-out websites.
- Labels come from the text that was rendered, never from another OCR
  engine's output, and no weights come from another model.
- The weights are released under MIT.

## Setup

```sh
oku sync                 # the toolchain, including uv
just train-setup         # the Python packages, into train/.venv
cd train
sh fetch_fonts.sh        # the open fonts, into train/fonts/
uv run python fetch_evals.py   # the open evaluation set, into train/data/
```

Training runs on Apple's GPU (MPS), on CUDA, or slowly on the CPU. Everything
generated goes to `train/data/` and `train/runs/`, which stay out of the
repository.

## Fonts and words

`fetch_fonts.sh` downloads every font into a folder per script family:
`train`, `cjk-train`, `rtl-train`, `indic-train`, `sea-train` and
`cau-train`, with a `*-heldout` folder beside each for validation. A few
folders serve one purpose: `display` holds the bold and serif faces headings
use, `fallback` the fonts for signs a primary font lacks, `rtl-nastaliq`
Urdu in Nastaliq, and `indic-deva` the interface sans faces for Devanagari
fine-tunes. The script also rewrites
`fonts/LICENSES.md`. Add a font by adding its path to the loop for its
script. It must be open (OFL-1.1, the Ubuntu Font Licence or Apache-2.0),
never one that comes with an operating system or an app.

`charsets.py` defines each family's characters. `textsrc.py` draws training
text: words from wordfreq or Wikipedia, interface-style tokens (numbers,
times, paths, versions), and an English word now and then. A fifth of each
word list is held out for scoring. `families.py` pairs each family's
characters, languages, text source and font folders.

## Lines

`synth.py` renders one line the way screens draw it: hinted and unhinted, at
1x and 2x, 10 to 32 px, in light, dark, dimmed and coloured themes, with
subpixel fringes, part of the line above or below, controls beside the text,
and the padding the line finder leaves. Arabic, Hebrew, the Indic scripts,
Thai, Lao, Khmer and Myanmar render through HarfBuzz and FreeType, and their
labels run in visual order. `prep.py` turns each crop into the 32-row ink map
the Go engine's `internal/prep` makes, and the two must stay the same.

`prerender.py FAMILY OUT COUNT [SEED]` renders a fixed set on every core to
disk, so training steps do no rendering:

```sh
uv run python prerender.py indic data/indic-200000 200000 1
```

## Pages

Single lines miss what pages do to text: headlines in display fonts, text over
gradients and pictures, chips, and neighbours in the margins. `pagegen.py
OUT COUNT [SEED] [FAMILIES]` writes web-like pages. On a Mac,
`pagerender.swift` renders each page twice, once with its text hidden, and
`pagecrops.py` cuts the lines yomi's own line finder finds (`go build -o
bin/crops ./train/crops` from the repository root builds it) and labels them
from the page's text, dropping any word whose pixels match the textless
render. A fine-tune takes the page set at about a sixth of its data.

## Train

```sh
uv run python train.py runs/my-indic --family indic --data data/indic-200000 \
  --width 1.25 --hstride 2 --steps 40000
```

`--width` scales every layer's channels and `--hstride` is the input columns
per output step: 4, or 2 for scripts with narrow letters or stacked marks.
`--arch ctx` (the default) adds the line-context layers, and `--head gated`
the low-rank head the CJK model uses for its 12,098 characters. A run keeps
`best.pt` by validation word accuracy and `last.pt` for `--resume`.

To fine-tune the model that ships, pass its directory to `--init`:

```sh
uv run python exported.py ../models/indic      # prints the flags it needs
uv run python train.py runs/my-indic --init ../models/indic \
  --family indic --width 1.25 --arch ctx --head full --hstride 2 \
  --data data/indic-200000 data/indic-mine --steps 3000 --lr 3e-4
```

`exported.py` rebuilds the network from `model.json` and `weights.bin`. The
export folded each batch norm into the layer before it, so those batch norms
stay frozen through the fine-tune. Train on the family's own lines as well as
whatever you add, or the model drifts on the scripts and styles you did not
render. 3,000 steps take about 45 minutes on an M5.

The router trains with `train_router.py OUTDIR --data data/router` on lines
from every family, rendered by `prerender.py router`.

## Score

```sh
./evalall.sh mine ../bin/yomi-mine          # every set in data/
./ft-eval.sh mine indic=runs/my-indic/best.pt
```

`evalall.sh` reads every evaluation set present in `data/` with a yomi binary
and scores it with `pagescore.py`. `ft-eval.sh` exports checkpoints into a
yomi build of their own, scores it on every set, the websites and the live
captures where those exist, and puts the committed models back when it is
done. [Benchmarks](benchmarks.md) describes the sets.

A model replaces the shipped one when its own family's sets read better at
1x and 2x on average and no other set reads worse.

## Export

```sh
uv run python export.py runs/my-indic/best.pt data/export-mine
```

`export.py` folds batch norm into each convolution, rounds the weights to
half floats, and writes `model.json`, `weights.bin`, and eight reference
lines with the network's outputs in `check/`. To ship a model, copy
`model.json` and `weights.bin` into `models/FAMILY/` and the files in
`check/` into `models/FAMILY/testdata/`. `go test ./...` then checks the Go
engine against those outputs.

## Script reference

[train/README.md](../train/README.md) lists every script in `train/` with
its usage line.
