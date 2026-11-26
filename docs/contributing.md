# Helping yomi read better

yomi gets better in three ways: reports of text it reads wrong, open fonts
and word lists for the scripts it reads, and training runs that clear the
benchmarks. Most fixes in [design.md](design.md) started as one page that
read wrong, so a good report is the most useful thing to send.

## Report a misread

Open an issue at [github.com/y3owk1n/yomi/issues](https://github.com/y3owk1n/yomi/issues)
with:

- a PNG capture of the text, cropped to the lines that read wrong, at the
  resolution your screen drew it (`screencapture -i -x shot.png` on macOS)
- the scale you read it at: 2 for a Retina capture, 1 for a standard
  display, 1.25 or 1.5 for a display scaled between
- what yomi read: `yomi read -model auto -scale 2 shot.png`
- what it should read, and the app or site it came from

A capture is screen content, and an issue is public. Crop out names, email
addresses, messages and anything else you would not post. If the text is
private, set the same words in a page or a note with the same app and font
and capture that instead. A capture you attach may become a test fixture in
`testdata/`, so attach only what you are happy to publish.

It helps to say whether the text was missed or read as something else. A
line yomi did not find is a line finder problem, and the fix is usually in
`internal/linefind`. A line it found but misspelled is the recognizer, and
the fix is training data.

## Add fonts or words for a script

The recognizers learn from lines rendered in open fonts, so a script reads
best in the styles its fonts cover. `train/fetch_fonts.sh` lists every font
and the folder it trains in. To add one, add its path in
[google/fonts](https://github.com/google/fonts), or a release URL for a font
published elsewhere, to the loop for its script and run the script. It also
rewrites `train/fonts/LICENSES.md`.

- Fonts must be open: OFL-1.1, the Ubuntu Font Licence, or Apache-2.0. Never
  add a font that comes with an operating system or an app, even one you can
  read on your own machine. Anyone must be able to rebuild the training set
  from public sources, and the models ship under MIT.
- Each script keeps a few fonts out of training, in the `*-heldout`
  folders, to validate on. Leave those out of the training folders.

Training words come from [wordfreq](https://github.com/rspeer/wordfreq), or
from Wikipedia through `train/wikitext.py` for languages wordfreq lacks.
Training keeps a fifth of each list out, for scoring. The evaluation sets
also use interface strings collected from apps, and training never sees
them, so the scores measure text the models have not learned.

## Train a model and send it

The pipeline is in `train/`, and [train/README.md](../train/README.md)
describes each script. You need:

- macOS, Linux or Windows. Training runs on Apple's GPU (MPS), on CUDA, or
  slowly on the CPU. The shell scripts need bash, so on Windows run them in
  Git Bash or WSL.
- [oku](https://github.com/y3owk1n/oku) for the toolchain: `oku sync`, then
  `just train-setup` for the Python packages.
- The open evaluation set: `uv run python fetch_evals.py` downloads it into
  `train/data`. It has 127 pages in 42 languages, made from Firefox's
  interface strings (MPL-2.0) in held-out open fonts. Everyone scores the
  same pixels, so your numbers compare with anyone's.

A run fine-tunes the shipped model, with `indic` standing for any family in
`train/families.py`:

```sh
cd train
sh fetch_fonts.sh
uv run python fetch_evals.py
uv run python exported.py ../models/indic      # prints the flags this model needs
uv run python prerender.py indic data/indic-200000 200000 1
uv run python train.py runs/my-indic --init ../models/indic --data data/indic-200000 \
  --family indic --width 1.25 --arch ctx --head full --hstride 2 --steps 3000 --lr 3e-4
./ft-eval.sh mine indic=runs/my-indic/best.pt
```

`--init` takes the exported model in `models/`, so a run starts from the
model that ships. Its batch norms stay frozen, because the export
folded them into the layers before them. 3,000 steps take about 45 minutes
on an M5. Train on the family's own lines as well as whatever you add, as
the run above does, or the model drifts on the scripts and styles you did
not render. `ft-eval.sh` builds a yomi with the new model in place, scores it
on every evaluation set in `train/data`, and puts the committed models back
when it is done.

A model replaces the shipped one when its own family's open sets read
better at 1x and 2x on average and no other set reads worse. Before merging,
the maintainer also scores it on sets built from macOS's own interface
strings, which cannot be published, on live captures, and on two sets of
real websites. To send one:

1. Export it: `uv run python export.py runs/my-indic/best.pt data/export-mine`.
2. Copy `model.json` and `weights.bin` into `models/indic/`, and the files in
   `data/export-mine/check/` into `models/indic/testdata/`, replacing what is
   there. `go test ./...` checks the Go engine against those reference
   outputs.
3. Open a pull request with the commands you ran, the data sets you rendered,
   and the scores `ft-eval.sh` printed before and after.

Three rules hold for every model:

- Train only on lines rendered from open fonts and the training words. Never
  train or tune on an evaluation set, the held-out words, or the held-out
  websites.
- Labels come from the text you rendered, never from another OCR engine's
  output, and no weights come from another model.
- You release the weights under yomi's MIT licence.

## Code changes

Run `just test`, `just lint` and `just check` before a pull request that
changes the engine. A change to
`internal/linefind` or `internal/prep` moves every read, so include its
scores on the evaluation sets, the way the commits in the history do. A
change to `internal/prep` must be mirrored in `train/prep.py`, or the models
train on input the engine never gives them.
