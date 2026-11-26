# Contributing

yomi gets better through reports of text it reads wrong, open fonts and word
lists for the scripts it reads, trained models, and engine fixes. Most fixes
in [docs/decisions.md](docs/decisions.md) started as one page that read
wrong, so a good report is the most useful thing to send.

## Report a misread

Open an issue at
[github.com/y3owk1n/yomi/issues](https://github.com/y3owk1n/yomi/issues)
with:

- a PNG capture of the text, cropped to the lines that read wrong, at the
  resolution your screen drew it (`screencapture -i -x shot.png` on macOS)
- the scale you read it at: 2 for a Retina capture, 1 for a standard display,
  1.25 or 1.5 for a display scaled between
- what yomi read: `yomi read -model auto -scale 2 shot.png`
- what it should read, and the app or site it came from

A capture is screen content, and an issue is public. Crop out names, email
addresses, messages and anything else you would not post. If the text is
private, set the same words in a page or note in the same app and font, and
capture that instead. A capture you attach may become a test fixture in
`testdata/`, so attach only what you are happy to publish.

Say whether the text was missed or read as something else. A line yomi did
not find is a line finder problem, fixed in `internal/linefind`. A line it
found but misspelled is the recognizer, fixed with training data.

## Add fonts or words

The recognizers learn from lines rendered in open fonts, so a script reads
best in the styles its fonts cover. Add a font by adding its path in
[google/fonts](https://github.com/google/fonts), or a release URL for a font
published elsewhere, to the loop for its script in `train/fetch_fonts.sh`.

- Fonts must be open: OFL-1.1, the Ubuntu Font Licence or Apache-2.0. Never
  add a font that comes with an operating system or an app, even one on your
  own machine. Anyone must be able to rebuild the training set from public
  sources, and the models ship under MIT.
- Leave the `*-heldout` font folders out of training. They exist to
  validate on.

Training words come from [wordfreq](https://github.com/rspeer/wordfreq), or
from Wikipedia through `train/wikitext.py` for languages wordfreq lacks.

## Train a model

Fine-tune the shipped model on any platform, score it on the open
evaluation set, and send it back. [docs/training.md](docs/training.md) walks
through it:

```sh
oku sync && just train-setup
cd train
sh fetch_fonts.sh
uv run python fetch_evals.py
uv run python exported.py ../models/indic      # the flags this model needs
uv run python prerender.py indic data/indic-200000 200000 1
uv run python train.py runs/my-indic --init ../models/indic --data data/indic-200000 \
  --family indic --width 1.25 --arch ctx --head full --hstride 2 --steps 3000 --lr 3e-4
./ft-eval.sh mine indic=runs/my-indic/best.pt
```

A model replaces the shipped one when its own family's sets read better at
1x and 2x on average and no other set reads worse. Before merging, the
maintainer also scores it on the Mac evaluation sets, which cannot be
published, on live captures, and on two sets of real websites.

To send one, export it with `export.py`, copy `model.json` and `weights.bin`
into `models/FAMILY/` and the export's `check/` files into
`models/FAMILY/testdata/`, run `go test ./...`, and open a pull request with
the commands you ran, the data you rendered, and the scores before and after.

Every model follows three rules: train only on lines rendered from open fonts
and the training words, never on an evaluation set or held-out data. Take
labels only from the rendered text, never from another OCR engine, and no
weights from another model. Release the weights under MIT.

## Change the engine

Run `just test`, `just lint` and `just check` before a pull request. A change
to `internal/linefind` or `internal/prep` moves every read, so include its
scores on the evaluation sets, as the commits in the history do. A change to
`internal/prep` needs the same change in `train/prep.py`, or the models train
on input the engine never gives them.

Commit subjects follow conventional commits: `fix(linefind): ...`,
`feat(models): ...`, `docs: ...`.
