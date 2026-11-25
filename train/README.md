# Training

The pipeline that trains yomi's recognizers and exports them for the Go
engine. `just train-setup` installs its Python and packages with uv (from
oku). Run the scripts with `uv run python SCRIPT` from this directory.

- `fetch_fonts.sh` downloads the open fonts: the training set, the held-out
  set used for validation, and the fallback fonts for signs a font lacks.
  `fonts/LICENSES.md` records each licence and source.
- `charsets.py` defines each model family's characters, `textsrc.py` the
  training text (wordfreq words with a held-out fifth, plus UI tokens), and
  `families.py` pairs them.
- `synth.py` renders synthetic lines and non-text crops, `prerender.py`
  renders a fixed set on every core, `train.py` trains on MPS, and
  `export.py` folds batch norm and writes `model.json` and `weights.bin`
  with the reference crops `internal/nn` tests against.
- `uistrings.py` collects the Mac's own interface strings per language,
  `evalgen.py` builds evaluation pages from them, `render.swift` renders the
  pages in WebKit with exact word boxes, and `mklines.py` cuts line crops.
- `vision.swift` reads the pages with Apple's Vision for the baseline,
  `yomi bench` reads them with yomi, and `pagescore.py` scores both.
  `tesseract.c` and `winocr.ps1` give the baselines Neru has on Linux and
  Windows, and `paddle.py` gives PaddleOCR's.
- `pagegen.py` writes training pages laid out like web UIs: headlines and
  promos in the `display` fonts, text over gradients, textured thumbnails and
  scrims, chips, and plain body text. `pagerender.swift` renders each page
  twice, once with its text hidden. `pagecrops.py` cuts the lines yomi's own
  line finder finds (`go build -o bin/crops ./train/crops` from the repo root
  builds it) and labels them from the DOM. It drops any word whose pixels
  match the textless render, since that word does not show. Fine-tune from
  the shipped checkpoint with the page set at about a sixth of `--data`.
- `urlrender.swift` renders the public sites in `urls-dev.txt` and
  `urls-heldout.txt` offscreen and saves their visible DOM words.
  `vision.swift` and `yomi bench` read them, and `webscore.py` scores a run
  against Vision accurate and the DOM. Never train or tune on the held-out
  sites.
- `evalall.sh` scores a binary on every eval set. `ft-eval.sh` exports
  checkpoints into a scratch binary and runs every benchmark on it.

Everything generated goes to `data/`, which stays out of the repository.
