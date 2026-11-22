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

Everything generated goes to `data/`, which stays out of the repository.
