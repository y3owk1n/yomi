# Training

These scripts come from Neru's `research/ocr-engine/proto/`, where the
current latin model was trained. Some still carry paths from there. Milestone 1
cleans them up.

- `fetch_fonts.sh` downloads the open training and held-out fonts
  (`fonts/LICENSES.md`).
- `synth.py` renders synthetic lines, `prerender.py` renders a fixed set on
  every core, `train.py` trains (MPS), `export.py` folds batch norm and writes
  `model.json` + `weights.bin` with reference crops for the Go test.
- `evalgen.py` + `render.swift` + `mklines.py` build evaluation pages with
  exact word boxes from WebKit; `score.py` and `e2e.py` score them.

Complex-script shaping (Arabic, Indic, Thai) needs Pillow with raqm. On
macOS, `brew install fribidi` and run Python with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.
