# train

The pipeline that trains yomi's recognizers and router and exports them for
the Go engine. [docs/training.md](../docs/training.md) explains how the
pieces fit together. `just train-setup` installs the Python packages, and
every script runs from this directory with `uv run python SCRIPT`.
Everything generated goes to `data/` and `runs/`, which stay out of the
repository.

## Data

| Script | Does | Usage |
| :----- | :--- | :---- |
| `fetch_fonts.sh` | Downloads the open fonts into `fonts/` and rewrites `fonts/LICENSES.md` | `sh fetch_fonts.sh` |
| `wikitext.py` | Fetches Wikipedia words for languages wordfreq lacks | `wikitext.py LANG [LANG ...]` |
| `charsets.py` | Each family's characters | module |
| `textsrc.py` | Training text: words, interface tokens, held-out fifth | module |
| `families.py` | Pairs each family's characters, languages, text and fonts | module |
| `shaping.py` | HarfBuzz and FreeType rendering for complex scripts | module |
| `synth.py` | Renders one synthetic line as screens draw it | module |
| `prep.py` | A crop to the 32-row ink map, as `internal/prep` does in Go | module |
| `prerender.py` | Renders a fixed set of lines to disk on every core | `prerender.py FAMILY OUT COUNT [SEED]` |
| `pagegen.py` | Writes web-like training pages | `pagegen.py OUTDIR COUNT [SEED] [FAMILY,...]` |
| `pagerender.swift` | Renders pages in WebKit, with and without text (macOS) | `./pagerender DIR [LIST]` |
| `pagecrops.py` | Cuts the lines yomi finds on rendered pages and labels them | `pagecrops.py PAGEDIR OUT [NEG_KEEP]` |
| `crops/` | The Go line finder `pagecrops.py` runs | `go build -o bin/crops ./train/crops` |

## Training

| Script | Does | Usage |
| :----- | :--- | :---- |
| `model.py` | The recognizer network | module |
| `train.py` | Trains or fine-tunes a recognizer | `train.py OUT --family F --data SET... [--init CKPT_OR_DIR]` |
| `train_router.py` | Trains the router | `train_router.py OUTDIR --data data/router` |
| `exported.py` | Loads an exported model back into PyTorch, and prints its flags | `exported.py ../models/FAMILY` |
| `export.py` | Writes `model.json`, `weights.bin` and reference outputs | `export.py CHECKPOINT OUTDIR [FAMILY]` |

## Evaluation

| Script | Does | Usage |
| :----- | :--- | :---- |
| `fetch_evals.py` | Downloads the open evaluation set into `data/` | `fetch_evals.py` |
| `openstrings.py` | Collects Firefox's interface strings for the open set | `openstrings.py L10NDIR OUT.json` |
| `uistrings.py` | Collects the Mac's own interface strings (macOS) | `uistrings.py OUT.json` |
| `evalgen.py` | Writes evaluation pages | `evalgen.py OUTDIR SOURCE [PAGES_PER_LANG] [SEED]` |
| `render_pw.py` | Renders pages in headless Chromium, any platform | `render_pw.py DIR` |
| `render.swift` | Renders pages in WebKit (macOS) | `./render DIR` |
| `mklines.py` | Cuts ground-truth line crops from evaluation pages | `mklines.py EVALDIR OUTDIR` |
| `urlrender.swift` | Renders the sites in `urls-dev.txt` and `urls-heldout.txt` (macOS) | `./urlrender URLS DIR` |
| `pagescore.py` | Scores reads of evaluation pages | `pagescore.py PAGESDIR RUN.jsonl... [--by-lang] [--chars]` |
| `webscore.py` | Scores reads of the website sets | `webscore.py REF.jsonl RUN.jsonl... [--by-page]` |
| `evalall.sh` | Reads and scores every set in `data/` with a yomi binary | `./evalall.sh TAG [BIN]` |
| `ft-eval.sh` | Builds yomi with new checkpoints and scores everything | `./ft-eval.sh TAG FAMILY=CKPT...` |
| `vision.swift` | Reads pages with Apple's Vision (macOS) | `./vision DIR fast\|accurate auto\|en-US REPS` |
| `paddle.py` | Reads pages with PaddleOCR | `uv run --group teacher python paddle.py EVALDIR` |
| `tesseract.c` | Reads pages with tesseract (Linux) | `tesseract-bench TESSDATA EVALDIR [REPS]` |
| `winocr.ps1` | Reads pages with Windows.Media.Ocr (Windows) | `powershell -File winocr.ps1 EVALDIR [REPS]` |

Never train or tune on the held-out words, the held-out websites, or any
evaluation set.
