# Benchmarks

How yomi compares with the OCR each platform ships, and with PaddleOCR. yomi's
numbers are from 2026-11-26. The other engines' runs are from 2026-11-22 on
the same pages, unless a section says otherwise. Results read as 1x / 2x.

## The sets

| Set | What it is | Who can run it |
| :-- | :--------- | :------------- |
| Open set | 127 interface pages in 42 languages, from Firefox's translations, drawn in fonts training never saw | anyone, on any platform |
| UI sets (`ui-*`) | Pages of the Mac's own interface strings in Apple's fonts, 11 to 16 px, light and dark | a Mac, not published |
| Held-out sets (`ho-*`) | Training-style lines from the fifth of each word list training never saw, in held-out fonts | a Mac |
| Code, Linux, layouts | English code and prose, the same rendered on Linux, and 11 app scenes (mail, chat, editor, Finder, settings, tables, a dashboard, an article) | a Mac |
| Websites | 43 development sites and 77 held out from all development, rendered at 1x and 2x | a Mac |
| Live captures | 28 captures of apps and web pages in 15 languages, scored against each element's accessibility text | private |

Western, right-to-left and Indic text is scored per word: a word counts when
a read word with exactly its text sits inside its box. CJK and Southeast Asian
scripts are scored per character: each character read goes to the
ground-truth word its box sits in, and each word scores by edit distance.
yomi gives a box per character. An engine that gives only word boxes gets an
even share of the word's box per character, so a box that spans two rows
costs it, as it would cost Neru's hints and cursor. Both sides are
NFKC-normalized, so full-width and ASCII punctuation count as the same. A
sentence that wraps counts line by line.

## On macOS

### Interface text, against Vision and PaddleOCR

| Set | yomi | Vision accurate | PaddleOCR |
| :-- | :--- | :-------------- | :-------- |
| Western, 17 languages | **95.4% / 95.8%** | 86.3% / 86.3% | 85.9% / 86.8% |
| CJK | 96.9% / 97.1% | 92.6% / 92.4% | **98.4% / 98.8%** |
| Arabic | **81.8%** / 84.5% | 81.6% / **87.3%** | 64.1% / 78.9% |
| Hebrew | **86.1% / 86.5%** | 2.2% / 2.3% | no recognizer |
| Hindi | 90.2% / 91.0% | **92.1% / 92.4%** | 83.1% / 86.7% |
| Thai | **98.0% / 98.1%** | 93.3% / 92.6% | 97.1% / 92.7% |
| Lao, Khmer, Myanmar, held out, 1x | **83.9%, 90.6%, 86.7%** | 21.8%, 26.8%, 7.1% | no recognizer |
| Georgian, Armenian, Amharic, held out, 1x | **93.7%, 84.4%, 79.3%** | 15.2%, 12.4%, 19.9% | no recognizer |
| Bengali, Tamil, Telugu, held out, 1x | **82.3%, 71.7%, 71.3%** | 18.8%, 12.5%, 15.4% | not run |
| App layouts, 11 scenes | **94.5% / 94.9%** | 90.2% / 92.4% | not run |
| Code | **99.0% / 98.9%** | 97.2% / 96.9% | not run |

Vision accurate takes 85 ms to 1.8 s a page on these sets and PaddleOCR 4 to
10 s, where yomi takes 16 to 57 ms. Vision's fast mode, which Neru's hints
used, reads Western interface text at 37% / 57% and the other scripts at
3 to 8%.

### The open set, against Vision

| Text | yomi | Vision accurate |
| :--- | :--- | :-------------- |
| 17 Western languages | **98.2% / 98.5%** | 89.6% / 90.2% |
| Chinese, Japanese, Korean | **98.2% / 98.5%** | 95.3% / 95.2% |
| Arabic | 76.3% / 76.7% | **82.6% / 86.3%** |
| Hebrew, Persian, Urdu, 1x | **91.9%, 68.8%, 79.0%** | 0.8%, 33.8%, 21.0% |
| Hindi, Marathi, Nepali, 1x | **89.9%, 93.8%, 86.7%** | 89.2%, 92.9%, 85.5% |
| Eight other Indic scripts | **48% to 83%** | 0% to 5% |
| Thai | **97.9% / 98.4%** | 96.7% / 96.2% |
| Lao, Khmer, Myanmar, 1x | **95.8%, 79.1%, 52.7%** | 3.3%, 1.3%, 0.2% |
| Georgian, Armenian, 1x | **96.1%, 78.1%** | 1.7%, 0.7% |

The held-out fonts for Bengali, Tamil, Telugu, Kannada, Malayalam, Sinhala,
Lao, Khmer and Myanmar are serifs, a style training does not render. Those
scripts score lowest.

### Websites and live captures

| | yomi | Vision accurate |
| :- | :--- | :-------------- |
| Held-out websites: share of Vision's text yomi also reads | 89.4% / 89.2% | |
| Held-out websites: the pages' own text | 81.0% / 81.3% | **81.5%** / 81.3% |
| Held-out websites: time per page | **31 to 33 ms** | 505 to 566 ms |
| Live captures, all 28 | **80.8%** | 65.5% |
| Live captures in scripts Vision reads, 23 | 83.7% | **84.3%** |
| Live captures in Bengali, Tamil, Hebrew, Georgian, Armenian | **76.7%** | 0.6% |

Most of what Vision reads on websites and yomi does not is text inside
pictures, text dimmed behind a cookie or sign-in dialog, and text a few
pixels tall.

### Through Neru

The 28 live captures read through Neru's own path, as its hints and vim mode
use the text. Neru without yomi reads with Vision, fast for hints and
accurate for vim.

| | Neru with Vision | Neru with yomi |
| :- | :--------------- | :------------- |
| Characters of the accessibility text read | 65.5% | **80.6%** |
| Clickable elements whose text was read | 29.7% | **44.7%** |
| Vim cursor at the right character | 65.4% | **86.6%** |
| Stray reads away from text, and on pictures | **108**, 79 | 182, **51** |
| Time per vim read, median | 726 ms | **36 to 41 ms** |

## On Linux, against tesseract

The same pages in a Linux VM. tesseract runs as Neru ran it, LSTM only and
sparse text, with each page's own language data, its best case. yomi was
timed in an idle VM, tesseract with the host busy. yomi's output there
matched the Mac's word for word on every page.

| Set | yomi | tesseract | Time per page, yomi / tesseract |
| :-- | :--- | :-------- | :------------------------------ |
| English, 1x | **99.0%** | 93.0% | 25 / 862 ms |
| Western | **95.4% / 95.8%** | 85.9% / 91.0% | 27 / 1,684 to 2,584 ms |
| CJK | **96.9% / 97.1%** | 82.2% / 85.4% | 87 to 95 / 9,054 to 12,549 ms |
| Right-to-left | **84.0% / 85.6%** | 65.2% / 77.9% | 43 / 1,167 to 1,655 ms |
| Thai | **98.0% / 98.1%** | 90.6% / 92.1% | 65 to 79 / 8,409 to 10,033 ms |
| Hindi | **90.2% / 91.0%** | 85.0% / 90.6% | 45 / 2,065 to 3,022 ms |
| App layouts, stray words | **48 / 57** | 114 / 105 | 25 to 27 / 721 to 1,247 ms |

## On Windows, against Windows.Media.Ocr

The same pages in a Windows 11 arm64 VM, Windows.Media.Ocr with each page's
own language pack, its best case. Windows has no pack for Hebrew, Ukrainian
or Vietnamese, so those pages are left out. yomi's output matched the Mac's
word for word.

| Set | yomi | Windows.Media.Ocr | Time per page at 2x, yomi / Windows |
| :-- | :--- | :---------------- | :---------------------------------- |
| Code | **99.0% / 98.9%** | 88.9% / 98.1% | 37 / 192 ms |
| Western, 15 languages | **95.8% / 96.1%** | 86.8% / 93.5% | 46 / 289 ms |
| CJK | **96.9% / 97.1%** | 88.6% / 95.9% | 156 / 1,006 ms |
| Arabic | **81.8%** / 84.5% | 54.9% / **85.3%** | 74 / 1,692 ms |
| App layouts, stray words | **48 / 57** | 70 / 96 | 40 / 252 ms |

## Fractional scales

Displays scaled to 125%, 150% or 175% draw text at 1.25, 1.5 or 1.75 pixels
per point. These pages are the 2x pages downsampled with a Lanczos filter,
as macOS and GNOME downsample a scaled display, and read at their own scale.

| Set | 1x / 2x | 1.25x | 1.5x | 1.75x |
| :-- | :------ | :---- | :--- | :---- |
| Western | 95.4% / 95.8% | 95.3% | 95.5% | 95.5% |
| CJK | 96.9% / 97.1% | 96.4% | 97.0% | 96.8% |
| Right-to-left | 84.0% / 85.6% | 85.0% | 85.0% | 85.9% |
| Hindi | 90.6% / 90.0% | 90.5% | 90.5% | 89.7% |
| Thai | 98.0% / 98.1% | 98.1% | 98.1% | 98.1% |
| Code | 99.0% / 98.9% | 98.4% | 99.1% | 98.9% |
| App layouts | 94.5% / 94.9% | 94.3% | 94.7% | 94.2% |

The Hindi row was read with the Indic model before its last fine-tune.

## Speed on x86 and arm64

CI's `bench` job reads five frames with every recognizer and the router on
GitHub's hosted runners, on all of each runner's cores. Median of five, in
ms per frame, measured 2026-11-22. Kernel work since then made every frame
another 1.2 to 1.3 times as fast.

| Frame | Apple M1, 3 vCPU | Linux arm64, 4 vCPU | EPYC 7763 (Zen 3), 4 vCPU | EPYC 9V45 (Zen 5), 4 vCPU |
| :---- | ---: | ---: | ---: | ---: |
| 1080p window, 11 pt | 18 | 18 | 43 | 24 |
| 4K window, 11 pt | 45 | 48 | 108 | 62 |
| Thai, dense, 2x | 49 | 41 | 127 | 76 |
| Myanmar, dense, 2x | 33 | 32 | 97 | 58 |
| Korean, 1x | 24 | 31 | 88 | 58 |

## Size

The weights ship as half floats. The six family recognizers and the router
come to 7.1 MB, and each package is 0.1 to 3.6 MB. Half precision reads every set
within 0.2 points of float32.

## Where yomi is behind

- **Arabic** trails Vision by 2.8 points at 2x on interface text and by 6 to
  10 on the open set, and Windows.Media.Ocr by 0.8 at 2x.
- **Hindi** trails Vision by 1.4 to 1.9 points.
- **CJK** trails PaddleOCR by 1.5 to 1.7 points on interface text, nearly
  all of it Simplified Chinese (96% against 99%). Korean trails Vision on the
  open set by a point at 1x.
- **Live pages in scripts Vision reads** average 0.6 points behind, worst on
  Arabic Wikipedia, YouTube and apple.com's faint navigation.
- **Lao, Armenian and Amharic** read 79% to 84%, and Kannada, Telugu, Tamil,
  Malayalam and Sinhala 55% to 77% at 1x. Urdu in Nastaliq reads about 37%.

## Run them yourself

Anyone can score the open set:

```sh
cd train
uv run python fetch_evals.py           # downloads the open set into data/
./evalall.sh mine ../bin/yomi          # scores every set in data/
```

On a Mac, `evalgen.py` and `render.swift` build the UI, held-out, code,
Linux and layout sets from the Mac's own strings. `vision.swift`,
`paddle.py`, `tesseract.c` and `winocr.ps1` read the pages with each engine,
and `pagescore.py` scores every run the same way. `urlrender.swift` renders
the sites in `urls-dev.txt` and `urls-heldout.txt`, and `webscore.py` scores
them. [Training](training.md) covers the pipeline.
