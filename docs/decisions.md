# Design decisions

What yomi chose, what it tried and dropped, and the measurement behind each.
Entries run oldest first within each part. The commit history has the full
numbers for every change, and [benchmarks.md](benchmarks.md) has where yomi
stands now.

## Why yomi exists

Neru, a keyboard tool, puts hints on what you can click and moves a cursor
through any window's text, so it needs every word on screen with its box,
in every script, within a keystroke, on macOS, Linux and Windows. In
November 2026 it read with each platform's own OCR, and each fell short:

- Vision's fast mode, which hints used, misread most non-English interface
  text. Its accurate mode took up to 0.7 s a frame and compiled its models
  for up to a minute the first time it met a script.
- tesseract on Linux was several times slower, needed a language package per
  script, and made the user choose languages.
- Windows.Media.Ocr needed an OCR pack per language.
- PaddleOCR was the most accurate on Chinese, but needs ONNX Runtime or ncnn
  through cgo and took seconds a page. ocrs read Latin only. No pure-Go
  engine existed, and vision-language models have hundreds of millions of
  parameters.

A first ASCII model already read fresh interface text at 97.7%, ahead of
every native engine at 1x and level at 2x, in 22 to 36 ms a page.

## Scope

- Read interface text, word for word, with boxes and a confidence, in every
  script Neru reads.
- Faster than each platform's own OCR, on the CPU alone.
- A few MB per script family, nothing to install or download.
- One code path everywhere, so a fix on one platform holds on the others.
- Not documents, photos, handwriting, or rotated or curved text, and no
  layout beyond lines.

## How models were judged

Each model family had its pass criteria fixed before it trained, on text no
training source draws from: the Mac's own interface strings, and later the
open set. A family had to read at least as well as Vision accurate where
Vision reads its script, and at least 85% or 90% where Vision does not.
Before yomi replaced a platform's OCR in Neru, it had to pass Neru's tests
there, read every set at least as well as that engine at 1x and 2x, make no
more stray words, read no slower, and read fractional scales within 2 points
of whole ones. On 2026-11-26 it met that bar everywhere except Arabic at 2x
(Vision and Windows.Media.Ocr), Hindi (Vision), and held-out Southeast Asian
text at 2x, 0.2 points short of tesseract.

## The engine

**Pure Go, small convolutional recognizers with CTC (2026-11-21).** A network
built from 3x3, depthwise and 1x1 convolutions is simple to write by hand in
Go and small enough to run per line on every core. The Go forward pass reads
the same text as the PyTorch model on every test crop, and a reference test
pins the outputs.

**One recognizer per script family, and a router (2026-11-21).** One model
per family keeps each small and lets a program link only the scripts it
needs. A router network names each line's family, so no one configures a
language.

**A gated head for CJK (2026-11-22).** 12,098 characters in a full output
layer would cost more than the rest of the network. A one-output layer gives
each step's blank score, and a low-rank pair gives character scores only
where a character can win, so inference skips most steps.

**Width 1.25 for Indic and right-to-left (2026-11-22).** At width 0.8 Indic
validated at 74.5% of words, at 1.25 at 81.5%, and Hindi interface text rose
from 84 to 89%. The weights grew from 0.6 to 1.1 MB and Indic pages take 20
ms longer. The wider right-to-left model read Arabic 2.7 to 5.1 points
better.

**Half-precision weights (2026-11-22).** Every script together went from 13
MB to 6.5 MB, and every set reads within 0.2 points of float32.

**Hand-written kernels (2026-11-22).** NEON on arm64 and AVX2 with FMA on
amd64, chosen at run time, with plain Go for the tails and older CPUs. A
line reads 2.2 to 3.2 times as fast as in plain Go. amd64 has half arm64's vector
registers, so its pointwise tile is four by two instead of four by four,
which reads 1.22 times as fast there and would slow arm64.

## Routing

**Route every line before reading any (2026-11-22).** The router sent icons
and the short tokens beside them on English pages to the CJK and Arabic
recognizers, which read them as characters. A family now counts as on the
page only once it leads a line two em wide by a clear margin. The layout
scenes went from 58 stray words to 42 and 33, as few as Western alone gives.

**Second reads (2026-11-22).** A line whose runner-up family comes within 2
nats per step is read by both recognizers and the surer read kept. Rejected:
a second read whenever confidence is low, since a recognizer reading a script
it never trained on can still be sure of its characters.

**A lone 〇 from the CJK recognizer is dropped (2026-11-22).** It reads round
icons as 〇. Dropping every lone symbol went further but lost the ☆ that
mail apps draw as text.

**Not adopted: a Western model fine-tuned on dimmed text (2026-11-22).** It
read every set the same and doubled the stray words. The dimmed banner it
aimed at read right once routing stopped sending it to other recognizers.

## Finding lines

**Ink against flat runs, not a global threshold (2026-11-21).** Screen
backgrounds are flat runs of colour, so runs of even luma mark background and
anything that stands out from the runs on both sides is ink. That works on
any theme without a threshold per page.

**Wide strokes are ink (2026-11-22).** A stroke wider than six pixels, in
bold figures or Han characters, counted as background and broke glyphs, so
不 read as 个. A short run that differs from agreeing background on both
sides now counts as ink. Every set gained, by 0.1 to 1.6 points.

**Large type (2026-11-22, 2026-11-25).** Bold stems no longer count as
borders, a heading split at a wide gap merges back, and pieces inside a
larger glyph do not set the x-height. Held-out websites rose from 84.8 to
86.7% at 1x.

**Underlines (2026-11-22).** Browsers draw an underline under the spaces
between linked words, which joined them. The eraser now takes faint but even
runs below the baseline. An article page rose 2.4 to 3.1 points.

**Faint text (2026-11-25, 2026-11-26).** A second pass at a lower edge finds
grey text, kept only on plain ground. A lower edge everywhere cost held-out
websites 1.5 points. The plain-ground test first required the whole ring to
spread over 8 levels, which text beside a line failed. It now leaves out
every line's ink and needs four fifths of the rest within 8 levels.
apple.com rose from 74.5 to 84.2%.

**Fragments beside a line do not vote on its baseline (2026-11-26).** Where
large type crosses the edge of a picture, the picture leaves 5 px fragments
that outvoted the glyphs. Hindi, Armenian and Chinese Wikipedia gained 0.4
to 0.8 points. Rejected: refusing tall blobs that would grow a line, which
broke large headings whose core is lowercase.

**Tight rows stay apart (2026-11-26).** A photo's texture stretched one row
of a bold headline over the next until the two merged. Lines of three or more
glyphs with similar cores that share no row now stay apart. BBC News rose
from 94.4 to 98.8%. Rejected: a minimum x-height for lines with little
evidence (no net gain, more missed lines), and widening every row a fused
blob touches (40 more strays on held-out websites).

**A large word's own pieces are not rows (2026-11-26).** The detached tail
and dots of a large Arabic word seeded lines of their own, so the word's row
kept a short crop. BBC Arabic rose from 85.0 to 86.6%.

**Fused bold rows are found again (2026-11-26).** Thai sets words without
spaces, so the flat tops of a bold headline's glyphs read as background and
joined the row into one blob. Counting runs that do not persist down their
column as ink fixes it, but applied everywhere it cost held-out websites 0.2
to 0.5 points at every setting tried, since a photo's flat patches do not
persist either. So it runs only on lines with a member ten times as wide as
it is tall, and the rows stacked on them, on plain ground. BBC Thai rose
from 72.0 to 78.6%.

## Preparing lines

**The background is the busiest band of lumas (2026-11-22).** The median of
the crop's border took a button's fill for text when the crop ran past the
button. The commonest luma took the text for background on a gradient. The
busiest band of 17 levels, then its commonest luma, handles both.

**Arabic ي and ك settle on the page's word endings (2026-11-22).** They draw
like Persian ی and ک at the start and middle of a word, and the recognizer
often chose the Persian letter. Arabic interface text rose about 2 points.

**Neighbouring lines are blanked (2026-11-25).** The bottom of the line above
and the top of the one below, inside a tight crop, made the recognizer
misread the line. Held-out websites reached 88.4%.

**The background under each pixel (2026-11-26).** Arabic Wikipedia sets
headings in white on a grey bar that fades into the page, so the whole bar
read as ink. The crop is now opened or closed by a rectangle half its height
and twice as wide, which no stroke or icon can hold. Stray characters on the
live captures fell from 341 to 279. Rejected: a square element, which fits
inside icons and left outlines that read as letters.

**Fractional frames keep their ringing out of the background (2026-11-26).**
A scaled display is downsampled with a filter that leaves a halo beside every
stroke, and the opening moved a whole line's background by it. Arabic and
Hebrew at 1.25 to 1.75x fell from 85% to 67% before anyone noticed, since
every benchmark then read whole scales. Fractional frames now count a
background up to 32 levels past the crop's own, away from the ink, as the
crop's own. At whole scales that added 25 to 50 stray characters, so it
applies to fractional frames only.

**Not adopted: a second read by colour (2026-11-26).** Text that differs
from its ground in colour more than in brightness, such as blue on grey,
is invisible in luma. Reading each frame a second time with each pixel at its
brightest channel gained 0.1 to 0.4 points of recall for 35% more CPU, and the
gain came from reading lines twice, not from colour.

## Training

**Synthetic lines from open fonts (2026-11-21).** Every training line is
rendered from open fonts and open word lists, so anyone can rebuild the set
and the weights can ship under MIT. No system or app fonts, and the Mac's own
interface strings only ever score.

**Rendered pages (2026-11-25).** Single lines never taught the models
headlines, text over pictures or crowded layouts. Web-like pages, cut into
lines by yomi's own line finder and labelled from the page's text, became
about a sixth of each fine-tune. Western, CJK, the router, Southeast Asian,
Caucasian and Indic adopted them. Rejected: page fine-tunes of the
right-to-left model (Arabic interface text fell about a point each time), a
second CJK round (flat), and routers that broke the routing tests.

**Hindi interface weights (2026-11-26).** Hindi's gap to Vision was one-letter
misreads, such as ड read as ङ. Fine-tuning on Devanagari interface sans at
the weights interfaces use took Hindi from 90.6 / 90.0% to 90.2 / 91.0% and
fixed most of that swap. Rejected: oversampling words with the confused
letters, which made both scales worse, and drawing training words without
the usual bias toward frequent ones, so rare conjuncts such as the half ग of
सेटिंग्स appear more. That read सेटिंग्स but broke other words, and Hindi at
2x fell from 91.0 to 89.1%.

**CJK punctuation (2026-11-27).** Simplified Chinese trailed PaddleOCR by
three points, and most of it was curly quotes read as straight ones, which
training drew far less often. A fine-tune from the shipped model with extra
lines at twice the punctuation, nearly all of it CJK forms, took open CJK
from 98.2 / 98.5% to 98.4 / 98.6% and Korean on the open set from 96.8 / 97.3%
to 97.3 / 97.9%, with held-out website recall up 0.1 / 0.3 points and every
other set unchanged. It fixed only 15 of the 108 quote swaps, and Japanese ブ
read as プ did not move.

The quotes were a rendering difference, not a frequency one. macOS sets curly
quotes in Chinese text with Latin metrics, narrow and tight against the Han,
while Noto Sans SC, the main Simplified Chinese training font, draws them
full width. Training now sets the curly quotes of half its CJK lines in a
Latin font. A second round with such lines took interface CJK from 97.0 /
97.1% to 97.2 / 97.4% and Simplified Chinese from 96.1% to 97.0% at 2x, and
curly quotes read as straight ones fell from 93 to 27.

**Not adopted: Arabic interface fonts (2026-11-26).** A fine-tune on Arabic
interface sans left Arabic interface text flat and did not fix ح read as ج
in Safari's Arabic font.

**Fine-tuning from the shipped models (2026-11-26).** The checkpoints behind
the shipped models were never published. `train.py --init` now loads an
exported model directly, rebuilding the network from its weights with the
folded batch norms frozen, so anyone can fine-tune what ships.

## Evaluation

**Wrapped sentences count line by line (2026-11-22).** The answer keys
dropped any word that wrapped, which in CJK and Thai is a whole sentence, so
a quarter of the CJK text went unscored.

**Each character scores at the box yomi gives it (2026-11-26).** The scorers
split a word's box evenly across its characters, while Neru places them at
yomi's own character boxes. Most of what looked like misreads on Chinese
Wikipedia was that.

**CJK and Thai score each character where it sits (2026-11-26).** The
character scorer grouped ground-truth words into whole rows and charged each
read word to the row its centre fell in. Where a layout's rows ran into each
other, as a sidebar item that wraps beside a heading, a correct word counted
against the wrong row. Characters now go to the ground-truth word their box
sits in, as on the websites. Japanese on the open set went from 90.7% to
98.7%, and yomi went from 1.4 points behind Vision on open CJK to 3 ahead.

**An open evaluation set (2026-11-26).** Every set was built on a Mac from
Apple's strings and fonts, so nobody else could score a model. The open set
draws on Firefox's translations (MPL-2.0) in held-out open fonts, renders in
headless Chromium on any platform, and is published so everyone scores the
same pixels.

## Open problems

- Arabic trails Vision on interface text, by 2.8 points at 2x and more on the
  open set. Two kinds of fine-tune have not moved it.
- Hindi trails Vision by 1.4 to 1.9 points, mostly इ after a vowel sign read
  as another letter.
- Serif faces of the Indic and Southeast Asian scripts read far worse than
  sans, since training renders no serifs there.
- Text inside pictures and dimmed behind dialogs goes unread.
