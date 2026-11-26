# How it works

A read goes through five stages. Each section below says what its stage does
and the reasons behind the choices that are not obvious.
[decisions.md](decisions.md) has the measurements behind each one.

```
frame (RGBA) and scale
  -> find lines     luma at one pixel per point, ink by flat runs, blobs, lines,
                    a crop per line from its baseline and x-height
  -> prepare        an ink map 32 rows tall, measured against the background
                    under each pixel
  -> route          each line's script family, then the families the page is in
  -> recognize      that family's network, CTC over its alphabet
  -> words          words split at spaces and wide gaps, a box per character,
                    logical order for right-to-left text
```

## Find lines

`internal/linefind` works in grey at one pixel per logical pixel. A whole
scale box-filters the frame down. A fractional scale such as 1.25
resamples it by area.

**Ink.** Each row splits into runs: six or more pixels within six luma levels
of each other. A run is background when it is at the level of the row's
longest run or too long to be the inside of a stroke. A pixel that is not
background and differs by more than 40 levels from the background on each
side is ink. A short run that differs from the background runs on both sides
of it, which agree with each other, is the inside of a wide stroke, as in
large bold text, and counts as ink too. Straight runs of ink at least 24
pixels long across, or 16 down, are rules, borders and outlines, and get
cleared, except a vertical run that sits in ink three or more pixels wide,
as a bold stem does. A one-pixel cut through a glyph, where a crossbar sits
at the text's own level, gets bridged.

**Blobs and lines.** Ink joins into 8-connected blobs. Blobs taller than 48
pixels are pictures or panels. Blobs from 5 to 48 pixels tall seed lines,
and smaller marks (dots, commas, quotes) only join them. A blob joins the
line whose core band it overlaps most, within a gap of 1.6 times the line's
height. The core is the mean top and bottom of the line's big members, so a
row of descenders cannot drag it down.

Blobs over 2.2 times the page's median glyph height wait for a second pass.
At small sizes the marks of tightly set Thai or Myanmar touch across rows,
and grouped with the rest such a blob would merge two rows. In the second
pass a tall blob joins the line it overlaps most. It merges lines on one row
that it spans, as the halves of a large heading split at a wide gap. Where
it overlaps two rows, it widens one row without growing it. Lines that lie
inside the blob's own bounds, such as the dots and detached tail of a large
Arabic word, count as its pieces, not as rows.

Then lines that lie mostly over one another merge, as the parts of a Hangul
syllable or the upper strokes of a heading do. Two lines of three or more
glyphs whose cores are of similar height and share no row stay apart, since
those are two rows of tight type.

**Faint text.** A second pass at an edge of 15 levels runs alongside the
first, for grey text on white or white text on a light bar. It keeps a line
only where the first pass found nothing and the ground around the line is
plain: four fifths of the ring around it, leaving out every line's ink,
within 8 levels of the ring's commonest luma. A photograph's texture fails
that test.

**Fused rows.** In bold text set without spaces, as in Thai headlines, the
flat tops of the glyphs can be taken for background, which turns the gaps
between letters into ink and joins a whole row into one blob. A line with a
member at least 16 pixels tall and ten times as wide is found again from the
region around it. In that pass a short run that does not persist 12 rows
down its column counts as ink, since background persists and a stroke does
not. Rows stacked on such a line, of the same size, are found again with it.
The pass runs only where the ground is plain, since a photograph's flat
patches do not persist either.

**The crop.** The baseline is the lower median of the members' bottoms.
Fragments under half the median height that stand beside the glyphs, not
over or under one as a dot or a vowel sign does, do not vote. The x-height
is the lower quartile of the heights on the baseline when the line has both
short letters and tall ones. A line of one height takes the x-height or cap
height of the other lines on the page it comes nearest. The crop runs 1.95
x-heights above the baseline, 0.85 below and 0.5 to each side, which matches
how training pads its lines.

## Prepare

`internal/prep` turns a crop into the recognizer's input. `train/prep.py`
does the same in training, and the two must stay in step.

- **Background.** The busiest band of 17 luma levels, then its commonest
  luma. A gradient behind a title spreads over many levels, while solid text
  sits on one, so the band picks the gradient. The polarity, light ink or
  dark, comes from which extreme lies further from the background.
- **Neighbours.** The marks of the lines above and below that reach into the
  crop are set to the background, so the recognizer sees one line as it
  trained on.
- **Background under each pixel.** The crop opened, for light ink, or closed,
  for dark, by a rectangle half the crop tall and twice as wide. No stroke is
  thick enough and no icon wide enough to hold the rectangle, so both stay
  ink, while a bar or a gradient under the text becomes background. In a
  frame read at a fractional scale, which was downsampled and so has a faint
  halo beside every stroke, a background up to 32 levels past the crop's own
  on the side away from the ink counts as the crop's own.
- **Ink map.** Each pixel's distance from its background toward the ink
  extreme, from 0 to 1. Underlines below the baseline are erased, since
  browsers draw them under the spaces between words. The map is box-filtered
  and then resampled to 32 rows.

## Route

With a router, every line of a frame is routed before any is read. The
router is a recognizer network with one class per script family plus "no
text". Each step of a line votes, and the line goes to the family with the
best mean.

A family counts as on the page once the router picks it for a line at least
two em wide, ahead of its runner-up by a clear margin. Lines in a script the
router knows lead by a wide margin, and icons and short tokens by a small
one. A line routed to a family not on the page reads with the base
recognizer instead. That rule keeps an English page's icons out of the
Chinese and Arabic recognizers, which would read them as characters.

When the runner-up comes close, the line is read by both recognizers and the
surer read kept. Wide lines also read with the page's main family when the
margin is small, which catches display type the router rarely saw.

## Recognize

Each recognizer is a small convolutional network with CTC decoding, from 0.2
to 1.8 million parameters:

- a 3x3 convolution, then depthwise-separable blocks that bring the 32 rows
  down to 4
- a collapse of those rows to one feature per column
- context along the line: a depthwise convolution, a layer that adds the
  line's mean to every step (which gives a lone letter the line's x-height
  to compare against), and a dilated convolution that reaches 13 steps
- a head with one output per character, plus blank

Each output step covers four input columns, or two for scripts with narrow
letters or stacked marks, where four would give some characters less than
two steps and CTC would drop them. The CJK recognizer has 12,098 characters,
so its head is gated: a one-output layer gives each step's blank score, and a
low-rank pair of layers gives character scores only at the steps where a
character can win.

Weights ship as half floats, widened once at load, which halves the files
and changes no read. On arm64 the convolutions, ReLU and the CJK head run in
NEON assembly. On amd64 they run in AVX2 with FMA when the CPU has it,
chosen at run time, and in plain Go otherwise.

## Words

Decoding is greedy CTC. A word ends at a space or at a gap wider than 0.75
of the line's height. Each character's box reaches halfway to the
characters drawn on either side, so the boxes tile the word. A word's
confidence is its least sure character's probability, and a lone character
under 0.5 is dropped, since that is nearly always an icon read as a letter.

Some fixes happen after decoding:

- Right-to-left words come back in logical order.
- Latin, Cyrillic and Greek letters that draw alike take the script of the
  rest of their word, or of their line.
- Arabic ي and ك draw like Persian ی and ک at the start and middle of a
  word. Read settles those positions on the page's word endings, where the
  two languages' letters differ.
- A lone 〇 read by a recognizer other than the base is dropped, since the
  CJK recognizer reads round icons as it.
