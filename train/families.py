"""The model families yomi trains: each pairs a character set with the
languages its training lines draw from and the font folders it renders in."""
from charsets import ASCII, CJK, RTL, WESTERN
from synth import Generator
from textsrc import CJK_LANGS, RTL_LANGS, WESTERN_LANGS, cjk_line, line, rtl_line

# name: (charset, languages, line generator, {split: font folders})
# Families in SHAPED render through HarfBuzz, and their labels run in visual
# order.
FAMILIES = {
    "latin": (ASCII, {"en": 1}, line, {"train": ["train"], "heldout": ["heldout"]}),
    "western": (WESTERN, WESTERN_LANGS, line, {"train": ["train"], "heldout": ["heldout"]}),
    "cjk": (CJK, CJK_LANGS, cjk_line, {"train": ["cjk-train"], "heldout": ["cjk-heldout"]}),
    "rtl": (RTL, RTL_LANGS, rtl_line, {"train": ["rtl-train"], "heldout": ["rtl-heldout"]}),
}
SHAPED = {"rtl"}


def generator(family, split="train", blank_rate=0.08):
    charset, langs, line_fn, folders = FAMILIES[family]
    return Generator(charset, langs, line_fn, folders[split], blank_rate, split, family in SHAPED)
