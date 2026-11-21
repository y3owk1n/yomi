"""The model families yomi trains. Each pairs a character set with the
languages its training lines draw from and the font folders it renders in."""
from charsets import ASCII, CAUCASIAN_ETHIOPIC, CJK, INDIC, RTL, SEA, WESTERN
from synth import Generator
from textsrc import (CAUCASIAN_ETHIOPIC_LANGS, CJK_LANGS, INDIC_LANGS, RTL_LANGS, SEA_LANGS, WESTERN_LANGS, cjk_line,
                     line, rtl_line)

# name: (charset, languages, line generator, {split: font folders})
# Families in SHAPED render through HarfBuzz, and their labels run in visual
# order.
FAMILIES = {
    "latin": (ASCII, {"en": 1}, line, {"train": ["train"], "heldout": ["heldout"]}),
    "western": (WESTERN, WESTERN_LANGS, line, {"train": ["train"], "heldout": ["heldout"]}),
    "cjk": (CJK, CJK_LANGS, cjk_line, {"train": ["cjk-train"], "heldout": ["cjk-heldout"]}),
    "rtl": (RTL, RTL_LANGS, rtl_line, {"train": ["rtl-train"], "heldout": ["rtl-heldout"]}),
    "indic": (INDIC, INDIC_LANGS, line, {"train": ["indic-train"], "heldout": ["indic-heldout"]}),
    "sea": (SEA, SEA_LANGS, line, {"train": ["sea-train"], "heldout": ["sea-heldout"]}),
    "caucasian-ethiopic": (CAUCASIAN_ETHIOPIC, CAUCASIAN_ETHIOPIC_LANGS, line,
                           {"train": ["cau-train"], "heldout": ["cau-heldout"]}),
}
SHAPED = {"rtl", "indic", "sea"}


def generator(family, split="train", blank_rate=0.08):
    charset, langs, line_fn, folders = FAMILIES[family]
    return Generator(charset, langs, line_fn, folders[split], blank_rate, split, family in SHAPED)


# The router reads a line's script family, so the engine can pick the model
# without being told the language. Its classes are the families below, plus
# "none" for crops with no text.
ROUTED = ["western", "cjk", "rtl", "indic", "sea", "caucasian-ethiopic"]


class RouterGenerator:
    """Draws lines from every routed family for the router. Each line's label
    is the first routed family whose alphabet covers every character drawn, so
    a Latin-only line from any family counts as Western. Routing is scored the
    same way. None is the last index."""

    def __init__(self, split="train"):
        self.gens = [generator(f, split, blank_rate=0) for f in ROUTED]
        self.alphabets = [set(FAMILIES[f][0]) for f in ROUTED]
        self.split = split

    def sample(self, rng):
        if rng.random() < 0.08:
            from synth import blank
            from prep import prepare
            x = prepare(blank(rng, 1 if rng.random() < 0.5 else 2))
            return x, [len(ROUTED)], "none"
        i = rng.randrange(len(ROUTED))
        x, _, text = self.gens[i].sample(rng)
        drawn = {c for c in text if not c.isspace()}
        i = next((j for j, a in enumerate(self.alphabets) if drawn <= a), i)
        return x, [i], ROUTED[i]


def router_generator(split="train"):
    return RouterGenerator(split)
