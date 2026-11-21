"""The model families yomi trains: each pairs a character set with the
languages its training lines draw from."""
from charsets import ASCII, WESTERN
from synth import Generator
from textsrc import WESTERN_LANGS, line

FAMILIES = {
    "latin": (ASCII, {"en": 1}),
    "western": (WESTERN, WESTERN_LANGS),
}


def generator(family, split="train", blank_rate=0.08):
    charset, langs = FAMILIES[family]
    return Generator(charset, langs, line, split, blank_rate)
