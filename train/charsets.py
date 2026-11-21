"""The character sets each model family reads. A model's alphabet is its set
in this order, after the space; label i+1 is alphabet[i], 0 is the CTC blank.
Every character is NFC: a precomposed letter, never a base plus a combining
mark.
"""
import string
import unicodedata


def _range(a, b):
    return "".join(chr(c) for c in range(a, b + 1))


def _printable(chars):
    return "".join(c for c in chars if unicodedata.category(c)[0] in "LNPS")


ASCII = string.ascii_letters + string.digits + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

LATIN_1 = _printable(_range(0xA1, 0xFF))  # ¡ to ÿ: Western European letters and signs
LATIN_EXT_A = _range(0x100, 0x17F)  # Central European, Baltic, Turkish, Maltese
LATIN_EXT_B = "ƒǍǎǏǐǑǒǓǔȘșȚț" + "ƠơƯư"  # Romanian comma letters, Vietnamese horned letters
VIETNAMESE = _range(0x1EA0, 0x1EF9)  # tone-marked letters
CYRILLIC = _range(0x400, 0x45F) + "Ґґ"  # Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Macedonian
GREEK = "ΆΈΉΊΌΎΏΐ" + _range(0x391, 0x3A1) + _range(0x3A3, 0x3CE)  # monotonic
# Signs UI text uses that the blocks above lack: typographic quotes and
# dashes, the bullet and ellipsis, currency, arrows, disclosure triangles,
# check marks, and the Mac modifier keys.
UI_SIGNS = "‘’“”„‚–—…•·€£¥₽₺₹™©®°±×÷§¶†‰′″←↑→↓↔⇧⌘⌥⌃⌫⏎▲▼▶◀▸▾✓✔✕✗★☆⋯"

WESTERN = "".join(dict.fromkeys(ASCII + LATIN_1 + LATIN_EXT_A + LATIN_EXT_B + VIETNAMESE + CYRILLIC + GREEK + UI_SIGNS))

assert all(unicodedata.normalize("NFC", c) == c for c in WESTERN)
