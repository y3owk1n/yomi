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


def _table(codec, rows, cols):
    out = []
    for a in rows:
        for b in cols:
            try:
                c = bytes([a, b]).decode(codec)
            except UnicodeDecodeError:
                continue
            if len(c) == 1:
                out.append(c)
    return out


def _han(chars):
    return [c for c in chars if unicodedata.name(c, "").startswith("CJK UNIFIED")]


# Hanzi and kanji: GB2312 levels 1 and 2 (Simplified Chinese), Big5 level 1
# (the frequent Traditional set) and JIS X 0208 level 1 (common kanji).
HAN = "".join(sorted(set(_han(_table("gb2312", range(0xB0, 0xF8), range(0xA1, 0xFF))))
                     | set(_han(_table("big5", range(0xA4, 0xC7), list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF)))))
                     | set(_han(_table("euc_jp", range(0xB0, 0xD0), range(0xA1, 0xFF))))))
HANGUL = "".join(_table("euc_kr", range(0xB0, 0xC9), range(0xA1, 0xFF)))  # KS X 1001's 2,350 syllables
KANA = _range(0x3041, 0x3096) + "ゝゞ" + _range(0x30A1, 0x30FA) + "ー・ヽヾ"
CJK_PUNCT = "、。〃〈〉《》「」『』【】〔〕〖〗〜〝〞・" + "".join(chr(0xFF01 + i) for i in range(94))  # fullwidth ASCII

# What the Mac's own Chinese, Japanese and Korean UI strings use beyond the
# sets above: the kanji iteration and closing marks, Korean compatibility
# jamo, a few Cantonese characters, and signs.
CJK_EXTRA = "々〆〇―" + _range(0x3131, 0x3163) + "饋枱牀吿嘅嗰喺焗" + "●∞πμ"

CJK = "".join(dict.fromkeys(ASCII + CJK_PUNCT + KANA + HANGUL + HAN + CJK_EXTRA + UI_SIGNS))

assert all(unicodedata.normalize("NFC", c) == c for c in CJK)
