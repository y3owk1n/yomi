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


# Arabic as Arabic, Persian and Urdu write it: the base letters, the
# Persian and Urdu letters, Arabic-Indic and Persian digits, punctuation, the
# common harakat, and tatweel. Hebrew letters with their final forms and
# geresh. UI text in these languages mixes in Latin words and digits.
ARABIC = (_range(0x621, 0x63A) + _range(0x640, 0x652) + "ٰ" + _range(0x660, 0x66D)
          + "ٹپچڈڑژکگںھہۂۃیےۓۀ" + _range(0x6F0, 0x6F9) + "،؛؟۔")
HEBREW = _range(0x5D0, 0x5EA) + "׳״־"
RTL = "".join(dict.fromkeys(ASCII + ARABIC + HEBREW + UI_SIGNS + "«»"))


def _block(*ranges):
    """The assigned, NFC-stable letters, marks and digits of the ranges.
    Precomposed letters NFC decomposes (Devanagari's nukta letters) stay out:
    a model reads the base letter and the nukta."""
    out = []
    for a, b in ranges:
        for c in range(a, b + 1):
            ch = chr(c)
            if unicodedata.name(ch, "") and unicodedata.category(ch)[0] in "LMNP" and unicodedata.normalize("NFC", ch) == ch:
                out.append(ch)
    return "".join(out)


# Devanagari (Hindi, Marathi, Nepali), Bengali, Gurmukhi, Gujarati, Tamil,
# Telugu, Kannada, Malayalam and Sinhala.
INDIC = "".join(dict.fromkeys(ASCII + UI_SIGNS + _block(
    (0x900, 0x97F), (0x980, 0x9FF), (0xA00, 0xA7F), (0xA80, 0xAFF), (0xB80, 0xBFF),
    (0xC00, 0xC7F), (0xC80, 0xCFF), (0xD00, 0xD7F), (0xD80, 0xDFF))))
# Thai, Lao, Khmer and Myanmar.
SEA = "".join(dict.fromkeys(ASCII + UI_SIGNS + _block((0xE01, 0xE5B), (0xE80, 0xEDF), (0x1780, 0x17F9), (0x1000, 0x109F))))
# Georgian (with Mtavruli capitals), Armenian and Ethiopic.
CAUCASIAN_ETHIOPIC = "".join(dict.fromkeys(ASCII + UI_SIGNS + _block(
    (0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x531, 0x58A), (0x1200, 0x137C))))
