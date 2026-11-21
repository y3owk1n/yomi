"""Training text. Words come from wordfreq (code Apache-2.0, word lists
CC-BY-SA 4.0 from its sources). A hash split holds out a fifth of every
language's words, so held-out words never appear in training. Lines mix those
words with the tokens UI text is full of (numbers, dates, versions, paths,
identifiers, acronyms, signs) and with random strings, so the model learns
letters rather than a word list.
"""
import random
import unicodedata
import zlib

import wordfreq

from charsets import ASCII, UI_SIGNS

# wordfreq languages per script family, with a weight for how often a line
# draws from each.
WESTERN_LANGS = {
    "en": 8, "de": 3, "fr": 3, "es": 3, "it": 2, "pt": 2, "nl": 1.5, "sv": 1, "da": 1,
    "nb": 1, "fi": 1, "is": 0.5, "pl": 2, "cs": 1.5, "sk": 1, "sl": 1, "hu": 1.5, "ro": 1.5,
    "tr": 2, "ca": 1, "lt": 1, "lv": 1, "id": 1, "ms": 1, "fil": 0.5, "vi": 2, "sh": 1,
    "ru": 4, "uk": 2, "bg": 1, "mk": 0.5, "el": 3,
}

_ACRONYMS = ("API CLI UI UX ID IDE IO AI GUI URL URI HTTP HTTPS HTML CSS JSON XML SQL SSH DNS VPN "
             "CPU GPU RAM SSD USB PDF PNG JPEG SVG GIF ZIP OK FAQ IT HR CEO VIP IBM MIT ISO UTC GMT "
             "iOS macOS iPadOS iPhone iPad iCloud AirDrop WiFi Wi-Fi LLC Inc Ltd PhD").split()
_UI = ("Settings Profile Account Billing Export Import Share Invite Archive Delete Rename Move Copy "
       "Upload Download Preview Publish Draft Review Merge Branch Commit Deploy Release Search Filter "
       "Sort Refresh Notes Inbox Sent Trash Calendar People Files Photos Music Videos Cancel OK Save "
       "Open Close Help Edit View Window File Tools Options Advanced General Privacy Security").split()


def _held_out(word):
    return zlib.crc32(word.encode()) % 5 == 0


def _fix(lang, w):
    # wordfreq folds Greek final sigma to σ.
    if lang == "el" and w.endswith("σ"):
        w = w[:-1] + "ς"
    return unicodedata.normalize("NFC", w)


class Words:
    """One language's words, train or held-out half, ordered by frequency."""

    _cache = {}

    def __init__(self, lang, split, charset, n=40000):
        key = (lang, split, n)
        if key not in Words._cache:
            ws = []
            for w in wordfreq.top_n_list(lang, n):
                w = _fix(lang, w)
                if w and all(c in charset for c in w) and (_held_out(w) == (split == "heldout")):
                    ws.append(w)
            Words._cache[key] = ws
        self.lang, self.words = lang, Words._cache[key]

    def pick(self, rng):
        # Frequent words more often: the index skews toward the head.
        return self.words[min(len(self.words) - 1, int(len(self.words) * rng.random() ** 2.5))]


def _case(rng, w, lang):
    r = rng.random()
    if r < 0.55:
        return w
    if r < 0.88:
        return w[:1].upper() + w[1:]
    up = w.upper()
    if lang == "el":  # Greek capitals drop the accent in all-caps text
        up = "".join(c for c in unicodedata.normalize("NFD", up) if unicodedata.category(c) != "Mn")
        up = unicodedata.normalize("NFC", up)
    return up


def ui_token(rng, words):
    """A whitespace-free token of the kind UI text is full of."""
    r = rng.random()
    if r < 0.14:
        return str(rng.choice([rng.randint(0, 99), rng.randint(100, 99999), rng.randint(1, 31)]))
    if r < 0.20:
        return rng.choice([f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(2000, 2030)}",
                           f"{rng.randint(1, 28)}.{rng.randint(1, 12):02d}.{rng.randint(2000, 2030)}",
                           f"{rng.randint(2000, 2030)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"])
    if r < 0.25:
        return f"v{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}"
    if r < 0.32:
        parts = [words.pick(rng).lower() for _ in range(rng.randint(1, 3))]
        return rng.choice(["~/", "/", "./", ""]) + "/".join(parts) + rng.choice(["", ".go", ".md", ".txt", ".png", ".pdf"])
    if r < 0.38:
        a, b = words.pick(rng), words.pick(rng)
        return rng.choice([f"{a}{b.capitalize()}", f"{a}_{b}", f"{a}-{b}", f"{a}.{b}()"])
    if r < 0.44:
        return f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}" + rng.choice(["", "", " AM", " PM"]).replace(" ", "")
    if r < 0.50:
        sign = rng.choice("$€£¥₽%#@+-−±×")
        num = rng.choice([str(rng.randint(1, 999)), f"{rng.randint(1, 99)},{rng.randint(0, 99):02d}",
                          f"{rng.randint(1, 9)}.{rng.randint(0, 999):03d}"])
        return sign + num if sign not in "%" else num + sign
    if r < 0.62:
        return rng.choice(_ACRONYMS)
    if r < 0.68:
        return rng.choice(_UI)
    if r < 0.76:
        return rng.choice(UI_SIGNS)
    # A random string over ASCII and the family's letters, in no word order.
    letters = "".join(words.words[rng.randrange(len(words.words))] for _ in range(3))
    pool = ASCII + letters
    return "".join(rng.choice(pool) for _ in range(rng.randint(2, 9)))


def word_token(rng, words):
    w = _case(rng, words.pick(rng), words.lang)
    if rng.random() < 0.15:
        w += rng.choice(".,:;!?)…")
    if rng.random() < 0.05:
        w = rng.choice("(\"'«„“‘[") + w + rng.choice(["", "\"", "»", "“", "”", "’", "]", ")"])
    return w


def line(rng, langs, charset, split="train", n=None):
    """One line of text: mostly words of one language, with UI tokens and the
    occasional English word mixed in."""
    names, weights = zip(*langs.items())
    lang = rng.choices(names, weights)[0]
    words = Words(lang, split, charset)
    en = Words("en", split, charset)
    n = n or rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 10])
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.70:
            out.append(word_token(rng, words))
        elif r < 0.78:
            out.append(word_token(rng, en))
        else:
            out.append(ui_token(rng, words))
    return " ".join(out)


# CJK lines. wordfreq's zh list is Simplified, and OpenCC converts it to
# Taiwan and Hong Kong Traditional with their own wording.
CJK_LANGS = {"zh-Hans": 4, "zh-Hant": 3, "ja": 4, "ko": 3}
_CJK_PUNCT = "，。、：；！？（）「」『』《》…・"
_opencc = {}


def _traditional(text, region):
    import opencc
    if region not in _opencc:
        _opencc[region] = opencc.OpenCC("s2twp" if region == "tw" else "s2hk")
    return _opencc[region].convert(text)


def cjk_line(rng, langs, charset, split="train", n=None):
    """One line of Chinese, Japanese or Korean UI-like text: words run
    together (Korean mostly spaced), CJK punctuation, and Latin tokens
    (numbers, acronyms, English words) mixed in, as CJK interfaces write them."""
    names, weights = zip(*langs.items())
    lang = rng.choices(names, weights)[0]
    src = {"zh-Hans": "zh", "zh-Hant": "zh"}.get(lang, lang)
    words = Words(src, split, set(charset) | set(_range_hans()), n=60000)
    en = Words("en", split, charset)
    n = n or rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 10, 14, 18])
    parts = []
    for i in range(n):
        r = rng.random()
        if r < 0.78:
            tok = words.pick(rng)
        elif r < 0.86:
            tok = rng.choice(_CJK_PUNCT)
        elif r < 0.93:
            tok = " " + ui_token(rng, en) + " "
        else:
            tok = " " + word_token(rng, en) + " "
        sep = " " if lang == "ko" and rng.random() < 0.6 and i else ""
        parts.append(sep + tok)
    text = "".join(parts)
    if lang == "zh-Hant":
        text = _traditional(text, "tw" if rng.random() < 0.7 else "hk")
    text = " ".join(text.split())
    return "".join(c for c in text if c in charset or c == " ")


def _range_hans():
    # Simplified forms convert to Traditional ones in the set, so the word list
    # keeps any Han character. The final filter drops what the set lacks.
    return [chr(c) for c in range(0x4E00, 0xA000)]
