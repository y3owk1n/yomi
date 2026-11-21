"""Text sources shared by the eval pages and the training renderer.

Dictionary words are split by a hash so evaluation words never appear as
training words. Training also mixes in random strings, so the model learns
letters rather than a word list.
"""
import random
import string
import zlib

CHARSET = (
    string.ascii_letters
    + string.digits
    + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
# Out-of-set glyphs are not in the label set. Training never shows them, so
# the model reads them as its nearest in-set shape. Evaluation counts them as
# misses.

_words = [w.strip() for w in open("/usr/share/dict/words") if 2 <= len(w.strip()) <= 12]


def _split(word):
    return zlib.crc32(word.lower().encode()) % 5 == 0


EVAL_WORDS = [w for w in _words if _split(w)]
TRAIN_WORDS = [w for w in _words if not _split(w)]

_UI = (
    "Settings Profile Account Billing Export Import Share Invite Archive Delete "
    "Rename Move Copy Upload Download Preview Publish Draft Review Merge Branch "
    "Commit Deploy Release Search Filter Sort Refresh Notes Inbox Sent Trash "
    "Calendar People Files Photos Music Videos Cancel OK Save Open Close Help "
    "Edit View Window File Tools Options Advanced General Privacy Security"
).split()


def _case(rng, w):
    r = rng.random()
    if r < 0.55:
        return w.lower()
    if r < 0.85:
        return w.capitalize()
    return w.upper()


def token(rng, words):
    """One whitespace-free token of UI-like text."""
    r = rng.random()
    if r < 0.55:
        w = _case(rng, rng.choice(words))
        if rng.random() < 0.12:
            w += rng.choice(".,:;!?)")
        if rng.random() < 0.04:
            w = rng.choice("(\"'[") + w
        return w
    if r < 0.62:
        return _case(rng, rng.choice(_UI))
    if r < 0.70:
        return str(rng.choice([rng.randint(0, 99), rng.randint(100, 99999), rng.randint(1, 31)]))
    if r < 0.74:
        return f"{rng.randint(1,12)}/{rng.randint(1,28)}/{rng.randint(2000,2030)}"
    if r < 0.77:
        return f"v{rng.randint(0,9)}.{rng.randint(0,30)}.{rng.randint(0,99)}"
    if r < 0.81:
        parts = [rng.choice(words).lower() for _ in range(rng.randint(1, 3))]
        return rng.choice(["~/", "/", "./"]) + "/".join(parts) + rng.choice(["", ".go", ".md", ".toml", ".txt"])
    if r < 0.85:
        a, b = rng.choice(words).lower(), rng.choice(words).capitalize()
        return rng.choice([f"{a}{b}", f"{a}_{b.lower()}", f"{a}-{b.lower()}", f"{a}.{b.lower()}()"])
    if r < 0.88:
        return f"{rng.randint(1,12)}:{rng.randint(0,59):02d}" + rng.choice(["", " AM", " PM"]).strip()
    if r < 0.91:
        return rng.choice(["$", "#", "@", "%", "+", "-"]) + str(rng.randint(1, 999))
    # random string: letters, digits and punctuation in no word order
    n = rng.randint(2, 10)
    return "".join(rng.choice(CHARSET) for _ in range(n))


def phrase(rng, words, n):
    return " ".join(token(rng, words) for _ in range(n))


class Docs:
    """Running text from Neru's docs/, as a second eval source the training
    generator never drew from. Markdown syntax is stripped, and tokens with
    characters outside CHARSET are dropped."""

    def __init__(self, root, source="docs"):
        import glob
        import re
        toks = []
        if source == "docs":
            paths = sorted(glob.glob(f"{root}/docs/**/*.md", recursive=True))
        else:  # "code": the changelog plus Go doc comments, a third unseen source
            paths = [f"{root}/CHANGELOG.md"] + sorted(glob.glob(f"{root}/internal/**/*.go", recursive=True))
        for path in paths:
            text = open(path, encoding="utf-8").read()
            if path.endswith(".go"):
                if path.endswith("_test.go"):
                    continue
                text = " ".join(l.strip()[3:] for l in text.splitlines() if l.strip().startswith("// "))
            text = re.sub(r"```.*?```", " ", text, flags=re.S)
            text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
            text = re.sub(r"[`*#>|]", " ", text)
            for t in text.split():
                t = t.strip("_")
                if t and all(c in CHARSET for c in t) and not set(t) <= set("-=:"):
                    toks.append(t)
        self.toks = toks

    def phrase(self, rng, _words, n):
        i = rng.randrange(0, len(self.toks) - n)
        return " ".join(self.toks[i:i + n])

    def token(self, rng, _words):
        return rng.choice(self.toks)
