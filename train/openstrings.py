"""Collects interface strings per language from Firefox's translations
(github.com/mozilla-l10n/firefox-l10n, MPL-2.0), for the open evaluation set.
Anyone can fetch them, so pages built from them can be published, unlike the
Mac's own strings uistrings.py collects.

Usage: python openstrings.py L10NDIR OUT.json
L10NDIR is a checkout of firefox-l10n, at least each locale's browser/ and
toolkit/. Writes {lang: [strings]} with yomi's language keys.
"""
import json
import os
import re
import sys

# Firefox locale to yomi's key, as uistrings.json and the eval pages name it.
LOCALES = {
    "cs": "cs", "de": "de", "el": "el", "en-GB": "en", "es-ES": "es", "fr": "fr", "hu": "hu", "it": "it",
    "nl": "nl", "pl": "pl", "pt-BR": "pt_BR", "ro": "ro", "ru": "ru", "sv-SE": "sv", "tr": "tr", "uk": "uk",
    "vi": "vi", "zh-CN": "zh_CN", "zh-TW": "zh_TW", "ja": "ja", "ko": "ko", "ar": "ar", "he": "he", "fa": "fa",
    "ur": "ur", "hi-IN": "hi", "bn": "bn", "gu-IN": "gu", "kn": "kn", "ml": "ml", "mr": "mr", "ne-NP": "ne",
    "pa-IN": "pa", "si": "si", "ta": "ta", "te": "te", "th": "th", "lo": "lo", "km": "km", "my": "my",
    "ka": "ka", "hy-AM": "hy",
}
# Attributes that hold text a user reads. The rest are access keys,
# shortcuts and CSS.
SHOWN = {"label", "title", "tooltiptext", "placeholder", "aria-label", "description", "heading", "value", "message"}
BRAND = re.compile(r"\{\s*-(brand-short-name|brand-shorter-name|brand-full-name|brand-product-name|vendor-short-name)\s*\}")
MESSAGE = re.compile(r"^([a-zA-Z][\w-]*)\s*=\s*(.*)$")
ATTRIBUTE = re.compile(r"^\s+\.([\w-]+)\s*=\s*(.*)$")
TAG = re.compile(r"<[^>]+>")
# Languages that write in Latin letters. Elsewhere a string must be mostly in
# its own script, since a locale shows the English string where it has no
# translation.
LATIN = {"cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt_BR", "ro", "sv", "tr", "vi"}


def native(s, key):
    letters = [c for c in s if c.isalpha()]
    if key in LATIN or not letters:
        return bool(letters)
    return sum(c > "\u02ff" for c in letters) * 2 > len(letters)


def strings_of(path):
    """The shown text of every message and attribute in the Fluent file at
    path, without the ones that need a variable filled in."""
    out = []
    key, parts = None, []

    def flush():
        if key and key not in ("term", "hidden"):
            text = " ".join(p.strip() for p in parts if p.strip())
            text = BRAND.sub("Firefox", text)
            if text and "{" not in text and "}" not in text:
                out.append(" ".join(TAG.sub("", text).split()))

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                flush()
                key, parts = None, []
                continue
            m = MESSAGE.match(line)
            if m:
                flush()
                key, parts = ("term" if m.group(1).startswith("-") else "msg"), [m.group(2)]
                continue
            m = ATTRIBUTE.match(line)
            if m:
                flush()
                key, parts = ("attr" if m.group(1) in SHOWN else "hidden"), [m.group(2)]
                continue
            if line[:1].isspace():
                parts.append(line)
    flush()
    return out


def main():
    root, out = sys.argv[1], sys.argv[2]
    res = {}
    for loc, key in LOCALES.items():
        seen, keep = set(), []
        for sub in ("browser", "toolkit"):
            for d, _, files in os.walk(os.path.join(root, loc, sub)):
                for name in sorted(files):
                    if not name.endswith(".ftl"):
                        continue
                    for s in strings_of(os.path.join(d, name)):
                        words = s.split()
                        if s not in seen and 1 <= len(words) <= 14 and len(s) <= 90 and native(s, key):
                            seen.add(s)
                            keep.append(s)
        res[key] = sorted(keep)
        print(f"{key:6} {len(keep)}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
