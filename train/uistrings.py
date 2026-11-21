"""Collects the interface strings of this Mac's system apps, per language, from
their .loctable files: real UI text in about forty languages. It is an
evaluation source only. Training never reads it, and the output stays out of
the repository (it is Apple's text).

Usage: python uistrings.py OUT.json
"""
import glob
import json
import plistlib
import re
import sys
import unicodedata

# printf and Apple format placeholders, which no screen shows as such.
PLACEHOLDER = re.compile(r"%(\d+\$)?[-+ #0]*\d*(\.\d+)?(ll|l|h)?[@dDuUxXoOfeEgGcCsSpaA%]|\^\[[^\]]*\]|\{[^}]*\}")


def clean(s):
    s = unicodedata.normalize("NFC", s).replace(" ", " ").replace(" ", " ")
    s = PLACEHOLDER.sub("", s)
    return " ".join(s.split())


def main(out):
    by_lang = {}
    for path in sorted(glob.glob("/System/Applications/**/*.loctable", recursive=True)):
        try:
            with open(path, "rb") as f:
                table = plistlib.load(f)
        except Exception:  # some tables are not plists
            continue
        for lang, strings in table.items():
            if lang == "LocProvenance" or not isinstance(strings, dict):
                continue
            for v in strings.values():
                if isinstance(v, str):
                    s = clean(v)
                    if 1 < len(s) <= 120:
                        by_lang.setdefault(lang, set()).add(s)
    data = {lang: sorted(v) for lang, v in sorted(by_lang.items())}
    json.dump(data, open(out, "w"), ensure_ascii=False)
    print(len(data), "languages:", {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main(sys.argv[1])
