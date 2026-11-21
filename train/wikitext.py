"""Fetches word lists for languages wordfreq lacks, from the intros of random
Wikipedia articles (text CC BY-SA 4.0), through the MediaWiki API. Training
only: the words feed the line generator and never ship. Each language's words
go to data/wiki/LANG.txt, most frequent first.

Usage: python wikitext.py LANG [LANG ...]   (ARTICLES per language, default 300)
"""
import collections
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "ocr-training-wordlist/0.1 (word frequency for synthetic text rendering)"


def fetch(lang, n):
    words = collections.Counter()
    got = 0
    while got < n:
        q = urllib.parse.urlencode({
            "action": "query", "generator": "random", "grnnamespace": 0, "grnlimit": 20,
            "prop": "extracts", "explaintext": 1, "exintro": 1, "format": "json",
        })
        req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    pages = json.load(r).get("query", {}).get("pages", {})
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                # Rate limited: wait as long as the server asks, or back off.
                time.sleep(int(e.headers.get("Retry-After") or 0) or 15 * 2 ** attempt)
        for p in pages.values():
            text = unicodedata.normalize("NFC", p.get("extract", ""))
            # Split on spaces, not \w: a word's vowel signs and other combining
            # marks are not \w. Thai, Lao, Khmer and Myanmar do not space their
            # words, so their tokens are phrases.
            for tok in text.split():
                w = tok.strip(".,;:!?()[]{}\"'«»“”‘’·।॥။၊።፣—–-")
                if w and len(w) <= 30 and all(unicodedata.category(c)[0] in "LM" for c in w):
                    words[w] += 1
            got += 1
        time.sleep(3)
    return words


if __name__ == "__main__":
    n = int(os.environ.get("ARTICLES", 300))
    os.makedirs(os.path.join(HERE, "data", "wiki"), exist_ok=True)
    for lang in sys.argv[1:]:
        words = fetch(lang, n)
        with open(os.path.join(HERE, "data", "wiki", lang + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(w for w, _ in words.most_common()))
        print(lang, len(words), "words", flush=True)
