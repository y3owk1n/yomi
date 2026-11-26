"""Scores reads of a rendered web set. Run it from the set's directory.

  python webscore.py REF.jsonl RUN.jsonl [RUN.jsonl ...] [--by-page]

recall:  of the characters REF (Vision accurate) reads, the share a run reads
         at the same place (characters whose centres fall in the REF word's
         box, in reading order, aligned by edit distance).
unseen:  REF words of 2+ characters with no run character inside.
dom:     character accuracy against the page's visible DOM words, the same
         alignment per DOM word.
strays:  run characters outside every DOM word and every REF word.
Pages and runs are `yomi bench` / vision.swift JSON lines.
"""
import collections
import json
import sys
import unicodedata

import numpy as np


def norm(t):
    return "".join(c for c in unicodedata.normalize("NFKC", t) if not c.isspace())


def chars(words, boxes=None):
    """Each character with its box: the engine's own when it gives one per
    character (yomi), else an even share of its word's box (Vision)."""
    out = []
    for i, w in enumerate(words or []):
        if boxes and len(boxes[i]) == len(w[0]):
            out += [(n, tuple(b)) for c, b in zip(w[0], boxes[i]) for n in norm(c)]
            continue
        t, (x, y, wd, h) = norm(w[0]), w[1:5]
        out += [(c, (x + wd * k / len(t), y, wd / len(t), h)) for k, c in enumerate(t)]
    return out


def rtl(t):
    """Whether t's first strong character is right-to-left."""
    for c in t:
        b = unicodedata.bidirectional(c)
        if b in ("L", "R", "AL"):
            return b != "L"
    return False


def inside(pt, b, pad=2):
    return b[0] - pad <= pt[0] <= b[0] + b[2] + pad and b[1] - pad <= pt[1] <= b[1] + b[3] + pad


def centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def match(truth_words, preds):
    """Matched characters, total, unseen words, and which preds were used."""
    got = tot = unseen = 0
    used = set()
    cent = [centre(b) for _, b in preds]
    for w in truth_words or []:
        t = norm(w[0])
        if not t:
            continue
        box = w[1:5]
        sign = -1 if rtl(t) else 1
        idx = sorted((i for i, c in enumerate(cent) if inside(c, box)), key=lambda i: sign * preds[i][1][0])
        used.update(idx)
        got += lcs(t, "".join(preds[i][0] for i in idx))
        tot += len(t)
        unseen += len(t) >= 2 and not idx
    return got, tot, unseen, used


def load(path):
    return {(r["page"], r["scale"]): r for r in map(json.loads, open(path))}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    by_page = "--by-page" in sys.argv
    ref, runs = load(args[0]), args[1:]
    for path in [args[0]] + runs:
        run = load(path)
        agg = collections.defaultdict(lambda: np.zeros(8))
        for (page, scale), r in sorted(run.items()):
            dom = [[d["t"], d["x"], d["y"], d["w"], d["h"]] for d in json.load(open(f"{page}.json"))]
            preds = chars(r["words"], r.get("chars"))
            rw = [w for w in ref[(page, scale)]["words"]]
            g, t, u, used_r = match(rw, preds)
            dg, dt, _, used_d = match(dom, preds)
            strays = len(preds) - len(used_r | used_d) if path != args[0] else \
                len(preds) - len(used_d)
            v = np.array([g, t, u, dg, dt, strays, r["median_ms"], 1])
            agg[scale] += v
            if by_page:
                print(f"  {page:20} {scale}x recall {g / max(t, 1):6.1%} unseen {u:3}  dom {dg / max(dt, 1):6.1%}  strays {strays}")
        for scale, a in sorted(agg.items()):
            print(f"{path.split('/')[-1]:24} {scale}x  recall {a[0] / a[1]:6.1%}  unseen {int(a[2]):4}  "
                  f"dom {a[3] / a[4]:6.1%}  strays {int(a[5]):5}  mean {a[6] / a[7]:6.1f} ms")


if __name__ == "__main__":
    main()
