"""Scores whole-page reads against the pages' ground truth.

A run file has one JSON line per frame, as `yomi bench` and vision.swift
write them: {"page","scale","first_ms","median_ms","words":[[text,x,y,w,h,...]]}.
A ground-truth word counts as read when a read word with exactly its text has
its centre inside the word's box grown by 2 px. Extra words are read words
whose centre is in no ground-truth box. Page names start with their language
(evalgen.py), which the per-language breakdown uses.

With --chars, accuracy is per character instead, for scripts written
without spaces. Each read character goes to the smallest ground-truth word
whose box, grown by 2 px, holds its centre: at the box yomi gives it, or an
even share of its word's box for an engine that gives none. Each word's
characters join in reading order, and the score is one minus the edit
distance over the ground-truth characters, both sides NFKC-normalized with
spaces dropped, as webscore.py does. Scoring whole words against whole rows
charged a word to a row it did not sit in wherever a layout's rows ran into
each other, as a wrapped sidebar item beside a heading does.

Usage: python pagescore.py PAGESDIR RUN.jsonl [RUN.jsonl ...] [--by-lang] [--chars]
"""
import collections
import json
import os
import statistics
import sys

from webscore import chars as char_boxes, norm


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_chars(pages_dir, rows):
    tot = collections.Counter()
    lang = collections.defaultdict(collections.Counter)
    for r in rows:
        gt = [g for g in json.load(open(os.path.join(pages_dir, r["page"] + ".json"))) if norm(g["t"])]
        boxes = [(g["x"] - 2, g["y"] - 2, g["x"] + g["w"] + 2, g["y"] + g["h"] + 2) for g in gt]
        owned = collections.defaultdict(list)
        used = set()
        for i, w in enumerate(r["words"]):
            for ch, (x, y, cw, chh) in char_boxes([w], [r["chars"][i]] if r.get("chars") else None):
                cx, cy = x + cw / 2, y + chh / 2
                hits = [k for k, (x0, y0, x1, y1) in enumerate(boxes) if x0 <= cx <= x1 and y0 <= cy <= y1]
                if hits:
                    k = min(hits, key=lambda k: (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]))
                    owned[k].append((cx, ch))
                    used.add(i)
        for k, g in enumerate(gt):
            want = norm(g["t"])
            read = "".join(ch for _, ch in sorted(owned[k], key=lambda t: t[0]))
            for c in (tot, lang[r["page"].split("-")[0]]):
                c["n"] += len(want)
                c["edits"] += min(len(want), edit(want, read))
        tot["extra"] += len(r["words"]) - len(used)
    return tot, lang


def score(pages_dir, path, chars=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    res = {}
    for scale in (1, 2):
        rs = [r for r in rows if r["scale"] == scale]
        if not rs:
            continue
        if chars:
            tot, lang = score_chars(pages_dir, rs)
            res[scale] = {
                "word_acc": 1 - tot["edits"] / max(1, tot["n"]), "words": tot["n"], "extra": tot["extra"],
                "first_ms": statistics.median(r["first_ms"] for r in rs),
                "median_ms": statistics.median(r["median_ms"] for r in rs),
                "by_lang": {k: 1 - v["edits"] / max(1, v["n"]) for k, v in sorted(lang.items())},
            }
            continue
        tot = collections.Counter()
        lang = collections.defaultdict(collections.Counter)
        for r in rs:
            gt = json.load(open(os.path.join(pages_dir, r["page"] + ".json")))
            used = set()
            for g in gt:
                gx0, gy0, gx1, gy1 = g["x"] - 2, g["y"] - 2, g["x"] + g["w"] + 2, g["y"] + g["h"] + 2
                hit = False
                for i, w in enumerate(r["words"]):
                    t, x, y, ww, hh = w[:5]
                    if gx0 <= x + ww / 2 <= gx1 and gy0 <= y + hh / 2 <= gy1:
                        used.add(i)
                        if t == g["t"]:
                            hit = True
                            break
                for c in (tot, lang[r["page"].split("-")[0]]):
                    c["n"] += 1
                    c["hit"] += hit
            tot["extra"] += len(r["words"]) - len(used)
        res[scale] = {
            "word_acc": tot["hit"] / max(1, tot["n"]), "words": tot["n"], "extra": tot["extra"],
            "first_ms": statistics.median(r["first_ms"] for r in rs),
            "median_ms": statistics.median(r["median_ms"] for r in rs),
            "by_lang": {k: v["hit"] / max(1, v["n"]) for k, v in sorted(lang.items())},
        }
    return res


if __name__ == "__main__":
    by_lang, chars = "--by-lang" in sys.argv, "--chars" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unit = "char" if chars else "word"
    for path in args[1:]:
        for scale, s in score(args[0], path, chars).items():
            print(f"{os.path.basename(path):34} {scale}x  {unit} {s['word_acc']:.1%}  extra {s['extra']:4}"
                  f"  first {s['first_ms']:6.1f} ms  median {s['median_ms']:6.1f} ms  ({s['words']} {unit}s)")
            if by_lang:
                print("    " + "  ".join(f"{k} {v:.1%}" for k, v in s["by_lang"].items()))
