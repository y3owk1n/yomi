"""Scores whole-page reads against the pages' ground truth.

A run file has one JSON line per frame, as `yomi bench` and vision.swift
write them: {"page","scale","first_ms","median_ms","words":[[text,x,y,w,h,...]]}.
A ground-truth word counts as read when a read word with exactly its text has
its centre inside the word's box grown by 2 px. Extra words are read words
whose centre is in no ground-truth box. Page names start with their language
(evalgen.py), which the per-language breakdown uses.

Usage: python pagescore.py PAGESDIR RUN.jsonl [RUN.jsonl ...] [--by-lang]
"""
import collections
import json
import os
import statistics
import sys


def score(pages_dir, path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    res = {}
    for scale in (1, 2):
        rs = [r for r in rows if r["scale"] == scale]
        if not rs:
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
    by_lang = "--by-lang" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--by-lang"]
    for path in args[1:]:
        for scale, s in score(args[0], path).items():
            print(f"{os.path.basename(path):34} {scale}x  word {s['word_acc']:.1%}  extra {s['extra']:4}"
                  f"  first {s['first_ms']:6.1f} ms  median {s['median_ms']:6.1f} ms  ({s['words']} words)")
            if by_lang:
                print("    " + "  ".join(f"{k} {v:.1%}" for k, v in s["by_lang"].items()))
