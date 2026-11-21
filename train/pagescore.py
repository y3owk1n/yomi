"""Scores whole-page reads against the pages' ground truth.

A run file has one JSON line per frame, as `yomi bench` and vision.swift
write them: {"page","scale","first_ms","median_ms","words":[[text,x,y,w,h,...]]}.
A ground-truth word counts as read when a read word with exactly its text has
its centre inside the word's box grown by 2 px. Extra words are read words
whose centre is in no ground-truth box. Page names start with their language
(evalgen.py), which the per-language breakdown uses.

With --chars, accuracy is per character instead, for scripts written
without spaces: ground-truth words group into lines, the read words whose
centres fall in a line join in reading order, and the score is one minus the
edit distance over the ground-truth characters, spaces ignored.

Usage: python pagescore.py PAGESDIR RUN.jsonl [RUN.jsonl ...] [--by-lang] [--chars]
"""
import collections
import json
import os
import statistics
import sys


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def gt_lines(gt):
    """Ground-truth words grouped into lines: same row, joined in x order."""
    rows = []
    for g in sorted(gt, key=lambda g: (g["y"] + g["h"] / 2, g["x"])):
        cy = g["y"] + g["h"] / 2
        for r in rows:
            if abs(r["cy"] - cy) < 0.4 * g["h"] and g["x"] - r["x1"] < 2 * g["h"]:
                r["words"].append(g)
                r["x1"] = max(r["x1"], g["x"] + g["w"])
                break
        else:
            rows.append({"cy": cy, "x1": g["x"] + g["w"], "words": [g]})
    out = []
    for r in rows:
        ws = sorted(r["words"], key=lambda g: g["x"])
        x0, y0 = min(g["x"] for g in ws), min(g["y"] for g in ws)
        x1, y1 = max(g["x"] + g["w"] for g in ws), max(g["y"] + g["h"] for g in ws)
        out.append(((x0 - 2, y0 - 2, x1 + 2, y1 + 2), "".join(g["t"] for g in ws)))
    return out


def score_chars(pages_dir, rows):
    tot = collections.Counter()
    lang = collections.defaultdict(collections.Counter)
    for r in rows:
        gt = json.load(open(os.path.join(pages_dir, r["page"] + ".json")))
        used = set()
        for (x0, y0, x1, y1), text in gt_lines(gt):
            got = [(w[1], i, w[0]) for i, w in enumerate(r["words"])
                   if x0 <= w[1] + w[3] / 2 <= x1 and y0 <= w[2] + w[4] / 2 <= y1]
            used.update(i for _, i, _ in got)
            read = "".join(t for _, _, t in sorted(got)).replace(" ", "")
            want = text.replace(" ", "")
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
