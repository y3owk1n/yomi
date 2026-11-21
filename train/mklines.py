"""Cuts line crops from the eval pages, the input a line finder would hand the
recognizer. Words on one row join into a line until a gap wider than 1.2 em
(a table column or a button edge) splits it. Each crop is padded by a
seeded-random 10-35% of the line height on every side.

Usage: python -I mklines.py EVALDIR OUTDIR
Writes OUTDIR/{1x,2x}/NNNNN.png and OUTDIR/labels.jsonl:
  {"id":..., "scale":1|2, "page":..., "size":..., "mono":..., "text":..., "words":[...]}
"""
import json
import os
import random
import sys

from PIL import Image

d, out = sys.argv[1], sys.argv[2]
pages = open(os.path.join(d, "pages.txt")).read().split()
rng = random.Random(7)
os.makedirs(os.path.join(out, "1x"), exist_ok=True)
os.makedirs(os.path.join(out, "2x"), exist_ok=True)


def lines_of(words):
    words = sorted(words, key=lambda w: (round(w["y"]), w["x"]))
    rows = []
    for w in words:
        for row in rows:
            r = row[-1]
            same_row = abs((r["y"] + r["h"] / 2) - (w["y"] + w["h"] / 2)) < 0.3 * r["h"]
            gap = w["x"] - (r["x"] + r["w"])
            if same_row and 0 <= gap < 1.2 * r["size"] and r["size"] == w["size"] and r["mono"] == w["mono"]:
                row.append(w)
                break
        else:
            rows.append([w])
    return rows


labels = open(os.path.join(out, "labels.jsonl"), "w")
n = 0
for p in pages:
    words = json.load(open(os.path.join(d, p + ".json")))
    imgs = {s: Image.open(os.path.join(d, f"{p}@{s}x.png")).convert("RGB") for s in (1, 2)
            if os.path.exists(os.path.join(d, f"{p}@{s}x.png"))}
    for row in lines_of(words):
        x0 = min(w["x"] for w in row)
        y0 = min(w["y"] for w in row)
        x1 = max(w["x"] + w["w"] for w in row)
        y1 = max(w["y"] + w["h"] for w in row)
        h = y1 - y0
        pad = [rng.uniform(0.10, 0.35) * h for _ in range(4)]
        box = (x0 - pad[0], y0 - pad[1], x1 + pad[2], y1 + pad[3])
        text = " ".join(w["t"] for w in row)
        for s, im in imgs.items():
            b = tuple(round(v * s) for v in box)
            b = (max(0, b[0]), max(0, b[1]), min(im.width, b[2]), min(im.height, b[3]))
            im.crop(b).save(os.path.join(out, f"{s}x", f"{n:05d}.png"))
        labels.write(json.dumps({"id": f"{n:05d}", "page": p, "size": row[0]["size"], "mono": row[0]["mono"],
                                 "box": [round(v, 2) for v in box],
                                 "text": text, "words": [w["t"] for w in row]}) + "\n")
        n += 1
print(n, "lines")
