"""End-to-end word accuracy on whole pages: line boxes -> crops -> model ->
words placed by their CTC time steps -> matched to the ground-truth words.

A ground-truth word is read when a predicted word with exactly its text has
its centre inside the word's box (grown by 2 px). Predicted words whose centre
is in no ground-truth box are counted as extra.

Usage:
  python -I e2e.py EVALDIR CHECKPOINT gt LINESDIR        # ground-truth lines
  python -I e2e.py EVALDIR CHECKPOINT found FOUND.jsonl  # linefind output
"""
import collections
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Recognizer  # noqa: E402
from prep import prepare  # noqa: E402
from synth import ALPHABET  # noqa: E402

d, ck, mode, src = sys.argv[1:5]
state = torch.load(ck, map_location="cpu")
model = Recognizer(state["classes"], state["width"], state.get("arch", "base")).eval()
model.load_state_dict(state["model"])
torch.set_num_threads(4)
pages = open(os.path.join(d, "pages.txt")).read().split()

boxes = collections.defaultdict(list)  # (page, scale) -> [x0, y0, x1, y1] in CSS px
if mode == "gt":
    labels = [json.loads(l) for l in open(os.path.join(src, "labels.jsonl"))]
    for lab in labels:
        for s in (1, 2):
            boxes[(lab["page"], s)].append(lab["box"])
else:
    for line in open(src):
        r = json.loads(line)
        boxes[(r["page"], r["scale"])] += [[x, y, x + w, y + h] for x, y, w, h in r["lines"]]


def read_line(img, box, s):
    """Words [(text, x0, y0, x1, y1)] in CSS px from one line box."""
    b = [round(v * s) for v in box]
    b = [max(0, b[0]), max(0, b[1]), min(img.width, b[2]), min(img.height, b[3])]
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return []
    x = prepare(np.asarray(img.crop(b)))
    with torch.no_grad():
        k = model(torch.from_numpy(x)[None, None])[0].argmax(-1).numpy()
    sx = (b[2] - b[0]) / x.shape[1]
    words, cur, prev = [], None, 0
    for t, c in enumerate(k):
        if c != prev and c != 0:
            ch = ALPHABET[c - 1]
            if ch == " ":
                if cur:
                    words.append(cur)
                cur = None
            elif cur is None:
                cur = [ch, t, t]
            else:
                cur[0] += ch
                cur[2] = t
        prev = c
    if cur:
        words.append(cur)
    out = []
    for text, t0, t1 in words:
        x0 = (b[0] + t0 * 4 * sx) / s
        x1 = (b[0] + (t1 + 1) * 4 * sx) / s
        out.append((text, x0, b[1] / s, x1, b[3] / s))
    return out


res = {}
for s in (1, 2):
    c = collections.Counter()
    for p in pages:
        path = os.path.join(d, f"{p}@{s}x.png")
        if not os.path.exists(path) or (p, s) not in boxes:
            continue
        img = Image.open(path).convert("RGB")
        pred = [w for box in boxes[(p, s)] for w in read_line(img, box, s)]
        gt = json.load(open(os.path.join(d, p + ".json")))
        used = set()
        dbg = os.environ.get("DEBUG_PAGE") == p and s == 1
        if os.environ.get("ASCII_ONLY"):
            from textsrc import CHARSET
            gt = [g for g in gt if all(ch in CHARSET for ch in g["t"])]
        for g in gt:
            gx0, gy0, gx1, gy1 = g["x"] - 2, g["y"] - 2, g["x"] + g["w"] + 2, g["y"] + g["h"] + 2
            c["words"] += 1
            c["dup"] += sum(1 for (t, x0, y0, x1, y1) in pred
                            if gx0 <= (x0 + x1) / 2 <= gx1 and gy0 <= (y0 + y1) / 2 <= gy1) >= 2
            for i, (t, x0, y0, x1, y1) in enumerate(pred):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if gx0 <= cx <= gx1 and gy0 <= cy <= gy1:
                    used.add(i)
                    if t == g["t"]:
                        c["hit"] += 1
                        break
            else:
                cands = [pred[i] for i, (t, x0, y0, x1, y1) in enumerate(pred)
                         if gx0 <= (x0 + x1) / 2 <= gx1 and gy0 <= (y0 + y1) / 2 <= gy1]
                inside = [w for w in pred if w[1] < gx1 and w[3] > gx0 and gy0 <= (w[2] + w[4]) / 2 <= gy1]
                if not inside:
                    kind = "nothing"
                elif len(inside) > 1 and "".join(w[0] for w in inside) == g["t"]:
                    kind = "split"
                elif len(inside) > 1 and g["t"] in "".join(w[0] for w in inside):
                    kind = "split+"
                elif any(g["t"] in w[0] and w[0] != g["t"] for w in inside):
                    kind = "merged"
                else:
                    kind = "misread"
                c["miss_" + kind] += 1
                if os.environ.get("SHOW") == kind and s == 1 and c["shown"] < int(os.environ.get("SHOWN", "40")):
                    c["shown"] += 1
                    print("  ", kind, p, repr(g["t"]), "->", [w[0] for w in inside][:3])
                if dbg:
                    near = [w for w in pred if abs((w[2] + w[4]) / 2 - (g["y"] + g["h"] / 2)) < g["h"] and w[3] > g["x"] - 30 and w[1] < g["x"] + g["w"] + 30]
                    print("MISS", repr(g["t"]), round(g["x"]), round(g["y"]), "near:", [(w[0], round(w[1]), round(w[3])) for w in near][:4])
        c["extra"] += len(pred) - len(used)
        c["pages"] += 1
    if c["words"]:
        kinds = "  ".join(f"{k[5:]} {v}" for k, v in sorted(c.items()) if k.startswith("miss_"))
        print(f"{mode:5} {os.path.basename(d.rstrip('/')):10} {s}x  word {c['hit']/c['words']:.1%}"
              f"  extra {c['extra']}  dup {c['dup']}  ({c['words']} words, {c['pages']} pages)  misses: {kinds}", flush=True)
