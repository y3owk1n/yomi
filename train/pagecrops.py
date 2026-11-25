"""Cuts training lines from rendered pages the way yomi reads them: the line
finder's own crops (bin/crops), labelled from the page's DOM words.

A crop keeps only when every word whose centre and middle sit in it is whole
inside it and visible (the page rendered without its text differs from it
there), and no other word reaches into it. A crop with no word near it is a
negative and reads as nothing. Lines go to OUT-western or OUT-cjk in
prerender.py's format, Latin-only lines from CJK pages to Western.

Usage: python pagecrops.py PAGEDIR OUT [NEG_KEEP]
"""
import array
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

TRAIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAIN)
from charsets import CJK, WESTERN  # noqa: E402
from prep import prepare  # noqa: E402
from synth import Alphabet, coverage_of  # noqa: E402

CROPS = os.path.join(TRAIN, "..", "bin", "crops")
VISIBLE = 40  # 90th percentile luma difference a word needs against the textless render
# Characters the open fonts closing every page's stack draw. A word with any
# other character came from a system font and stays out.
COVERED = set()
for _f in ["train/Arimowght.ttf", "train/NotoSanswdth_wght.ttf", "cjk-train/NotoSansJPwght.ttf", "cjk-train/NotoSansKRwght.ttf",
           "cjk-train/NotoSansSCwght.ttf", "cjk-train/NotoSansTCwght.ttf"]:
    COVERED |= {chr(c) for c in coverage_of(os.path.join(TRAIN, "fonts", _f))}
ALPHA = {"western": (set(WESTERN), Alphabet(WESTERN)), "cjk": (set(CJK), Alphabet(CJK))}


def luma(a):
    a = a.astype(np.int32)
    return (299 * a[..., 0] + 587 * a[..., 1] + 114 * a[..., 2]) // 1000


def down2(path, out):
    if os.path.exists(out):
        return
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2].reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    Image.fromarray(np.round(a).astype(np.uint8)).save(out)


class Writer:
    def __init__(self, path):
        self.path, self.cols = path, open(path + ".lines", "wb")
        self.offsets, self.widths, self.labels, self.label_offsets, self.texts = [], [], array.array("i"), [0], []
        self.total = 0

    def add(self, x, lab, text):
        self.offsets.append(self.total)
        self.widths.append(x.shape[1])
        self.total += x.shape[1]
        self.cols.write(np.ascontiguousarray(x.T).tobytes())
        self.labels.extend(lab)
        self.label_offsets.append(len(self.labels))
        self.texts.append(text)

    def close(self):
        self.cols.close()
        np.savez(self.path + ".npz", offsets=np.array(self.offsets, np.int64), widths=np.array(self.widths, np.int32),
                 labels=np.array(self.labels, np.int32), label_offsets=np.array(self.label_offsets, np.int64))
        open(self.path + ".txt", "w", encoding="utf-8").write("\n".join(self.texts))
        return len(self.widths)


def label_of(words):
    words = sorted(words, key=lambda w: w["x"])
    h = float(np.median([w["h"] for w in words]))
    out = words[0]["t"]
    for a, b in zip(words, words[1:]):
        gap = b["x"] - (a["x"] + a["w"])
        out += (" " if b.get("sp") or gap > 0.3 * h else "") + b["t"]
    return out


def main():
    d, out = sys.argv[1], sys.argv[2]
    neg_keep = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    names = [n for n in open(os.path.join(d, "pages.txt")).read().split()
             if os.path.exists(os.path.join(d, n + "@2x.png")) and os.path.exists(os.path.join(d, n + "-bare@2x.png"))]
    for n in names:
        down2(os.path.join(d, n + "@2x.png"), os.path.join(d, n + "@1x.png"))
    boxes = {}
    for scale in (1, 2):
        r = subprocess.run([CROPS, d, str(scale)], input="\n".join(names), capture_output=True, text=True, check=True)
        for rec in map(json.loads, r.stdout.splitlines()):
            boxes[(rec["page"], scale)] = rec["boxes"] or []
    writers = {f: Writer(f"{out}-{f}") for f in ALPHA}
    rng = np.random.default_rng(0)
    stats = {"kept": 0, "cut": 0, "intrusion": 0, "invisible": 0, "alphabet": 0, "neg": 0}
    for n in names:
        words = json.load(open(os.path.join(d, n + ".json")))
        full = luma(np.asarray(Image.open(os.path.join(d, n + "@2x.png")).convert("RGB")))
        bare = luma(np.asarray(Image.open(os.path.join(d, n + "-bare@2x.png")).convert("RGB")))
        diff = np.abs(full - bare)
        for w in words:
            x0, y0 = int(w["x"] * 2), int(w["y"] * 2)
            part = diff[y0:int((w["y"] + w["h"]) * 2) + 1, x0:int((w["x"] + w["w"]) * 2) + 1]
            # Both halves must show ink, so a word a box clips stays out.
            h = part.shape[0] // 2
            w["vis"] = min(float(np.percentile(part[:h], 90)), float(np.percentile(part[h:], 90))) \
                if part.shape[0] >= 2 and part.size else 0.0
        for scale in (1, 2):
            frame = np.asarray(Image.open(os.path.join(d, f"{n}@{scale}x.png")).convert("RGB"))
            for cx0, cy0, cx1, cy1, ix0, iy0, ix1, iy1 in boxes[(n, scale)]:
                ih = iy1 - iy0
                inl, bad = [], None
                for w in words:
                    wx1, wy1 = w["x"] + w["w"], w["y"] + w["h"]
                    ox = max(0, min(cx1, wx1) - max(cx0, w["x"]))
                    oy = max(0, min(cy1, wy1) - max(cy0, w["y"]))
                    if ox * oy <= 0:
                        continue
                    mx, my = w["x"] + w["w"] / 2, w["y"] + w["h"] / 2
                    if cx0 <= mx <= cx1 and iy0 - 0.25 * ih <= my <= iy1 + 0.25 * ih:
                        if ox < 0.85 * w["w"]:
                            bad = "cut"
                        elif w["vis"] < VISIBLE or not set(w["t"]) <= COVERED:
                            bad = "invisible"
                        inl.append(w)
                    elif ox * oy > 0.25 * w["w"] * w["h"]:
                        bad = "intrusion"
                if bad:
                    stats[bad] += 1
                    continue
                text = label_of(inl) if inl else ""
                chars = set(text) - {" "}
                fams = [f for f in ("western", "cjk") if chars <= ALPHA[f][0]]
                if not inl:
                    if rng.random() > neg_keep:
                        continue
                    fams = ["western", "cjk"][rng.integers(2):][:1]
                    stats["neg"] += 1
                elif not fams:
                    stats["alphabet"] += 1
                    continue
                r0x, r0y = int(np.floor(cx0 * scale)), int(np.floor(cy0 * scale))
                r1x, r1y = int(np.ceil(cx1 * scale)), int(np.ceil(cy1 * scale))
                crop = frame[max(0, r0y):r1y, max(0, r0x):r1x]
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue
                x = np.round(prepare(crop) * 255).astype(np.uint8)
                fam = fams[0] if len(fams) == 1 or rng.random() < 0.7 else fams[1]
                writers[fam].add(x, ALPHA[fam][1].labels(text), text)
                stats["kept"] += bool(inl)
    for f, w in writers.items():
        print(f, w.close(), "lines")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
