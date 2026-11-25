"""Cuts training lines from rendered pages the way yomi reads them: the line
finder's own crops (bin/crops), labelled from the page's DOM words.

A crop keeps only when every word whose centre and middle sit in it is whole
inside it and visible (the page rendered without its text differs from it
there), and no other word reaches into it. A crop with no word near it is a
negative and reads as nothing. Lines go to OUT-FAMILY in prerender.py's
format (family_of says which), and every line also goes to OUT-router with
its routed family as the label.

Usage: python pagecrops.py PAGEDIR OUT [NEG_KEEP]
"""
import array
import json
import os
import random
import subprocess
import sys
import unicodedata

import numpy as np
from PIL import Image

TRAIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAIN)
from families import FAMILIES, ROUTED  # noqa: E402
from prep import prepare  # noqa: E402
from shaping import runs  # noqa: E402
from synth import Alphabet, coverage_of  # noqa: E402

CROPS = os.path.join(TRAIN, "..", "bin", "crops")
VISIBLE = 40  # 90th percentile luma difference a word needs against the textless render
# Characters the open fonts closing every page's stack draw. A word with any
# other character came from a system font and stays out.
COVERED = set()
for _f in ["train/Arimowght.ttf", "train/NotoSanswdth_wght.ttf", "cjk-train/NotoSansJPwght.ttf", "cjk-train/NotoSansKRwght.ttf",
           "cjk-train/NotoSansSCwght.ttf", "cjk-train/NotoSansTCwght.ttf", "rtl-train/NotoSansArabicwdth_wght.ttf",
           "rtl-train/NotoSansHebrewwdth_wght.ttf"] + \
        [f"indic-train/NotoSans{s}wdth_wght.ttf" for s in
         ["Devanagari", "Bengali", "Gujarati", "Gurmukhi", "Kannada", "Malayalam", "Sinhala", "Tamil", "Telugu"]] + \
        [f"sea-train/NotoSans{s}wdth_wght.ttf" for s in ["Thai", "Lao", "Khmer", "Myanmar"]] + \
        [f"cau-train/NotoSans{s}wdth_wght.ttf" for s in ["Georgian", "Armenian", "Ethiopic"]]:
    COVERED |= {chr(c) for c in coverage_of(os.path.join(TRAIN, "fonts", _f))}
rng = random.Random(0)
ALPHA = {f: (set(FAMILIES[f][0]), Alphabet(FAMILIES[f][0])) for f in ROUTED}


def visual(word):
    """A word in the order the shaped families train on: right-to-left runs
    reversed, as shaping.coverage labels its lines."""
    base_rtl, rs = runs(word)
    return "".join(t[::-1] if rtl else t for rtl, t in (reversed(rs) if base_rtl else rs))


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


def settled(word):
    """Whether a word's visual order is its own on a right-to-left line. A
    word with no right-to-left letter keeps its order only when it starts
    and ends with a letter or digit and has no currency or percent sign.
    After Arabic, "$638" draws as "638$"."""
    if any(unicodedata.bidirectional(ch) in ("R", "AL") for ch in word):
        return True
    return word[0].isalnum() and word[-1].isalnum() and not any(unicodedata.bidirectional(ch) == "ET" for ch in word)


def label_of(words, rtl):
    """The line's words left to right, or None when bidi context would order
    one differently. On a right-to-left page each word is in visual order,
    and the word on the right is the one logically before."""
    words = sorted(words, key=lambda w: w["x"])
    if rtl and not all(settled(w["t"]) for w in words):
        return None
    h = float(np.median([w["h"] for w in words]))
    t = [visual(w["t"]) if rtl else w["t"] for w in words]
    out = t[0]
    for k, (a, b) in enumerate(zip(words, words[1:]), 1):
        gap = b["x"] - (a["x"] + a["w"])
        out += (" " if (a if rtl else b).get("sp") or gap > 0.3 * h else "") + t[k]
    return out


def family_of(chars, page):
    """The family a line trains: Latin-only lines go to Western (one in twenty
    stays with its page's family, which must read embedded Latin too), the
    rest to their page's family when it covers them, else the first that
    does, as the router labels them."""
    fams = [f for f in ROUTED if chars <= ALPHA[f][0]]
    if not fams:
        return None
    if fams[0] == "western" and page in fams and page != "western":
        return page if rng.random() < 0.05 else "western"
    return page if page in fams else fams[0]


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
    router = Writer(f"{out}-router")
    stats = {"kept": 0, "cut": 0, "intrusion": 0, "invisible": 0, "alphabet": 0, "bidi": 0, "neg": 0}
    for n in names:
        page = n.rsplit("-", 1)[0]
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
                text = label_of(inl, page == "rtl") if inl else ""
                if text is None:
                    stats["bidi"] += 1
                    continue
                chars = set(text) - {" "}
                if not inl:
                    if rng.random() > neg_keep:
                        continue
                    fam = page if rng.random() < 0.5 else "western"
                    stats["neg"] += 1
                else:
                    fam = family_of(chars, page)
                    if fam is None:
                        stats["alphabet"] += 1
                        continue
                r0x, r0y = int(np.floor(cx0 * scale)), int(np.floor(cy0 * scale))
                r1x, r1y = int(np.ceil(cx1 * scale)), int(np.ceil(cy1 * scale))
                crop = frame[max(0, r0y):r1y, max(0, r0x):r1x]
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue
                x = np.round(prepare(crop) * 255).astype(np.uint8)
                writers[fam].add(x, ALPHA[fam][1].labels(text), text)
                # The router's label is the first family covering the line.
                r = next((i for i, f in enumerate(ROUTED) if chars <= ALPHA[f][0]), len(ROUTED)) if inl else len(ROUTED)
                router.add(x, [r], (ROUTED + ["none"])[r])
                stats["kept"] += bool(inl)
    for f, w in list(writers.items()) + [("router", router)]:
        print(f, w.close(), "lines")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
