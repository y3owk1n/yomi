"""Synthetic training lines: open fonts rendered the ways screens draw text.

Each sample is one line crop, padded like a line finder's box, run through
prep.prepare. Variation: font, weight, size at 1x and 2x, hinted FreeType or
supersampled unhinted rendering (the macOS look), coverage gamma (font
smoothing), light, dark and coloured themes, subpixel colour fringes, a
neighbouring line or an outline intruding, and a few crops with no text.
"""
import glob
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prep import prepare
from textsrc import CHARSET, TRAIN_WORDS, phrase

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHABET = " " + CHARSET  # index 0 is the CTC blank, so labels are 1 + index
CHAR_INDEX = {c: i + 1 for i, c in enumerate(ALPHABET)}


def fonts(split):
    return sorted(glob.glob(os.path.join(HERE, "fonts", split, "*.ttf")))


class Font:
    _cache = {}

    @classmethod
    def get(cls, path, px, weight):
        key = (path, px, weight)
        f = cls._cache.get(key)
        if f is None:
            f = ImageFont.truetype(path, px)
            try:
                axes = f.get_variation_axes()
            except OSError:
                axes = []
            if axes:
                vals = []
                for a in axes:
                    name = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
                    if name.lower().startswith("weight"):
                        vals.append(min(max(weight, a["minimum"]), a["maximum"]))
                    elif name.lower().startswith("optical"):
                        vals.append(min(max(px, a["minimum"]), a["maximum"]))
                    else:
                        vals.append(a["default"])
                f.set_variation_by_axes(vals)
            if len(cls._cache) > 4000:
                cls._cache.clear()
            cls._cache[key] = f
        return f


def coverage(text, path, px, weight, supersample, subpixel=False):
    """Coverage of text on its line box and the line box height in pixels.
    Grey (h, w), or with subpixel (h, w, 3), one column of subpixels per
    channel, the way LCD antialiasing draws."""
    ss = 3 if subpixel else (4 if supersample else 1)
    font = Font.get(path, px * ss, weight)
    asc, desc = font.getmetrics()
    w = int(math.ceil(font.getlength(text))) + 2 * ss
    img = Image.new("L", (w, asc + desc + 2 * ss), 0)
    ImageDraw.Draw(img).text((ss, ss), text, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255
    if ss > 1:
        h2, w2 = (a.shape[0] // ss) * ss, (a.shape[1] // ss) * ss
        a = a[:h2, :w2]
        if subpixel:
            a = a.reshape(h2 // ss, ss, w2).mean(axis=1).reshape(h2 // ss, w2 // ss, ss)
        else:
            a = a.reshape(h2 // ss, ss, w2 // ss, ss).mean(axis=(1, 3))
    return a, (asc + desc) / ss


def theme(rng):
    r = rng.random()
    if r < 0.45:  # light
        bg = [rng.randint(225, 255)] * 3
        fg = [rng.randint(0, 90)] * 3
    elif r < 0.75:  # dark
        bg = [rng.randint(15, 60)] * 3
        fg = [rng.randint(175, 255)] * 3
    elif r < 0.85:  # muted grey text
        bg = [rng.randint(235, 255)] * 3
        fg = [rng.randint(100, 150)] * 3
    else:  # coloured: link, button, tinted panel
        hue = [rng.randint(0, 255) for _ in range(3)]
        if rng.random() < 0.5:
            bg = [rng.randint(220, 255) for _ in range(3)]
            fg = [min(c, 140) for c in hue]
        else:
            bg = [min(c, 150) for c in hue]
            fg = [255, 255, 255]
    if rng.random() < 0.2:
        bg = [min(255, max(0, c + rng.randint(-12, 12))) for c in bg]
    return np.array(bg, np.float32), np.array(fg, np.float32)


def render(rng, text, path, scale):
    px = round(rng.uniform(10, 18) * scale)
    weight = rng.choice([400, 400, 400, 450, 500, 600, 700])
    supersample = rng.random() < 0.5
    subpixel = rng.random() < 0.1
    a, line_h = coverage(text, path, px, weight, supersample, subpixel)
    a = a ** rng.uniform(0.6, 1.3)
    if rng.random() < 0.15:  # slight blur or smear
        a = (a + np.roll(a, 1, axis=1) * rng.uniform(0, 0.5)) / 1.25
        a = np.clip(a, 0, 1)
    chans = a if subpixel else None
    if subpixel:
        a = a.mean(axis=2)

    # line box -> padded crop, as a line finder would cut it
    h, w = a.shape
    pads = [int(round(rng.uniform(0.10, 0.35) * line_h)) for _ in range(4)]
    canvas = np.zeros((h + pads[1] + pads[3], w + pads[0] + pads[2]), np.float32)
    canvas[pads[1]:pads[1] + h, pads[0]:pads[0] + w] = a
    if subpixel:
        sub = np.repeat(canvas[..., None], 3, axis=2)
        sub[pads[1]:pads[1] + h, pads[0]:pads[0] + w] = chans
    if rng.random() < 0.15:  # part of the line above or below
        other, _ = coverage(phrase(rng, TRAIN_WORDS, 6), path, px, weight, supersample, False)
        cut = rng.randint(1, max(1, pads[1] + 2))
        strip = other[-cut:, : canvas.shape[1]] if rng.random() < 0.5 else other[:cut, : canvas.shape[1]]
        if rng.random() < 0.5:
            canvas[: strip.shape[0], : strip.shape[1]] = np.maximum(canvas[: strip.shape[0], : strip.shape[1]], strip)
        else:
            canvas[-strip.shape[0]:, : strip.shape[1]] = np.maximum(canvas[-strip.shape[0]:, : strip.shape[1]], strip)
    if rng.random() < 0.08:  # an input or button outline near the edge
        y = rng.choice([rng.randint(0, max(0, pads[1] - 1)), canvas.shape[0] - 1 - rng.randint(0, max(0, pads[3] - 1))])
        canvas[y, :] = np.maximum(canvas[y, :], rng.uniform(0.2, 0.6))

    bg, fg = theme(rng)
    cov = canvas[..., None]
    if subpixel:
        sub = np.maximum(sub, canvas[..., None] * (sub == 0))  # keep intruders and outlines
        cov = sub
    rgb = bg[None, None, :] * (1 - cov) + fg[None, None, :] * cov
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def blank(rng, scale):
    """A crop with no text: a flat field, an outline or a simple icon shape."""
    h = round(rng.uniform(14, 26) * scale)
    w = round(rng.uniform(1, 6) * h)
    bg, fg = theme(rng)
    img = Image.new("RGB", (w, h), tuple(int(c) for c in bg))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(0, 3)):
        x0, y0 = rng.randint(0, w - 2), rng.randint(0, h - 2)
        x1, y1 = rng.randint(x0 + 1, w), rng.randint(y0 + 1, h)
        shape = rng.choice(["rect", "ellipse", "line"])
        col = tuple(int(c) for c in fg)
        if shape == "rect":
            d.rectangle([x0, y0, x1, y1], outline=col, width=max(1, scale))
        elif shape == "ellipse":
            d.ellipse([x0, y0, x1, y1], outline=col, width=max(1, scale))
        else:
            d.line([x0, y0, x1, y1], fill=col, width=max(1, scale))
    return np.asarray(img)


def sample(rng, split="train"):
    """(x float32 (H, W), label list[int], text)."""
    scale = 1 if rng.random() < 0.5 else 2
    if rng.random() < 0.04:
        return prepare(blank(rng, scale)), [], ""
    n = rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 12])
    text = phrase(rng, TRAIN_WORDS, n)[:70]
    path = rng.choice(fonts(split))
    rgb = render(rng, text, path, scale)
    return prepare(rgb), [CHAR_INDEX[c] for c in text if c in CHAR_INDEX], text
