"""Synthetic training lines: open fonts rendered the ways screens draw text.

Each sample is one line crop, padded like the line finder's box, run through
prep.prepare. A line renders in one primary font. Characters it lacks fall
back to the fallback fonts, run by run, as an OS draws them. Variation: font,
weight, size at 1x and 2x, hinted FreeType or supersampled unhinted rendering
(the macOS look), coverage gamma, light, dark, muted and coloured themes,
subpixel colour fringes, a neighbouring line or an outline in the padding, and
crops with no text at all (icons, avatars, photo detail, controls), labelled
empty so the model reads nothing there.
"""
import glob
import math
import os
import random

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

import shaping
from prep import prepare

HERE = os.path.dirname(os.path.abspath(__file__))


class Alphabet:
    """A model's characters. Label 0 is the CTC blank, 1 the space."""

    def __init__(self, charset):
        self.chars = " " + charset
        self.index = {c: i + 1 for i, c in enumerate(self.chars)}

    def labels(self, text):
        return [self.index[c] for c in text if c in self.index]


_cover = {}


def coverage_of(path):
    if path not in _cover:
        _cover[path] = set(TTFont(path, fontNumber=0, lazy=True).getBestCmap())
    return _cover[path]


def fonts(split):
    return sorted(glob.glob(os.path.join(HERE, "fonts", split, "*.[ot]t[fc]")))


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


def runs(text, primary, fallbacks):
    """Splits text into (font path, run) by the first font covering each
    character. None means no font has it."""
    out = []
    for ch in text:
        path = primary if ord(ch) in coverage_of(primary) or ch == " " else None
        if path is None:
            path = next((f for f in fallbacks if ord(ch) in coverage_of(f)), None)
        if out and out[-1][0] == path:
            out[-1][1] += ch
        else:
            out.append([path, ch])
    return out


def coverage(text, primary, fallbacks, px, weight, supersample, subpixel=False):
    """Coverage of text on its line box and the line box height in pixels.
    Grey (h, w), or with subpixel (h, w, 3), one column of subpixels per
    channel, the way LCD antialiasing draws. None when a character has no
    font."""
    rs = runs(text, primary, fallbacks)
    if any(p is None for p, _ in rs):
        return None, 0
    ss = 3 if subpixel else (4 if supersample else 1)
    main = Font.get(primary, px * ss, weight)
    asc, desc = main.getmetrics()
    parts = [(Font.get(p, px * ss, weight), t) for p, t in rs]
    w = sum(int(math.ceil(f.getlength(t))) for f, t in parts) + 2 * ss
    img = Image.new("L", (w, asc + desc + 2 * ss), 0)
    d = ImageDraw.Draw(img)
    x = ss
    for f, t in parts:
        a2 = f.getmetrics()[0]
        d.text((x, ss + asc - a2), t, fill=255, font=f)
        x += int(math.ceil(f.getlength(t)))
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


def control(rng, size):
    """Coverage (size, w) of a UI control that sits beside a label: a radio
    ring, a checked or empty checkbox, a check mark, a toggle or a chevron."""
    w = size * 2 if rng.random() < 0.15 else size
    img = Image.new("L", (w * 4, size * 4), 0)
    d = ImageDraw.Draw(img)
    S, W = size * 4, w * 4
    lw = max(4, S // 8)
    kind = rng.random()
    if W > S:  # toggle
        d.rounded_rectangle([2, S // 6, W - 2, S - S // 6], radius=S // 3, outline=255, width=lw)
        x = rng.choice([S // 6, W - S + S // 6])
        d.ellipse([x, S // 4, x + S // 2, S - S // 4], fill=255)
    elif kind < 0.35:  # radio, empty or selected
        d.ellipse([S // 8, S // 8, S - S // 8, S - S // 8], outline=255, width=lw)
        if rng.random() < 0.5:
            d.ellipse([S // 3, S // 3, S - S // 3, S - S // 3], fill=255)
    elif kind < 0.7:  # checkbox, empty or checked
        d.rounded_rectangle([S // 8, S // 8, S - S // 8, S - S // 8], radius=S // 8, outline=255, width=lw)
        if rng.random() < 0.5:
            d.line([(S // 4, S // 2), (S // 2.2, S - S // 4), (S - S // 4, S // 4)], fill=255, width=lw)
    elif kind < 0.85:  # a bare check mark
        d.line([(S // 6, S // 2), (S // 2.4, S - S // 5), (S - S // 6, S // 5)], fill=255, width=lw)
    else:  # chevron
        if rng.random() < 0.5:
            d.line([(S // 3, S // 5), (S - S // 3, S // 2), (S // 3, S - S // 5)], fill=255, width=lw)
        else:
            d.line([(S // 5, S // 3), (S // 2, S - S // 3), (S - S // 5, S // 3)], fill=255, width=lw)
    a = np.asarray(img, dtype=np.float32) / 255
    return a.reshape(size, 4, w, 4).mean(axis=(1, 3))


def render(rng, text, primary, fallbacks, scale, other_text, shaped=False):
    """The line as an RGB crop, and its label: text as written, or in visual
    order for shaped lines."""
    px = round(rng.uniform(10, 18) * scale)
    weight = rng.choice([400, 400, 400, 450, 500, 600, 700])
    supersample = rng.random() < 0.5
    subpixel = not shaped and rng.random() < 0.1
    label = text
    if shaped:
        a, line_h, label = shaping.coverage(text, primary, px, weight, supersample)
    else:
        a, line_h = coverage(text, primary, fallbacks, px, weight, supersample, subpixel)
    if a is None:
        return None, label
    a = a ** rng.uniform(0.6, 1.3)
    if rng.random() < 0.15:  # slight blur or smear
        a = np.clip((a + np.roll(a, 1, axis=1) * rng.uniform(0, 0.5)) / 1.25, 0, 1)
    chans = a if subpixel else None
    if subpixel:
        a = a.mean(axis=2)
    if rng.random() < 0.12:
        # A control beside the label, as radio buttons and checkboxes sit in
        # forms. The label stays the text alone, so the model learns to read
        # nothing there.
        size = max(6, int(line_h * rng.uniform(0.55, 0.8)))
        ctl = control(rng, size)
        gap = int(line_h * rng.uniform(0.25, 0.6))
        y0 = max(0, (a.shape[0] - size) // 2 + int(line_h * rng.uniform(-0.05, 0.1)))
        col = np.zeros((a.shape[0], ctl.shape[1] + gap), np.float32)
        col[y0:y0 + size, :ctl.shape[1]] = ctl[: a.shape[0] - y0]
        before = rng.random() < 0.8
        a = np.hstack([col, a] if before else [a, col[:, ::-1]])
        if subpixel:
            c3 = np.repeat(col[..., None], 3, axis=2)
            chans = np.concatenate([c3, chans] if before else [chans, c3[:, ::-1]], axis=1)

    # Line box to padded crop, as the line finder cuts it.
    h, w = a.shape
    pads = [int(round(rng.uniform(0.10, 0.35) * line_h)) for _ in range(4)]
    canvas = np.zeros((h + pads[1] + pads[3], w + pads[0] + pads[2]), np.float32)
    canvas[pads[1]:pads[1] + h, pads[0]:pads[0] + w] = a
    if subpixel:
        sub = np.repeat(canvas[..., None], 3, axis=2)
        sub[pads[1]:pads[1] + h, pads[0]:pads[0] + w] = chans
    if rng.random() < 0.15:  # part of the line above or below
        if shaped:
            other = shaping.coverage(other_text, primary, px, weight)[0]
        else:
            other, _ = coverage(other_text, primary, fallbacks, px, weight, supersample, False)
        if other is not None:
            cut = rng.randint(1, max(1, pads[1] + 2))
            strip = other[-cut:, : canvas.shape[1]] if rng.random() < 0.5 else other[:cut, : canvas.shape[1]]
            sh, sw = strip.shape
            if rng.random() < 0.5:
                canvas[:sh, :sw] = np.maximum(canvas[:sh, :sw], strip)
            else:
                canvas[-sh:, :sw] = np.maximum(canvas[-sh:, :sw], strip)
    if rng.random() < 0.08:  # an input or button outline near the edge
        y = rng.choice([rng.randint(0, max(0, pads[1] - 1)), canvas.shape[0] - 1 - rng.randint(0, max(0, pads[3] - 1))])
        canvas[y, :] = np.maximum(canvas[y, :], rng.uniform(0.2, 0.6))

    bg, fg = theme(rng)
    cov = canvas[..., None]
    if subpixel:
        cov = np.maximum(sub, canvas[..., None] * (sub == 0))  # keep intruders and outlines
    rgb = bg[None, None, :] * (1 - cov) + fg[None, None, :] * cov
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8), label


def blank(rng, scale):
    """A crop with no text: icons and outlines, an avatar, a toggle or
    checkbox, a slice of a photo, or a flat field."""
    h = round(rng.uniform(14, 30) * scale)
    w = round(rng.uniform(1, 8) * h)
    bg, fg = theme(rng)
    img = Image.new("RGB", (w, h), tuple(int(c) for c in bg))
    d = ImageDraw.Draw(img)
    col = tuple(int(c) for c in fg)
    kind = rng.random()
    if kind < 0.25:  # a photo slice: smooth colour plus noise and blobs
        a = np.zeros((h, w, 3), np.float32)
        for c in range(3):
            gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
            yy, xx = np.mgrid[0:h, 0:w]
            a[..., c] = rng.uniform(40, 200) + 60 * (gx * xx / w + gy * yy / h)
        a += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 20), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(0, 8)):
            x0, y0 = rng.randint(-h, w), rng.randint(-h, h)
            r = rng.randint(2, h)
            d.ellipse([x0, y0, x0 + r, y0 + r], fill=tuple(rng.randint(0, 255) for _ in range(3)))
        return np.asarray(img)
    if kind < 0.4:  # avatar circles and toggles
        for _ in range(rng.randint(1, 3)):
            s = rng.randint(h // 2, h)
            x0 = rng.randint(0, max(0, w - s))
            if rng.random() < 0.5:
                d.ellipse([x0, (h - s) // 2, x0 + s, (h + s) // 2], fill=col)
            else:
                d.rounded_rectangle([x0, (h - s) // 2, x0 + 2 * s, (h + s) // 2], radius=s // 2, outline=col, width=max(1, scale))
                d.ellipse([x0 + 2, (h - s) // 2 + 2, x0 + s - 2, (h + s) // 2 - 2], fill=col)
        return np.asarray(img)
    for _ in range(rng.randint(0, 5)):  # icon-like strokes and shapes
        x0, y0 = rng.randint(0, w - 2), rng.randint(0, h - 2)
        x1, y1 = rng.randint(x0 + 1, w), rng.randint(y0 + 1, h)
        shape = rng.choice(["rect", "ellipse", "line", "poly"])
        if shape == "rect":
            d.rectangle([x0, y0, x1, y1], outline=col, width=max(1, scale))
        elif shape == "ellipse":
            d.ellipse([x0, y0, x1, y1], outline=col, width=max(1, scale))
        elif shape == "line":
            d.line([x0, y0, x1, y1], fill=col, width=max(1, scale))
        else:
            pts = [(rng.randint(x0, x1), rng.randint(y0, y1)) for _ in range(rng.randint(3, 6))]
            d.polygon(pts, fill=col if rng.random() < 0.5 else None, outline=col)
    return np.asarray(img)


class Generator:
    """Draws samples for one model family: text from line_fn over langs (the
    split half of the words), primary fonts from the folders, labels in the
    family's alphabet."""

    def __init__(self, charset, langs, line_fn, folders, blank_rate=0.08, split="train", shaped=False):
        self.shaped = shaped
        self.alphabet = Alphabet(charset)
        self.charset = set(charset)
        self.langs, self.line_fn, self.split, self.blank_rate = langs, line_fn, split, blank_rate
        self.primaries = [f for d in folders for f in fonts(d)]
        self.fallbacks = fonts("fallback")

    def sample(self, rng):
        """(x float32 (H, W), labels list[int], text)."""
        scale = 1 if rng.random() < 0.5 else 2
        if rng.random() < self.blank_rate:
            return prepare(blank(rng, scale)), [], ""
        for _ in range(20):
            text = self.line_fn(rng, self.langs, self.charset, self.split)[:70].strip()
            if not text:
                continue
            # The primary font must have every letter. Signs may fall back.
            need = {ord(c) for c in text if c.isalnum()}
            fits = [f for f in self.primaries if need <= coverage_of(f)]
            if not fits:
                continue
            primary = rng.choice(fits)
            rgb, label = render(rng, text, primary, self.fallbacks, scale,
                                self.line_fn(rng, self.langs, self.charset, self.split, 6), self.shaped)
            if rgb is not None:
                return prepare(rgb), self.alphabet.labels(label), label
        raise RuntimeError("no font covers the generated text")
