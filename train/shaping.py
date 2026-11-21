"""Renders text that needs shaping: Arabic and Hebrew right to left, and the
scripts whose letters join, combine or stack (Indic, Thai, Khmer, Myanmar).

HarfBuzz (uharfbuzz) shapes each run and FreeType (freetype-py) rasterizes the
glyphs. A line is split into right-to-left and left-to-right runs, laid out
in visual order. The label is the line's characters in that visual order:
right-to-left runs reversed, left-to-right runs as written. The model reads
left to right, and the Go decoder turns right-to-left runs back into logical
order.
"""
import unicodedata

import freetype
import numpy as np
import uharfbuzz as hb

_RTL = ("ARABIC", "HEBREW", "SYRIAC", "THAANA", "NKO")


def is_rtl(ch):
    return unicodedata.bidirectional(ch) in ("R", "AL")


def runs(text):
    """Splits text into [rtl, run] by direction. Neutral characters (spaces,
    punctuation) join the run before them when the run after has the same
    direction, and otherwise take the line's direction."""
    base_rtl = any(is_rtl(c) for c in text)
    out = []
    for ch in text:
        b = unicodedata.bidirectional(ch)
        strong = True if b in ("R", "AL") else False if b in ("L", "EN", "AN") else None
        if strong is None:
            if out:
                out[-1][1] += ch
            else:
                out.append([base_rtl, ch])
            continue
        if out and out[-1][0] == strong:
            out[-1][1] += ch
        else:
            out.append([strong, ch])
    # Trailing neutrals of a run whose direction differs from the line's go
    # back to the line's direction.
    fixed = []
    for rtl, t in out:
        if rtl != base_rtl:
            core = t.rstrip(" .,:;!?()[]«»\"'-")
            tail = t[len(core):]
            fixed.append([rtl, core])
            if tail:
                fixed.append([base_rtl, tail])
        else:
            fixed.append([rtl, t])
    fixed = [r for r in fixed if r[1]]
    return base_rtl, fixed


class Face:
    _cache = {}

    @classmethod
    def get(cls, path, weight=400):
        key = (path, weight)
        if key not in cls._cache:
            font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
            ft = freetype.Face(path)
            try:
                axes = ft.get_variation_info().axes
            except Exception:  # a static font has no variation table
                axes = []
            if axes:
                coords = [min(max(weight, a.minimum), a.maximum) if a.tag == "wght" else a.default for a in axes]
                ft.set_var_design_coords(coords)
                font.set_variations({a.tag: c for a, c in zip(axes, coords)})
            cls._cache[key] = (font, ft)
        return cls._cache[key]


def shape(text, path, px, rtl, weight=400):
    """Glyph ids and positions of one run, in visual order, in pixels at size
    px."""
    font, _ = Face.get(path, weight)
    scale = px * 64
    font.scale = (scale, scale)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    buf.direction = "rtl" if rtl else "ltr"
    hb.shape(font, buf, {})
    out, x = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, x + pos.x_offset / 64, pos.y_offset / 64))
        x += pos.x_advance / 64
    return out, x


def coverage(text, path, px, weight=400, supersample=False):
    """Coverage (h, w) of text set in font path at px pixels, its line box
    height, and the label text in visual order. Supersampling draws at four
    times the size and averages down, as unhinted screen text looks."""
    if supersample:
        a, lh, label = coverage(text, path, px * 4, weight)
        h2, w2 = a.shape[0] // 4 * 4, a.shape[1] // 4 * 4
        return a[:h2, :w2].reshape(h2 // 4, 4, w2 // 4, 4).mean(axis=(1, 3)), lh / 4, label
    base_rtl, rs = runs(text)
    order = list(reversed(rs)) if base_rtl else rs
    _, ft = Face.get(path, weight)
    ft.set_pixel_sizes(0, px)
    asc, desc = ft.size.ascender / 64, -ft.size.descender / 64
    shaped, width = [], 0.0
    for rtl, t in order:
        glyphs, adv = shape(t, path, px, rtl, weight)
        shaped.append((glyphs, width))
        width += adv
    h = int(np.ceil(asc + desc)) + 2
    canvas = np.zeros((h, int(np.ceil(width)) + 4), np.float32)
    for glyphs, x0 in shaped:
        for gid, gx, gy in glyphs:
            ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            g = ft.glyph
            bm = g.bitmap
            if bm.width == 0 or bm.rows == 0:
                continue
            a = np.array(bm.buffer, np.float32).reshape(bm.rows, bm.pitch)[:, : bm.width] / 255
            x = int(round(x0 + gx + g.bitmap_left)) + 2
            y = int(round(asc - gy - g.bitmap_top)) + 1
            ys, xs = max(0, y), max(0, x)
            ye, xe = min(h, y + bm.rows), min(canvas.shape[1], x + bm.width)
            if ye > ys and xe > xs:
                canvas[ys:ye, xs:xe] = np.maximum(canvas[ys:ye, xs:xe], a[ys - y:ye - y, xs - x:xe - x])
    label = "".join(t[::-1] if rtl else t for rtl, t in order)
    return canvas, asc + desc, label
