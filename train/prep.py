"""Crop preprocessing, written so a Go port can reproduce it exactly with
integer luma and plain loops.

1. Luma per pixel, Neru's formula: (299 R + 587 G + 114 B) / 1000, integer.
2. Background: median of the crop's 1 px border. Ink: the pixel farthest from
   it. Ink strength x = clip((bg - v) * sign / max(32, |bg - ink|), 0, 1),
   so ink is 1 and background 0 whatever the polarity.
3. Rules: each run of x >= 0.5 within a row at least the crop's height long
   is set to 0. No glyph stroke is that long, so it is an underline, a
   strikethrough or a rule.
4. Height to H: integer box downsample by k = h // H when k >= 2, then
   bilinear (half-pixel centres, edge clamp) to height H, width scaled by the
   same factor, at least 4.
"""
import numpy as np

H = 32


def luma(rgb):
    rgb = rgb.astype(np.int32)
    return (299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2]) // 1000


def ink(gray):
    border = np.concatenate([gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]])
    bg = int(np.median(border))
    far = gray.min() if bg - gray.min() >= gray.max() - bg else gray.max()
    span = max(32, abs(bg - int(far)))
    sign = 1 if far < bg else -1
    return np.clip((bg - gray.astype(np.float32)) * sign / span, 0, 1)


def box_down(x, k):
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def bilinear(x, oh, ow):
    ih, iw = x.shape
    ys = np.clip((np.arange(oh) + 0.5) * ih / oh - 0.5, 0, ih - 1)
    xs = np.clip((np.arange(ow) + 0.5) * iw / ow - 0.5, 0, iw - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, ih - 1)
    x1 = np.minimum(x0 + 1, iw - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = x[y0][:, x0] * (1 - fx) + x[y0][:, x1] * fx
    bot = x[y1][:, x0] * (1 - fx) + x[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def erase_rules(x):
    h, w = x.shape
    for row in x:
        i = 0
        while i < w:
            if row[i] < 0.5:
                i += 1
                continue
            s = i
            while i < w and row[i] >= 0.5:
                i += 1
            if i - s >= h:
                row[s:i] = 0


def prepare(rgb):
    """RGB uint8 crop (h, w, 3) -> float32 (H, W') ink map."""
    x = ink(luma(rgb))
    erase_rules(x)
    k = x.shape[0] // H
    if k >= 2:
        x = box_down(x, k)
    h, w = x.shape
    ow = max(4, round(w * H / h))
    return bilinear(x, H, ow)
