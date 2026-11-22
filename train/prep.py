"""Crop preprocessing, written so a Go port can reproduce it exactly with
integer luma and plain loops.

1. Luma per pixel, Neru's formula: (299 R + 587 G + 114 B) / 1000, integer.
2. Background: the luma most of the crop's pixels have, the lowest on a tie.
   A crop fitted to text is mostly background, while its border can run past
   the button or field the text sits in. Ink: the pixel farthest from it. Ink strength x = clip((bg - v) * sign / max(32, |bg - ink|), 0, 1),
   so ink is 1 and background 0 whatever the polarity.
3. Underlines: in the crop's bottom quarter, runs of x >= 0.5 at least the
   crop's height long that touch neither side, in a band of rows at most a
   tenth of the crop thick, are set to 0.
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
    bg = int(np.bincount(gray.ravel(), minlength=256).argmax())
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

    def runs(y):
        row, out, i = x[y], [], 0
        while i < w:
            if row[i] < 0.5:
                i += 1
                continue
            s = i
            while i < w and row[i] >= 0.5:
                i += 1
            if i - s >= h and s > 0 and i < w:
                out.append((s, i))
        return out

    thin = max(2, h // 10)
    y = h * 3 // 4
    while y < h:
        if not runs(y):
            y += 1
            continue
        s = y
        while y < h and runs(y):
            y += 1
        if y - s > thin:
            continue
        for r in range(s, y):
            for a, b in runs(r):
                x[r, a:b] = 0


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
