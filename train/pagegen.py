"""Training pages for the recognizers: UI-like HTML built around what real
pages showed yomi missing. Display type 20-72 px in open display fonts, text
over gradients and noise textures, titles over thumbnails, promo tiles that
mix sizes in one line, pills and chips, dimming scrims, plus ordinary body
text so the old distribution stays. Each page draws its text from one family
in families.py. pagerender.swift renders them in WebKit and words.js returns
each visible word with its box and whether a space precedes it.

Usage: python pagegen.py OUTDIR COUNT [SEED] [FAMILY,...]  (default western,cjk)
"""
import glob
import html
import os
import random
import sys

TRAIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAIN)
from families import FAMILIES, ROUTED  # noqa: E402

DISPLAY_LATIN = ["Anton-Regular", "Oswaldwght", "PlayfairDisplaywght", "AbrilFatface-Regular",
                 "Lobster-Regular", "Pacifico-Regular", "ArchivoBlack-Regular", "Montserratwght", "Poppins-Bold",
                 "Poppins-ExtraBold", "DMSerifDisplay-Regular", "Merriweatheropsz_wdth_wght", "Ralewaywght"]
FALLBACK = ["Arimowght", "NotoSanswdth_wght", "NotoSansJPwght", "NotoSansKRwght", "NotoSansSCwght", "NotoSansTCwght",
            "NotoSansArabicwdth_wght", "NotoSansHebrewwdth_wght", "NotoSansDevanagariwdth_wght", "NotoSansBengaliwdth_wght",
            "NotoSansGujaratiwdth_wght", "NotoSansGurmukhiwdth_wght", "NotoSansKannadawdth_wght", "NotoSansMalayalamwdth_wght",
            "NotoSansSinhalawdth_wght", "NotoSansTamilwdth_wght", "NotoSansTeluguwdth_wght", "NotoSansThaiwdth_wght",
            "NotoSansLaowdth_wght", "NotoSansKhmerwdth_wght", "NotoSansMyanmarwdth_wght", "NotoSansGeorgianwdth_wght",
            "NotoSansArmenianwdth_wght", "NotoSansEthiopicwdth_wght"]
DISPLAY_CJK = ["DelaGothicOne-Regular", "MochiyPopOne-Regular", "ZenMaruGothic-Bold", "YujiSyuku-Regular",
               "MaShanZheng-Regular", "ZCOOLKuaiLe-Regular", "BlackHanSans-Regular", "DoHyeon-Regular",
               "RocknRollOne-Regular", "ReggaeOne-Regular", "NotoSansJPwght"]

JS = r"""
window.__words = () => {
  const out = [];
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n, prevEndsSpace = true;
  while ((n = walk.nextNode())) {
    const p = n.parentElement;
    if (!p || p.closest("script,style")) continue;
    const t = n.textContent;
    const re = /\S+/g; let m;
    while ((m = re.exec(t))) {
      const sp = m.index > 0 ? /\s/.test(t[m.index - 1]) : prevEndsSpace;
      let first = true;
      // One entry per rendered line the token sits on, from its characters.
      let run = "", box = null, i = m.index;
      const flush = () => {
        if (!box) return;
        const b = box;
        if (b.right - b.left >= 2 && b.right <= innerWidth && b.bottom <= innerHeight && b.top >= 0 && b.left >= 0)
          out.push({t: run, x: b.left, y: b.top, w: b.right - b.left, h: b.bottom - b.top, sp: first ? sp : false});
        first = false;
      };
      for (const ch of m[0]) {
        const cr = document.createRange();
        cr.setStart(n, i); cr.setEnd(n, i + ch.length);
        i += ch.length;
        const cb = cr.getClientRects()[0];
        if (!cb) continue;
        if (box && Math.abs(cb.top - box.top) < box.h / 2) {
          run += ch;
          box = {left: box.left, top: Math.min(box.top, cb.top), right: Math.max(box.right, cb.right),
                 bottom: Math.max(box.bottom, cb.bottom), h: box.h};
        } else {
          flush();
          run = ch;
          box = {left: cb.left, top: cb.top, right: cb.right, bottom: cb.bottom, h: cb.height};
        }
      }
      flush();
    }
    if (t.length) prevEndsSpace = /\s/.test(t[t.length - 1]) || getComputedStyle(p).display !== "inline";
  }
  return JSON.stringify(out);
};
"""


def esc(s):
    return html.escape(s, quote=False)


class Page:
    def __init__(self, rng, family):
        self.rng, self.family = rng, family
        self.charset, self.langs, self.line, folders = FAMILIES[family]
        self.body_fonts = [os.path.splitext(os.path.basename(f))[0] for d in folders["train"]
                           for f in sorted(glob.glob(os.path.join(TRAIN, "fonts", d, "*.ttf")))]
        # Display faces exist for Latin and CJK. Other scripts set headlines
        # in their own fonts at heavy weights.
        self.display = {"western": DISPLAY_LATIN, "cjk": DISPLAY_CJK + DISPLAY_LATIN[:3]}.get(family, self.body_fonts)

    def text(self, n=None):
        return self.line(self.rng, self.langs, self.charset, "train", n)

    def short(self):
        return self.text(self.rng.choice([1, 1, 2, 2, 3]))

    def number(self):
        r = self.rng
        return r.choice([f"{r.randint(2, 90)}%", f"{r.randint(1, 50)},{r.randint(0, 999):03d}", str(r.randint(2, 99)),
                         f"{r.randint(1, 99)}.{r.randint(0, 99):02d}", f"{r.randint(0, 59)}:{r.randint(0, 59):02d}",
                         f"${r.randint(5, 999)}", f"{r.randint(1, 9)}x"])

    def font(self, display=False):
        f = self.rng.choice(self.display if display else self.body_fonts)
        # Open fonts close every stack, so no glyph falls through to a system
        # font, which training never uses (pagecrops drops what they lack).
        return ",".join(f'"{x}"' for x in [f] + FALLBACK) + ",sans-serif"

    def color(self, light=None):
        r = self.rng
        light = r.random() < 0.5 if light is None else light
        lo, hi = (170, 255) if light else (0, 90)
        return "#%02x%02x%02x" % tuple(r.randint(lo, hi) for _ in range(3))

    def vivid(self):
        r = self.rng
        h, s, li = r.randint(0, 359), r.randint(55, 95), r.randint(35, 60)
        return f"hsl({h},{s}%,{li}%)"

    def background(self):
        r = self.rng
        kind = r.choice(["linear", "radial", "texture", "texture", "photo"])
        cs = [self.vivid() if r.random() < 0.6 else self.color() for _ in range(r.randint(2, 4))]
        if kind == "linear":
            return f"linear-gradient({r.randint(0, 359)}deg,{','.join(cs)})"
        if kind == "radial":
            layers = [f"radial-gradient(circle at {r.randint(0, 100)}% {r.randint(0, 100)}%,{c} 0,transparent {r.randint(30, 70)}%)"
                      for c in cs]
            return ",".join(layers) + f",{self.color()}"
        freq = r.uniform(0.004, 0.04) if kind == "texture" else r.uniform(0.002, 0.012)
        oct_ = r.randint(2, 5)
        m = " ".join(f"{r.uniform(-0.6, 1.2):.2f}" for _ in range(12)) + " 0 0 0 0 1"
        blobs = "".join(f"<circle cx='{r.randint(0, 600)}' cy='{r.randint(0, 400)}' r='{r.randint(40, 200)}' fill='{c}'/>"
                        for c in cs) if kind == "photo" else ""
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><filter id='f'>"
               f"<feTurbulence type='fractalNoise' baseFrequency='{freq:.4f}' numOctaves='{oct_}' seed='{r.randint(0, 9999)}'/>"
               f"<feColorMatrix values='{m}'/></filter><filter id='b'><feGaussianBlur stdDeviation='{r.randint(10, 40)}'/></filter>"
               f"<rect width='600' height='400' fill='{cs[0]}'/><g filter='url(#b)'>{blobs}</g>"
               f"<rect width='600' height='400' filter='url(#f)' opacity='{r.uniform(0.3, 0.9):.2f}'/></svg>")
        uri = "data:image/svg+xml;utf8," + svg.replace("#", "%23").replace("<", "%3C").replace(">", "%3E")
        return f"url(\"{uri}\") center/cover"

    def effects(self, dark_bg):
        r = self.rng
        fg = self.color(light=dark_bg) if r.random() < 0.85 else self.vivid()
        css = [f"color:{fg}"]
        x = r.random()
        if x < 0.3:
            css.append(f"text-shadow:0 {r.randint(0, 3)}px {r.randint(1, 8)}px rgba(0,0,0,{r.uniform(0.3, 0.9):.2f})")
        elif x < 0.42:
            css.append(f"-webkit-text-stroke:{r.uniform(0.5, 3):.1f}px {self.color(light=not dark_bg)}")
        elif x < 0.5:
            css.append(f"background:linear-gradient({r.randint(0, 180)}deg,{self.vivid()},{self.vivid()});"
                       "-webkit-background-clip:text;color:transparent")
        return ";".join(css)

    def hero(self):
        r = self.rng
        dark = r.random() < 0.6
        size, weight = r.randint(26, 72), r.choice([600, 700, 800, 900, 400])
        title = esc(self.text(r.randint(1, 5)))
        sub = esc(self.text(r.randint(3, 10)))
        cta = f"<span class=pill style='background:{self.vivid()};color:#fff;font-size:{r.randint(12, 18)}px'>{esc(self.short())}</span>"
        return (f"<section class=hero style='height:{r.randint(180, 340)}px;background:{self.background()};"
                f"text-align:{r.choice(['left', 'center'])};padding:{r.randint(20, 60)}px {r.randint(24, 80)}px'>"
                f"<div style='font:{weight} {size}px/{r.uniform(1.0, 1.25):.2f} {self.font(True)};{self.effects(dark)}'>{title}</div>"
                f"<div style='font:400 {r.randint(13, 22)}px {self.font()};margin:10px 0;{self.effects(dark)}'>{sub}</div>{cta}</section>")

    def thumbs(self):
        r = self.rng
        n, w = r.randint(3, 6), r.randint(160, 260)
        cells = []
        for _ in range(n):
            over = esc(self.text(r.randint(1, 4)))
            badge = (f"<span class=badge style='{r.choice(['right', 'left'])}:6px;bottom:6px'>{esc(self.number())}</span>"
                     if r.random() < 0.7 else "")
            big = r.random() < 0.4
            fs = r.randint(20, 40) if big else r.randint(12, 17)
            cells.append(
                f"<div class=thumb style='width:{w}px;height:{int(w * 0.56)}px;background:{self.background()}'>"
                f"<div class=ov style='font:{r.choice([700, 800, 900])} {fs}px/1.15 {self.font(big)};{self.effects(True)};"
                f"{'top' if big else 'bottom'}:{r.randint(6, 30)}px'>{over}</div>{badge}</div>"
                f"<div style='width:{w}px;font:500 {r.randint(12, 15)}px/1.3 {self.font()};margin:6px 0 14px'>"
                f"{esc(self.text(r.randint(3, 9)))}</div>")
        return "<section class=row>" + "".join(f"<div>{c}</div>" for c in cells) + "</section>"

    def promos(self):
        r = self.rng
        tiles = []
        for _ in range(r.randint(3, 5)):
            dark = r.random() < 0.7
            base = r.randint(14, 22)
            big = f"<span style='font-size:{r.uniform(1.8, 3.2):.1f}em'>{esc(self.number())}</span>"
            pre, post = esc(self.text(1)), esc(self.text(1))
            joined = r.random() < 0.6  # sizes meet with no space between them, as banners set them
            inner = f"{pre}{big}{post}" if joined else f"{pre} {big} {post}"
            tiles.append(f"<div class=tile style='background:{self.background()};width:{r.randint(180, 260)}px'>"
                         f"<div style='font:{r.choice([700, 800, 900])} {base}px/1.2 {self.font(True)};{self.effects(dark)}'>{inner}</div>"
                         f"<div style='font:500 {r.randint(11, 14)}px {self.font()};{self.effects(dark)};margin-top:6px'>{esc(self.text(r.randint(2, 5)))}</div></div>")
        return "<section class=row>" + "".join(tiles) + "</section>"

    def chips(self):
        r = self.rng
        out = []
        for _ in range(r.randint(5, 10)):
            filled = r.random() < 0.6
            bg = self.vivid() if filled else "transparent"
            out.append(f"<span class=pill style='background:{bg};border:1px solid {self.color()};"
                       f"color:{'#fff' if filled else self.color(False)};font:{r.choice([400, 500, 600])} {r.randint(9, 14)}px {self.font()}'>"
                       f"{esc(self.short())}</span>")
        return f"<section style='padding:10px 20px;background:{self.color()}'>{''.join(out)}</section>"

    def body(self):
        r = self.rng
        ps = "".join(f"<p>{esc(self.text(r.randint(8, 24)))}</p>" for _ in range(r.randint(2, 4)))
        head = f"<h2 style='font:{r.choice([600, 700, 800])} {r.randint(20, 34)}px {self.font(r.random() < 0.5)};margin:6px 0'>{esc(self.text(r.randint(2, 6)))}</h2>"
        return f"<section style='padding:12px 24px;font:400 {r.randint(11, 16)}px/1.4 {self.font()}'>{head}{ps}</section>"

    def nav(self):
        r = self.rng
        dark = r.random() < 0.5
        bg = self.color(light=not dark) if r.random() < 0.6 else self.vivid()
        items = "".join(f"<span style='margin-right:{r.randint(14, 30)}px'>{esc(self.short())}</span>" for _ in range(r.randint(4, 9)))
        logo = f"<b style='font:{r.choice([700, 900])} {r.randint(20, 32)}px {self.font(True)};margin-right:30px'>{esc(self.text(1))}</b>"
        return (f"<nav style='background:{bg};color:{self.color(light=dark)};padding:10px 20px;"
                f"font:500 {r.randint(12, 15)}px {self.font()}'>{logo}{items}</nav>")

    def scrim(self):
        r = self.rng
        a = r.uniform(0.35, 0.75)
        c = "0,0,0" if r.random() < 0.75 else "255,255,255"
        return (f"<div style='position:absolute;left:{r.randint(0, 600)}px;top:{r.randint(0, 400)}px;width:{r.randint(300, 1280)}px;"
                f"height:{r.randint(200, 800)}px;background:rgba({c},{a:.2f});pointer-events:none'></div>")

    def html(self, fonts):
        r = self.rng
        dark = r.random() < 0.4
        parts = [self.nav()] + [getattr(self, k)() for k in r.sample(["hero", "thumbs", "promos", "chips", "body", "body", "thumbs"], r.randint(3, 4))]
        if r.random() < 0.3:
            parts.append(self.scrim())
        face = "".join(f"@font-face{{font-family:'{os.path.splitext(f)[0]}';src:url('fonts/{f}');font-weight:100 900}}" for f in fonts)
        css = (face + "*{box-sizing:border-box}body{margin:0;width:1280px;height:800px;overflow:hidden;position:relative;"
               f"background:{self.color(light=not dark)};color:{self.color(light=dark)}}}"
               ".pill{display:inline-block;padding:4px 12px;border-radius:999px;margin:4px 6px 4px 0}"
               ".row{display:flex;gap:16px;padding:14px 20px;flex-wrap:wrap}"
               ".thumb{position:relative;border-radius:8px;overflow:hidden}"
               ".ov{position:absolute;left:10px;right:10px}"
               ".badge{position:absolute;background:rgba(0,0,0,.75);color:#fff;font:600 12px sans-serif;padding:2px 5px;border-radius:4px}"
               ".tile{border-radius:10px;padding:14px;min-height:110px}p{margin:6px 0}")
        rtl = " dir=rtl" if self.family == "rtl" else ""
        return (f"<!doctype html><html{rtl}><head><meta charset=utf-8><style>{css}</style><script>{JS}</script></head>"
                f"<body>{''.join(parts)}</body></html>")


def main():
    out, count = sys.argv[1], int(sys.argv[2])
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    families = sys.argv[4].split(",") if len(sys.argv) > 4 else ["western", "cjk"]
    if not set(families) <= set(ROUTED):
        sys.exit(f"families must be among {','.join(ROUTED)}")
    os.makedirs(os.path.join(out, "fonts"), exist_ok=True)
    # Every family's train fonts, which hold everything FALLBACK names.
    dirs = {d for f in FAMILIES.values() for d in f[3]["train"]} | {"display"}
    srcs = [s for d in sorted(dirs) for s in glob.glob(os.path.join(TRAIN, "fonts", d, "*.ttf"))]
    fonts = []
    for s in srcs:
        dst = os.path.join(out, "fonts", os.path.basename(s))
        if not os.path.exists(dst):
            os.link(s, dst)
        fonts.append(os.path.basename(s))
    names = []
    for i in range(count):
        family = rng.choice(families)
        name = f"{family}-{i:05d}"
        with open(os.path.join(out, name + ".html"), "w") as f:
            f.write(Page(rng, family).html(fonts))
        names.append(name)
    with open(os.path.join(out, "pages.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(count, "pages in", out)


if __name__ == "__main__":
    main()
