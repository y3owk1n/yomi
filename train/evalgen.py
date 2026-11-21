"""Writes evaluation pages: UI-like HTML in the macOS system font and SF Mono,
light and dark, 11 to 16 px, each page in one language. render.swift renders
them in WebKit and collects each word's exact box and text.

Sources:
  ui:fr,de,ru,...   interface strings of this Mac's apps (uistrings.py), never
                    used in training
  heldout:FAMILY    training-style lines over the held-out half of the words

Usage: python evalgen.py OUTDIR SOURCE [PAGES_PER_LANG] [SEED]
"""
import html
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILIES  # noqa: E402
from textsrc import line  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class UIStrings:
    def __init__(self, lang):
        self.strings = json.load(open(os.path.join(HERE, "data", "uistrings.json")))[lang]

    def phrase(self, rng, n):
        words = []
        while len(words) < n:
            words += rng.choice(self.strings).split()
        return " ".join(words[:n])

    def token(self, rng):
        return rng.choice([s for s in (rng.choice(self.strings) for _ in range(8)) if len(s.split()) <= 2] or ["OK"]).split()[0]


class HeldOut:
    def __init__(self, family, lang):
        charset, langs = FAMILIES[family]
        self.charset, self.langs = set(charset), {lang: 1}

    def phrase(self, rng, n):
        return line(rng, self.langs, self.charset, "heldout", n)

    def token(self, rng):
        return line(rng, self.langs, self.charset, "heldout", 1)


JS = r"""
window.__words = () => {
  const out = [];
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walk.nextNode())) {
    const p = n.parentElement;
    if (!p || p.closest("script,style")) continue;
    const cs = getComputedStyle(p);
    const t = n.textContent;
    const re = /\S+/g; let m;
    while ((m = re.exec(t))) {
      const r = document.createRange();
      r.setStart(n, m.index); r.setEnd(n, m.index + m[0].length);
      const rects = r.getClientRects();
      if (rects.length !== 1) continue;           // wrapped across lines
      const b = rects[0];
      if (b.width < 2 || b.right > innerWidth || b.bottom > innerHeight || b.top < 0 || b.left < 0) continue;
      out.push({t: m[0], x: b.left, y: b.top, w: b.width, h: b.height,
                size: parseFloat(cs.fontSize), mono: cs.fontFamily.includes("mono")});
    }
  }
  return JSON.stringify(out);
};
"""

CSS = """
*{box-sizing:border-box} body{margin:0;width:1280px;height:800px;overflow:hidden;
 font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
body{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5e7;--acc:#0a66d6;--card:#f5f5f7}
body.dark{--bg:#1e1e1e;--fg:#e8e8e8;--mut:#9a9a9a;--line:#333;--acc:#3a8ef6;--card:#2a2a2a}
.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace}
nav{position:absolute;left:0;top:0;bottom:0;width:230px;border-right:1px solid var(--line);padding:10px}
nav div{padding:5px 8px;border-radius:6px} nav div.on{background:var(--acc);color:#fff}
main{position:absolute;left:250px;top:0;right:0;bottom:0;padding:14px 20px}
.mut{color:var(--mut)} a{color:var(--acc);text-decoration:none}
.btn{display:inline-block;padding:4px 12px;border-radius:6px;border:1px solid var(--line);margin:4px 6px 4px 0}
.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.card{background:var(--card);border-radius:8px;padding:8px 12px;margin:8px 0}
table{border-collapse:collapse} td{padding:3px 12px 3px 0;border-bottom:1px solid var(--line);white-space:nowrap}
p{margin:6px 0;line-height:1.35}
"""


def esc(s):
    return html.escape(s, quote=False)


def page(rng, src, lang, size, dark):
    ph, tok = src.phrase, src.token
    nav = "".join(f'<div class="{"on" if i == 3 else ""}">{esc(ph(rng, rng.randint(1, 2)))}</div>'
                  for i in range(rng.randint(10, 22)))
    body = [f"<h3 style='margin:4px 0'>{esc(ph(rng, rng.randint(2, 4)))}</h3>"]
    for _ in range(rng.randint(2, 3)):
        body.append(f"<p>{esc(ph(rng, rng.randint(14, 30)))} <a href=#>{esc(ph(rng, 2))}</a> "
                    f"<span class=mut>{esc(ph(rng, rng.randint(4, 10)))}</span></p>")
    body.append("<div>" + "".join(f'<span class="btn {"pri" if i == 0 else ""}">{esc(ph(rng, rng.randint(1, 2)))}</span>'
                                  for i in range(rng.randint(3, 6))) + "</div>")
    rows = "".join("<tr>" + "".join(f"<td{' class=mono' if c == 2 else ''}>{esc(tok(rng))}</td>" for c in range(5)) + "</tr>"
                   for _ in range(rng.randint(5, 9)))
    body.append(f"<table>{rows}</table>")
    body.append("<div class='card mono'>" + "<br>".join(esc(ph(rng, rng.randint(3, 8))) for _ in range(rng.randint(3, 5))) + "</div>")
    body.append(f"<p class=mut>{esc(ph(rng, rng.randint(10, 20)))}</p>")
    cls = "dark" if dark else ""
    return (f"<!doctype html><html lang={lang}><head><meta charset=utf-8><style>{CSS} body{{font-size:{size}px}}</style>"
            f"<script>{JS}</script></head><body class='{cls}'><nav>{nav}</nav><main>{''.join(body)}</main></body></html>")


def main():
    out, source = sys.argv[1], sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261122)
    kind, arg = source.split(":", 1)
    os.makedirs(out, exist_ok=True)
    if kind == "ui":
        langs = arg.split(",")
        make = UIStrings
    else:
        family = arg
        langs = list(FAMILIES[family][1])
        make = lambda lang: HeldOut(family, lang)  # noqa: E731
    sizes = [11, 12, 13, 14, 16]
    names = []
    for lang in langs:
        src = make(lang)
        for i in range(per):
            size, dark = rng.choice(sizes), i % 2 == 1
            name = f"{lang}-{i}-{size}px{'-dark' if dark else ''}"
            with open(os.path.join(out, name + ".html"), "w") as f:
                f.write(page(rng, src, lang.split("_")[0], size, dark))
            names.append(name)
    with open(os.path.join(out, "pages.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(len(names), "pages in", out)


if __name__ == "__main__":
    main()
