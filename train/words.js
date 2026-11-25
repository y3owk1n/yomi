
window.__words = () => {
  const out = [];
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walk.nextNode())) {
    const p = n.parentElement;
    if (!p || p.closest("script,style,noscript")) continue;
    const cs = getComputedStyle(p);
    if (cs.visibility === "hidden" || parseFloat(cs.opacity) < 0.3 || cs.color === "rgba(0, 0, 0, 0)") continue;
    const t = n.textContent;
    const re = /\S+/g; let m;
    while ((m = re.exec(t))) {
      const r = document.createRange();
      r.setStart(n, m.index); r.setEnd(n, m.index + m[0].length);
      const rects = r.getClientRects();
      // A token that wraps, as a CJK sentence with no spaces does, becomes a
      // word per line it sits on, from its characters' own boxes.
      const parts = [];
      if (rects.length === 1) {
        parts.push([m[0], rects[0]]);
      } else {
        let run = "", box = null, i = m.index;
        for (const ch of m[0]) {
          const cr = document.createRange();
          cr.setStart(n, i); cr.setEnd(n, i + ch.length);
          i += ch.length;
          const cb = cr.getClientRects()[0];
          if (!cb) continue;
          if (box && Math.abs(cb.top - box.top) < box.height / 2) {
            run += ch;
            box = {left: box.left, top: Math.min(box.top, cb.top), right: Math.max(box.right, cb.right),
                   bottom: Math.max(box.bottom, cb.bottom), height: box.height};
          } else {
            if (box) parts.push([run, box]);
            run = ch;
            box = {left: cb.left, top: cb.top, right: cb.right, bottom: cb.bottom, height: cb.height};
          }
        }
        if (box) parts.push([run, box]);
      }
      for (const [t2, b0] of parts) {
        const b = {left: b0.left, top: b0.top, right: b0.right, bottom: b0.bottom,
                   width: b0.right - b0.left, height: b0.bottom - b0.top};
        if (b.width < 2 || b.right > innerWidth || b.bottom > innerHeight || b.top < 0 || b.left < 0) continue;
        // Covered by something else (a banner, a modal): the topmost element at
        // the word's centre must hold the text node.
        const top = document.elementFromPoint((b.left + b.right) / 2, (b.top + b.bottom) / 2);
        if (!top || !(top === p || p.contains(top) || top.contains(p))) continue;
        out.push({t: t2, x: b.left, y: b.top, w: b.width, h: b.height,
                  size: parseFloat(cs.fontSize), mono: cs.fontFamily.includes("mono")});
      }
    }
  }
  return JSON.stringify(out);
};
