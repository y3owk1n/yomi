"""Trains a recognizer on a pre-rendered set (prerender.py), validating on lines
rendered in the held-out fonts.

Usage: python train.py OUTDIR --family western --data data/western [--width 0.8 --steps 40000]

A fine-tune starts from a checkpoint (--init) and takes fewer steps at a lower
learning rate. It should train on the old set as well as the new one, since a
model trained on new lines alone forgets the old ones:
  python train.py runs/western-c --family western --init runs/western-b/best.pt \
      --data data/western-b-600k data/western-c-200k --steps 20000
"""
import argparse
import json
import os
import queue
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import generator  # noqa: E402
from model import Recognizer  # noqa: E402
from prerender import Set  # noqa: E402

# (batch, width) shapes. MPS recompiles for every new input shape, which made
# varying shapes 20x slower per step, so every batch takes one of these.
BUCKETS = [(200, 128), (100, 256), (64, 384), (50, 512), (32, 768), (25, 1024), (16, 1536), (12, 2048)]


def bucket_of(width):
    return next((b for b in BUCKETS if width <= b[1]), None)


class Fixed:
    """Batches from one or more pre-rendered sets (prerender.Set) in fixed
    shape buckets, reshuffled each pass. Lines of all the sets mix within a
    batch."""

    def __init__(self, paths, seed=0, hstride=4):
        sets = [Set(p) for p in ([paths] if isinstance(paths, str) else paths)]
        self.rng = random.Random(seed)
        self.hstride = hstride
        self.members = {b: [] for b in BUCKETS}
        self.lines = []
        for d in sets:
            for i, w in enumerate(d.widths):
                if w <= BUCKETS[-1][1]:
                    self.members[bucket_of(w)].append(len(self.lines))
                    self.lines.append((d, i))

    def __iter__(self):
        while True:
            batches = []
            for (n, w), idx in self.members.items():
                self.rng.shuffle(idx)
                batches += [(w, idx[k:k + n]) for k in range(0, len(idx) - n + 1, n)]
            self.rng.shuffle(batches)
            for w, g in batches:
                rows = [d[i] for d, i in (self.lines[k] for k in g)]
                yield collate([(x.astype(np.float32) / 255, lab, text) for x, lab, text in rows], w, self.hstride)


def prefetched(batches, depth=8):
    """Yields from batches while a background thread reads ahead, so loading
    the next batches overlaps the GPU's work on this one."""
    q = queue.Queue(depth)
    threading.Thread(target=lambda: [q.put(b) for b in batches], daemon=True).start()
    while True:
        yield q.get()


def collate(b, w, hstride=4):
    x = np.zeros((len(b), 1, 32, w), np.float32)
    for i, s in enumerate(b):
        x[i, 0, :, : s[0].shape[1]] = s[0]
    targets = torch.tensor([c for s in b for c in s[1]], dtype=torch.long)
    tlen = torch.tensor([len(s[1]) for s in b], dtype=torch.long)
    ilen = torch.tensor([s[0].shape[1] // hstride for s in b], dtype=torch.long)
    return torch.from_numpy(x), targets, ilen, tlen, [s[2] for s in b]


def ctc_loss(logp, targets, ilen, tlen, dev):
    """Mean CTC loss over the batch, each line's divided by its label length.

    CTC reads only the blank column and the columns of the characters in the
    batch's targets, so only those cross to the CPU, where CTC runs (MPS has
    none). PyTorch's CTC backward also adds exp(log p) to every column it is
    given, which cancels only when those columns are a whole log-softmax. The
    correction term below removes it, so the gradient is exact for any subset
    of columns and for the gated head."""
    cols, inv = torch.unique(targets, return_inverse=True)
    cols = torch.cat([torch.zeros(1, dtype=cols.dtype), cols])
    lp = logp.index_select(2, cols.to(dev)).permute(1, 0, 2).float().cpu()  # (T, N, K)
    nll = F.ctc_loss(lp, inv + 1, ilen, tlen, blank=0, reduction="none", zero_infinity=True)
    w = 1 / (tlen.clamp(min=1).float() * len(tlen))
    valid = (torch.arange(lp.shape[0])[:, None] < ilen[None, :]).float()[..., None]  # (T, N, 1)
    corr = ((lp.detach().exp() * lp * valid).sum((0, 2)) * w).sum()
    return (nll * w).sum() - corr + corr.detach()


def decode(logp, ilen, chars):
    out = []
    for row, n in zip(logp.argmax(-1).cpu().numpy(), ilen.tolist()):
        prev, s = 0, []
        for k in row[:n]:
            if k != prev and k != 0:
                s.append(chars[k - 1])
            prev = k
        out.append("".join(s))
    return out


def word_acc(preds, texts):
    hit = tot = 0
    for p, t in zip(preds, texts):
        tw, pw = t.split(), p.split()
        tot += len(tw)
        hit += sum(a == b for a, b in zip(tw, pw)) if len(tw) == len(pw) else 0
    return hit / max(1, tot)


def validation(family, batch, hstride):
    """Held-out-font lines, grouped into the training buckets so evaluation
    compiles no new shapes."""
    gen, rng, val = generator(family, "heldout"), random.Random(99), []
    for _ in range(30):
        by = {}
        for s in (gen.sample(rng) for _ in range(batch)):
            b = bucket_of(s[0].shape[1])
            if b:
                by.setdefault(b, []).append(s)
        val += [collate(g, b[1], hstride) for b, g in by.items()]
    return val, gen.alphabet.chars


def init_from(model, path, meta):
    """Loads the checkpoint at path into model. Its network must be the same
    shape. Characters of the new alphabet that the checkpoint knows keep their
    output rows, and new ones start untrained. Returns how many were kept."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("width", "arch", "head"):
        if ck[k] != meta[k]:
            sys.exit(f"--init {path}: its {k} is {ck[k]}, this run's is {meta[k]}")
    if meta["head"] == "gated" and ck["rank"] != meta["rank"]:
        sys.exit(f"--init {path}: its rank is {ck['rank']}, this run's is {meta['rank']}")
    # Output rows go by character. The full head's row 0 is blank, and the
    # gated head's character layer has no blank row.
    rows, off = (("head.chars.weight", "head.chars.bias"), 0) if meta["head"] == "gated" else (("head.weight", "head.bias"), 1)
    old, new = ck["model"], model.state_dict()
    known = {c: i for i, c in enumerate(ck["alphabet"])}
    kept = [(i, known[c]) for i, c in enumerate(meta["alphabet"]) if c in known]
    for k, v in new.items():
        if k not in rows:
            v.copy_(old[k])
            continue
        if off:
            v[0] = old[k][0]
        for i, j in kept:
            v[i + off] = old[k][j + off]
    model.load_state_dict(new)
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--family", required=True)
    ap.add_argument("--data", required=True, nargs="+", help="pre-rendered sets from prerender.py")
    ap.add_argument("--init", help="checkpoint to fine-tune from")
    ap.add_argument("--resume", action="store_true", help="continue OUTDIR's run from its last.pt")
    ap.add_argument("--width", type=float, default=0.8)
    ap.add_argument("--arch", default="ctx", choices=["base", "ctx"])
    ap.add_argument("--head", default="full", choices=["full", "gated"])
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--hstride", type=int, default=4, choices=[4, 2], help="input columns per output step")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, help="peak learning rate, 3e-3 from scratch and 1e-3 with --init")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(a.device)

    val, chars = validation(a.family, a.batch, a.hstride)
    classes = len(chars) + 1
    model = Recognizer(classes, a.width, a.arch, a.head, a.rank, a.hstride)
    meta = {"width": a.width, "arch": a.arch, "head": a.head, "rank": a.rank, "hstride": a.hstride, "classes": classes,
            "alphabet": chars, "family": a.family}
    if a.init:
        kept = init_from(model, a.init, meta)
        print(json.dumps({"init": a.init, "chars_kept": kept, "chars": len(chars)}), flush=True)
    model.to(dev)
    lr = a.lr or (1e-3 if a.init else 3e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.steps, pct_start=0.05)
    start, best = 0, 0.0
    if a.resume:
        # A resume takes the run's own arguments. The weights, the step and
        # the schedule's place come back. A checkpoint from before the
        # optimizer was saved restarts its moments and steps the schedule
        # forward to the same place.
        ck = torch.load(os.path.join(a.out, "last.pt"), map_location="cpu", weights_only=False)
        init_from(model, os.path.join(a.out, "last.pt"), meta)
        start = ck["step"]
        if start >= a.steps:
            sys.exit(f"--resume: {a.out} already ran {start} of {a.steps} steps")
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        else:
            for _ in range(start):
                sched.step()
        with open(os.path.join(a.out, "log.jsonl")) as f:
            best = max((json.loads(line)["val_word"] for line in f if "val_word" in line), default=0.0)
        print(json.dumps({"resume": start, "best": best}), flush=True)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, run = time.time(), 0.0
    for step, (x, tg, il, tl, _) in enumerate(prefetched(Fixed(a.data, seed=start, hstride=a.hstride)), start + 1):
        model.train()
        logp = model(x.to(dev), il)
        loss = ctc_loss(logp, tg, il, tl, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > start + 1 else loss.item()
        if step % 1000 == 0 or step == a.steps:
            model.eval()
            preds, texts = [], []
            with torch.no_grad():
                for vx, _, vil, _, vt in val:
                    preds += decode(model(vx.to(dev), vil), vil, chars)
                    texts += vt
            acc = word_acc(preds, texts)
            rec = {"step": step, "loss": round(run, 4), "val_word": round(acc, 4),
                   "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            state = dict(meta, model=model.state_dict(), step=step, opt=opt.state_dict(), sched=sched.state_dict())
            torch.save(state, os.path.join(a.out, "last.pt"))
            if acc >= best:
                best = acc
                torch.save(state, os.path.join(a.out, "best.pt"))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
