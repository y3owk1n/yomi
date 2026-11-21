"""Trains a recognizer on a pre-rendered set (prerender.py), validating on lines
rendered in the held-out fonts.

Usage: python train.py OUTDIR --family western --data data/western.pkl [--width 0.8 --steps 40000]
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import generator  # noqa: E402
from model import Recognizer  # noqa: E402

# (batch, width) shapes. MPS recompiles for every new input shape, which made
# varying shapes 20x slower per step, so every batch takes one of these.
BUCKETS = [(200, 128), (100, 256), (64, 384), (50, 512), (32, 768), (25, 1024), (16, 1536), (12, 2048)]


def bucket_of(width):
    return next((b for b in BUCKETS if width <= b[1]), None)


class Fixed:
    """Batches from a pre-rendered set in fixed shape buckets, reshuffled each pass."""

    def __init__(self, path, seed=0):
        with open(path, "rb") as f:
            self.data = [s for s in pickle.load(f) if s[0].shape[1] <= BUCKETS[-1][1]]
        self.rng = random.Random(seed)
        self.members = {b: [] for b in BUCKETS}
        for i, s in enumerate(self.data):
            self.members[bucket_of(s[0].shape[1])].append(i)

    def __iter__(self):
        while True:
            batches = []
            for (n, w), idx in self.members.items():
                self.rng.shuffle(idx)
                batches += [(w, idx[k:k + n]) for k in range(0, len(idx) - n + 1, n)]
            self.rng.shuffle(batches)
            for w, g in batches:
                yield collate([(self.data[i][0].astype(np.float32) / 255, self.data[i][1], self.data[i][2]) for i in g], w)


def collate(b, w):
    x = np.zeros((len(b), 1, 32, w), np.float32)
    for i, s in enumerate(b):
        x[i, 0, :, : s[0].shape[1]] = s[0]
    targets = torch.tensor([c for s in b for c in s[1]], dtype=torch.long)
    tlen = torch.tensor([len(s[1]) for s in b], dtype=torch.long)
    ilen = torch.tensor([s[0].shape[1] // 4 for s in b], dtype=torch.long)
    return torch.from_numpy(x), targets, ilen, tlen, [s[2] for s in b]


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


def validation(family, batch):
    """Held-out-font lines, grouped into the training buckets so evaluation
    compiles no new shapes."""
    gen, rng, val = generator(family, "heldout"), random.Random(99), []
    for _ in range(30):
        by = {}
        for s in (gen.sample(rng) for _ in range(batch)):
            b = bucket_of(s[0].shape[1])
            if b:
                by.setdefault(b, []).append(s)
        val += [collate(g, b[1]) for b, g in by.items()]
    return val, gen.alphabet.chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--family", required=True)
    ap.add_argument("--data", required=True, help="pre-rendered set from prerender.py")
    ap.add_argument("--width", type=float, default=0.8)
    ap.add_argument("--arch", default="ctx", choices=["base", "ctx"])
    ap.add_argument("--head", default="full", choices=["full", "gated"])
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(a.device)

    val, chars = validation(a.family, a.batch)
    classes = len(chars) + 1
    model = Recognizer(classes, a.width, a.arch, a.head, a.rank).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    meta = {"width": a.width, "arch": a.arch, "head": a.head, "rank": a.rank, "classes": classes,
            "alphabet": chars, "family": a.family}

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best, t0, run = 0.0, time.time(), 0.0
    for step, (x, tg, il, tl, _) in enumerate(Fixed(a.data), 1):
        model.train()
        logp = model(x.to(dev), il)
        loss = ctc(logp.permute(1, 0, 2).float().cpu(), tg, il, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
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
            state = dict(meta, model=model.state_dict(), step=step)
            torch.save(state, os.path.join(a.out, "last.pt"))
            if acc >= best:
                best = acc
                torch.save(state, os.path.join(a.out, "best.pt"))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
