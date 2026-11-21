"""Trains the recognizer on synthetic lines, validating on held-out fonts.

Usage: python -I train.py OUTDIR --width 1.0 --steps 30000 [--workers 4]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Recognizer  # noqa: E402
from synth import ALPHABET, sample  # noqa: E402

CLASSES = len(ALPHABET) + 1


class Lines(torch.utils.data.IterableDataset):
    def __init__(self, seed, batch, split="train"):
        self.seed, self.batch, self.split = seed, batch, split

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        budget = self.batch * 400  # columns per batch, padded
        while True:
            pool = sorted((sample(rng, self.split) for _ in range(self.batch * 8)), key=lambda s: s[0].shape[1])
            batches, cur = [], []
            for s in pool:
                if cur and (len(cur) + 1) * s[0].shape[1] > budget:
                    batches.append(cur)
                    cur = []
                cur.append(s)
            batches.append(cur)
            rng.shuffle(batches)
            for b in batches:
                yield collate(b)


# (batch, width) shapes. MPS recompiles for every new input shape, which made
# varying shapes 20x slower per step, so every batch takes one of these.
BUCKETS = [(200, 128), (100, 256), (64, 384), (50, 512), (32, 768), (25, 1024), (16, 1536), (12, 2048)]


class Fixed:
    """Batches from a pre-rendered set in fixed shape buckets, reshuffled each pass."""

    def __init__(self, path, seed=0):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.data = [s for s in data if s[0].shape[1] <= BUCKETS[-1][1]]
        self.rng = random.Random(seed)
        self.members = {b: [] for b in BUCKETS}
        for i, s in enumerate(self.data):
            w = s[0].shape[1]
            self.members[next(b for b in BUCKETS if w <= b[1])].append(i)

    def __iter__(self):
        while True:
            batches = []
            for (n, w), idx in self.members.items():
                self.rng.shuffle(idx)
                batches += [((n, w), idx[k:k + n]) for k in range(0, len(idx) - n + 1, n)]
            self.rng.shuffle(batches)
            for (n, w), g in batches:
                yield collate([(self.data[i][0].astype(np.float32) / 255, self.data[i][1], self.data[i][2])
                               for i in g], w)


def collate(b, w=None):
    if w is None:
        w = max(s[0].shape[1] for s in b)
        w = (w + 3) // 4 * 4
    x = np.zeros((len(b), 1, 32, w), np.float32)
    for i, s in enumerate(b):
        x[i, 0, :, : s[0].shape[1]] = s[0]
    targets = torch.tensor([c for s in b for c in s[1]], dtype=torch.long)
    tlen = torch.tensor([len(s[1]) for s in b], dtype=torch.long)
    ilen = torch.tensor([s[0].shape[1] // 4 for s in b], dtype=torch.long)
    return torch.from_numpy(x), targets, ilen, tlen, [s[2] for s in b]


def decode(logp, ilen):
    out = []
    best = logp.argmax(-1).cpu().numpy()
    for row, n in zip(best, ilen.tolist()):
        prev, s = 0, []
        for k in row[:n]:
            if k != prev and k != 0:
                s.append(ALPHABET[k - 1])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--arch", default="base", choices=["base", "ctx"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--data", help="pre-rendered set from prerender.py; renders live when omitted")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(a.device)

    model = Recognizer(CLASSES, a.width, a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    rng = random.Random(99)
    val = []
    for _ in range(30):  # validation in the same shape buckets, so eval compiles nothing new
        vs = [sample(rng, "heldout") for _ in range(a.batch)]
        by = {}
        for s in vs:
            by.setdefault(next((b for b in BUCKETS if s[0].shape[1] <= b[1]), BUCKETS[-1]), []).append(s)
        val += [collate([s for s in g if s[0].shape[1] <= b[1]], b[1]) for b, g in by.items()]
    if a.data:
        loader = Fixed(a.data)
    else:
        loader = torch.utils.data.DataLoader(Lines(1, a.batch), batch_size=None, num_workers=a.workers,
                                             persistent_workers=True, prefetch_factor=8)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best, t0, run = 0.0, time.time(), 0.0
    for step, (x, tg, il, tl, _) in enumerate(loader, 1):
        model.train()
        logp = model(x.to(dev), il)
        loss = ctc(logp.permute(1, 0, 2).float().cpu(), tg, il, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            print(f"step {step} loss {run:.3f} {(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
        if step % 1000 == 0 or step == a.steps:
            model.eval()
            preds, texts = [], []
            with torch.no_grad():
                for vx, _, vil, _, vt in val:
                    preds += decode(model(vx.to(dev), vil), vil)
                    texts += vt
            acc = word_acc(preds, texts)
            rec = {"step": step, "loss": round(run, 4), "val_word": round(acc, 4),
                   "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "width": a.width, "arch": a.arch, "classes": CLASSES, "step": step},
                       os.path.join(a.out, "last.pt"))
            if acc >= best:
                best = acc
                torch.save({"model": model.state_dict(), "width": a.width, "arch": a.arch, "classes": CLASSES, "step": step},
                           os.path.join(a.out, "best.pt"))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
