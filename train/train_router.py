"""Trains the router: the recognizer's network with one class per routed family
plus none, trained so every time step of a line names the line's family. The
engine averages the steps' log-probabilities and picks the best family, which
needs no operation the recognizers do not already use.

Usage: python train_router.py OUTDIR --data data/router [--width 0.5 --steps 15000]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import ROUTED  # noqa: E402
from model import Recognizer  # noqa: E402
from train import Fixed  # noqa: E402

LABELS = ROUTED + ["none"]


def line_scores(logp, ilen):
    """Mean log-probability per class over each line's valid steps, (N, C)."""
    t = logp.shape[1]
    mask = (torch.arange(t, device=logp.device)[None, :] < ilen.to(logp.device)[:, None]).float()
    return (logp * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    model = Recognizer(len(LABELS), a.width, "ctx").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    meta = {"width": a.width, "arch": "ctx", "head": "full", "classes": len(LABELS), "alphabet": "",
            "labels": LABELS, "family": "router"}
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, run, hits, seen = time.time(), 0.0, 0, 0
    for step, (x, tg, il, tl, _) in enumerate(Fixed(a.data), 1):
        model.train()
        logp = model(x.to(dev), il)  # (N, T, C)
        y = tg.to(dev)  # one label per line
        scores = line_scores(logp, il)
        # Every valid step learns the line's family.
        t = logp.shape[1]
        mask = (torch.arange(t, device=dev)[None, :] < il.to(dev)[:, None]).float()
        nll = -(logp.gather(2, y[:, None, None].expand(-1, t, 1)).squeeze(2) * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * nll.item() if step > 1 else nll.item()
        hits += (scores.argmax(1) == y).sum().item()
        seen += len(y)
        if step % 1000 == 0 or step == a.steps:
            rec = {"step": step, "loss": round(run, 4), "train_line_acc": round(hits / seen, 4),
                   "min": round((time.time() - t0) / 60, 1)}
            hits = seen = 0
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save(dict(meta, model=model.state_dict(), step=step), os.path.join(a.out, "last.pt"))
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
