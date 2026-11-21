"""Renders a fixed synthetic training set once, on every core, so training
steps do no rendering. Lines go to disk, not memory, so a set can be larger
than RAM. OUT.lines holds every line's uint8 ink map one after another, each
stored column by column (width x 32), so reading a line is one contiguous
read. OUT.npz holds the offsets, widths and labels, and OUT.txt the texts.

Usage: python prerender.py FAMILY OUT COUNT [SEED]   (PROCS caps the workers)
FAMILY "router" renders every routed family, each line labelled with its
family's index.

Each worker retires after four chunks, so the font faces it caches never
grow without bound.
"""
import array
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import generator, router_generator  # noqa: E402


def chunk(args):
    family, seed, n = args
    gen = router_generator() if family == "router" else generator(family)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x, lab, text = gen.sample(rng)
        out.append((np.round(x * 255).astype(np.uint8), lab, text))
    return out


class Set:
    """A pre-rendered set, read lazily from disk."""

    def __init__(self, path):
        meta = np.load(path + ".npz")
        self.offsets, self.widths = meta["offsets"], meta["widths"]
        self.label_offsets, self.labels = meta["label_offsets"], meta["labels"]
        self.lines = np.memmap(path + ".lines", dtype=np.uint8, mode="r").reshape(-1, 32)
        self.texts = open(path + ".txt", encoding="utf-8").read().split("\n")

    def __len__(self):
        return len(self.widths)

    def __getitem__(self, i):
        o, w = self.offsets[i], self.widths[i]
        return (np.asarray(self.lines[o:o + w]).T, self.labels[self.label_offsets[i]:self.label_offsets[i + 1]].tolist(),
                self.texts[i])


if __name__ == "__main__":
    family, path, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    base = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    per = 2000
    jobs = [(family, base * 100000 + i, per) for i in range((count + per - 1) // per)]
    t = time.time()
    # Labels run to tens of millions of entries, so they live in a compact
    # array, not a list of Python ints.
    offsets, widths, labels, label_offsets, texts = [], [], array.array("i"), [0], []
    total = 0
    with open(path + ".lines", "wb") as cols, mp.Pool(int(os.environ.get("PROCS", os.cpu_count())), maxtasksperchild=4) as pool:
        # Lines arrive in any order, and each is appended column by column.
        for i, part in enumerate(pool.imap_unordered(chunk, jobs), 1):
            for x, lab, text in part:
                if len(widths) == count:
                    break
                offsets.append(total)
                widths.append(x.shape[1])
                total += x.shape[1]
                cols.write(np.ascontiguousarray(x.T).tobytes())
                labels.extend(lab)
                label_offsets.append(len(labels))
                texts.append(text.replace("\n", " "))
            if i % 20 == 0:
                print(f"{len(widths)} samples, {len(widths) / (time.time() - t):.0f}/s", flush=True)
    np.savez(path + ".npz", offsets=np.array(offsets, np.int64), widths=np.array(widths, np.int32),
             labels=np.array(labels, np.int32), label_offsets=np.array(label_offsets, np.int64))
    open(path + ".txt", "w", encoding="utf-8").write("\n".join(texts))
    print(f"wrote {len(widths)} samples ({total * 32 / 1e6:.0f} MB) to {path} in {time.time() - t:.0f}s")
