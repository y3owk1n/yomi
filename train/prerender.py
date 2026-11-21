"""Renders a fixed synthetic training set once, on every core, so training
steps do no rendering. Usage: python prerender.py FAMILY OUT.pkl COUNT [SEED]
Each sample is (uint8 ink map (32, W), label list, text).
"""
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import generator  # noqa: E402


def chunk(args):
    family, seed, n = args
    gen, rng = generator(family), random.Random(seed)
    out = []
    for _ in range(n):
        x, lab, text = gen.sample(rng)
        out.append((np.round(x * 255).astype(np.uint8), lab, text))
    return out


if __name__ == "__main__":
    family, path, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    base = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    per = 2000
    jobs = [(family, base * 100000 + i, per) for i in range((count + per - 1) // per)]
    t = time.time()
    data = []
    with mp.Pool(os.cpu_count()) as pool:
        for i, part in enumerate(pool.imap_unordered(chunk, jobs), 1):
            data += part
            if i % 20 == 0:
                print(f"{len(data)} samples, {len(data) / (time.time() - t):.0f}/s", flush=True)
    with open(path, "wb") as f:
        pickle.dump(data[:count], f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = sum(x.nbytes for x, _, _ in data[:count]) / 1e6
    print(f"wrote {count} samples ({mb:.0f} MB) to {path} in {time.time() - t:.0f}s")
