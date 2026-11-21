"""Reads the eval line crops with a trained checkpoint.

Usage: python -I predict.py CHECKPOINT LINESDIR OUT.jsonl
Prints CPU time per crop with batch size 1 and 1 thread, the shape the Go
port will have.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Recognizer  # noqa: E402
from prep import prepare  # noqa: E402
from train import decode  # noqa: E402

ck, d, out = sys.argv[1:4]
state = torch.load(ck, map_location="cpu")
model = Recognizer(state["classes"], state["width"], state.get("arch", "base"))
model.load_state_dict(state["model"])
model.eval()
torch.set_num_threads(1)
ids = [json.loads(l)["id"] for l in open(f"{d}/labels.jsonl")]
total, count = 0.0, 0
with open(out, "w") as f, torch.no_grad():
    for scale in (1, 2):
        for i in ids:
            if not os.path.exists(f"{d}/{scale}x/{i}.png"):
                continue
            x = prepare(np.asarray(Image.open(f"{d}/{scale}x/{i}.png").convert("RGB")))
            t = time.perf_counter()
            logp = model(torch.from_numpy(x)[None, None])
            total += time.perf_counter() - t
            count += 1
            text = decode(logp, torch.tensor([x.shape[1] // 4]))[0]
            f.write(json.dumps({"id": i, "scale": scale, "text": text}) + "\n")
print(f"{out}: {total / max(1, count) * 1000:.2f} ms/crop torch CPU, 1 thread, step {state['step']}")
