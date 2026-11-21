"""Exports a trained recognizer for the Go forward pass: batch norm folded into
each convolution, weights as raw little-endian float32 in one file, and a JSON
manifest listing the layers in order.

Usage: python -I export.py CHECKPOINT OUTDIR
Writes OUTDIR/weights.bin, OUTDIR/model.json, and OUTDIR/check/ (a few
prepared crops with PyTorch logits, for the Go test).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LineContext, Recognizer  # noqa: E402
from synth import ALPHABET  # noqa: E402

ck, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "check"), exist_ok=True)
st = torch.load(ck, map_location="cpu")
m = Recognizer(st["classes"], st["width"], st.get("arch", "base")).eval()
m.load_state_dict(st["model"])

blob = bytearray()
layers = []


def put(arr):
    off = len(blob) // 4
    a = np.ascontiguousarray(arr, dtype="<f4")
    blob.extend(a.tobytes())
    return off


def conv(c, bn, relu):
    """A Conv2d, folding the BatchNorm2d that follows it when there is one."""
    w = c.weight.detach().double()
    b = c.bias.detach().double() if c.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        s = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * s[:, None, None, None]
        b = (b - bn.running_mean.double()) * s + bn.bias.detach().double()
    kh, kw = c.kernel_size
    layers.append({
        "op": "conv", "cin": c.in_channels, "cout": c.out_channels, "groups": c.groups,
        "kh": kh, "kw": kw, "sh": c.stride[0], "sw": c.stride[1], "ph": c.padding[0], "pw": c.padding[1],
        "dh": c.dilation[0], "dw": c.dilation[1], "relu": relu,
        "w": put(w.numpy()), "b": put(b.numpy()),
    })


def seq(s):
    """Conv, BN, ReLU triples in an nn.Sequential, nested or flat."""
    mods = [x for x in s.modules() if isinstance(x, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU))]
    i = 0
    while i < len(mods):
        c = mods[i]
        assert isinstance(c, nn.Conv2d), mods[i]
        bn = mods[i + 1] if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm2d) else None
        j = i + 1 + (bn is not None)
        relu = j < len(mods) and isinstance(mods[j], nn.ReLU)
        conv(c, bn, relu)
        i = j + relu


def line_context(lc: LineContext):
    w = lc.fc.weight.detach().double()[:, :, 0, 0]
    layers.append({"op": "linectx", "c": w.shape[0], "w": put(w.numpy()), "b": put(lc.fc.bias.detach().double().numpy())})


seq(m.features)
seq(m.collapse)
if m.arch == "ctx":
    line_context(m.line1)
    seq(m.context)
    seq(m.wide)
    line_context(m.line2)
else:
    seq(m.context)
seq(nn.Sequential(m.head))

with open(os.path.join(out, "weights.bin"), "wb") as f:
    f.write(blob)
json.dump({"height": 32, "stride": 4, "alphabet": ALPHABET, "layers": layers},
          open(os.path.join(out, "model.json"), "w"), indent=1)
print(f"{len(layers)} layers, {len(blob) / 1e6:.2f} MB")

# Check crops: prepared inputs and PyTorch logits (pre-softmax head output).
from PIL import Image  # noqa: E402
from prep import prepare  # noqa: E402

labs = [json.loads(l) for l in open("evcode/labels.jsonl")][:40]
with torch.no_grad():
    for lab in labs:
        x = prepare(np.asarray(Image.open(f"evcode/2x/{lab['id']}.png").convert("RGB")))
        feats = m.collapse(m.features(torch.from_numpy(x)[None, None]))
        if m.arch == "ctx":
            feats = m.line2(m.wide(m.context(m.line1(feats))))
        else:
            feats = m.context(feats)
        logits = m.head(feats)[0, :, 0, :].T.numpy()  # (T, C)
        base = os.path.join(out, "check", lab["id"])
        np.ascontiguousarray(x, dtype="<f4").tofile(base + ".in")
        np.ascontiguousarray(logits, dtype="<f4").tofile(base + ".logits")
        json.dump({"h": x.shape[0], "w": x.shape[1], "t": logits.shape[0], "c": logits.shape[1]},
                  open(base + ".json", "w"))
print(len(labs), "check crops")
