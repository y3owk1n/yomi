"""Exports a trained recognizer for the Go forward pass: batch norm folded into
each convolution, weights as little-endian IEEE half floats in one file, and
a JSON manifest listing the layers in order. Half precision halves the files
and reads the evaluation pages as well as float32. The model is rounded in
place before the check crops run, so the references are the rounded model's.

Usage: python export.py CHECKPOINT OUTDIR [FAMILY]
Writes OUTDIR/weights.bin, OUTDIR/model.json, and OUTDIR/check/: eight
held-out-font lines of FAMILY (the checkpoint's own by default), prepared,
with the PyTorch logits the Go test compares against.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GatedHead, LineContext, Recognizer  # noqa: E402

ck, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "check"), exist_ok=True)
st = torch.load(ck, map_location="cpu")
hstride = st.get("hstride", 4)
m = Recognizer(st["classes"], st["width"], st.get("arch", "base"), st.get("head", "full"), st.get("rank", 128), hstride).eval()
m.load_state_dict(st["model"])

blob = bytearray()
layers = []


def put(arr):
    off = len(blob) // 2
    a = np.ascontiguousarray(arr, dtype="<f2")
    blob.extend(a.tobytes())
    return off


def half(t):
    """t rounded to half precision, in float64."""
    return t.float().half().double()


def store(mod, w, b):
    """Writes rounded weights back into mod, so the check crops run on them."""
    mod.weight.data = w.float().reshape(mod.weight.shape)
    mod.bias = nn.Parameter(b.float())


def conv(c, bn, relu):
    """A Conv2d, folding the BatchNorm2d that follows it when there is one."""
    w = c.weight.detach().double()
    b = c.bias.detach().double() if c.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        s = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * s[:, None, None, None]
        b = (b - bn.running_mean.double()) * s + bn.bias.detach().double()
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1 - bn.eps)
    w, b = half(w), half(b)
    store(c, w, b)
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
    w, b = half(lc.fc.weight.detach().double()), half(lc.fc.bias.detach().double())
    store(lc.fc, w, b)
    layers.append({"op": "linectx", "c": w.shape[0], "w": put(w[:, :, 0, 0].numpy()), "b": put(b.numpy())})


seq(m.features)
seq(m.collapse)
if m.arch == "ctx":
    line_context(m.line1)
    seq(m.context)
    seq(m.wide)
    line_context(m.line2)
else:
    seq(m.context)
if isinstance(m.head, GatedHead):
    h = m.head
    for c in (h.blank, h.proj, h.chars):
        store(c, half(c.weight.detach().double()), half(c.bias.detach().double()))
    flat = lambda c: c.weight.detach().double()[:, :, 0, 0].numpy()  # noqa: E731
    layers.append({
        "op": "gated", "c": h.blank.in_channels, "rank": h.proj.out_channels, "classes": h.chars.out_channels,
        "w": put(flat(h.blank)), "b": put(h.blank.bias.detach().double().numpy()),
        "proj_w": put(flat(h.proj)), "proj_b": put(h.proj.bias.detach().double().numpy()),
        "char_w": put(flat(h.chars)), "char_b": put(h.chars.bias.detach().double().numpy()),
    })
else:
    seq(nn.Sequential(m.head))

with open(os.path.join(out, "weights.bin"), "wb") as f:
    f.write(blob)
json.dump({"height": 32, "stride": hstride, "alphabet": st["alphabet"], "family": st["family"], "labels": st.get("labels"),
           "dtype": "f16", "layers": layers},
          open(os.path.join(out, "model.json"), "w"), indent=1)
print(f"{len(layers)} layers, {len(blob) / 1e6:.2f} MB")

# Check crops: prepared inputs and PyTorch logits (pre-softmax head output).
import random  # noqa: E402

from families import generator, router_generator  # noqa: E402

family = sys.argv[3] if len(sys.argv) > 3 else st["family"]
gen = router_generator("heldout") if family == "router" else generator(family, "heldout", blank_rate=0)
rng = random.Random(5)
with torch.no_grad():
    for n in range(8):
        x = gen.sample(rng)[0]
        base = os.path.join(out, "check", f"{n:05d}")
        np.ascontiguousarray(x, dtype="<f4").tofile(base + ".in")
        if isinstance(m.head, GatedHead):
            # The reference is each step's argmax class and its probability.
            logp = m(torch.from_numpy(x)[None, None])[0]
            prob, cls = logp.exp().max(-1)
            np.ascontiguousarray(cls.numpy(), dtype="<i4").tofile(base + ".classes")
            np.ascontiguousarray(prob.numpy(), dtype="<f4").tofile(base + ".probs")
            json.dump({"h": x.shape[0], "w": x.shape[1], "t": int(cls.shape[0]), "c": st["classes"]},
                      open(base + ".json", "w"))
            continue
        feats = m.collapse(m.features(torch.from_numpy(x)[None, None]))
        if m.arch == "ctx":
            feats = m.line2(m.wide(m.context(m.line1(feats))))
        else:
            feats = m.context(feats)
        logits = m.head(feats)[0, :, 0, :].T.numpy()  # (T, C)
        np.ascontiguousarray(logits, dtype="<f4").tofile(base + ".logits")
        json.dump({"h": x.shape[0], "w": x.shape[1], "t": logits.shape[0], "c": logits.shape[1]},
                  open(base + ".json", "w"))
print("8 check crops")
