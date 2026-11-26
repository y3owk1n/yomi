"""Loads an exported model (model.json and weights.bin, as models/FAMILY/
holds) back into a PyTorch Recognizer, so a run can fine-tune from the model
that ships. export.py folds each batch norm into the convolution before it,
so here the batch norms come back as identities that add the folded bias,
and the network gives the exported model's outputs. In training mode those
batch norms would renormalize every layer from batch statistics, so train.py
keeps them frozen for a model loaded this way.

Usage: python exported.py DIR   prints the train.py flags the model needs.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GatedHead, Recognizer  # noqa: E402


def _order(m):
    """m's parts in the order export.py writes their layers."""
    parts = [m.features, m.collapse]
    parts += [m.line1, m.context, m.wide, m.line2] if m.arch == "ctx" else [m.context]
    return parts + [m.head]


def _conv_shapes(m):
    return [tuple(c.weight.shape) for p in _order(m) for c in p.modules() if isinstance(c, nn.Conv2d)]


def load_exported(path):
    """The checkpoint train.py would have saved for the exported model in path,
    with frozen_bn set."""
    with open(os.path.join(path, "model.json")) as f:
        meta = json.load(f)
    if meta["family"] == "router":
        sys.exit(f"{path}: the router is a different network, trained by train_router.py")
    blob = np.fromfile(os.path.join(path, "weights.bin"), dtype="<f2").astype(np.float32)
    layers = meta["layers"]
    arch = "ctx" if any(l["op"] == "linectx" for l in layers) else "base"
    head = "gated" if layers[-1]["op"] == "gated" else "full"
    rank = layers[-1]["rank"] if head == "gated" else 128
    classes = len(meta["alphabet"]) + 1
    want = []
    for l in layers:
        if l["op"] == "conv":
            want.append((l["cout"], l["cin"] // l["groups"], l["kh"], l["kw"]))
        elif l["op"] == "linectx":
            want.append((l["c"], l["c"], 1, 1))
        else:
            want += [(1, l["c"], 1, 1), (l["rank"], l["c"], 1, 1), (l["classes"], l["rank"], 1, 1)]
    # The width is not recorded. It sets every layer's channel count, so the
    # one whose network has the exported layers' shapes is it.
    for k in range(10, 401, 5):
        m = Recognizer(classes, k / 100, arch, head, rank, meta["stride"])
        if _conv_shapes(m) == want:
            width = k / 100
            break
    else:
        sys.exit(f"{path}: no width from 0.10 to 4.00 gives its layer shapes")

    it = iter(layers)

    def take(off, shape):
        n = int(np.prod(shape))
        return torch.from_numpy(blob[off:off + n].copy()).reshape(shape)

    def seq(s):
        mods = [x for x in s.modules() if isinstance(x, (nn.Conv2d, nn.BatchNorm2d))]
        i = 0
        while i < len(mods):
            c = mods[i]
            bn = mods[i + 1] if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm2d) else None
            l = next(it)
            c.weight.data.copy_(take(l["w"], c.weight.shape))
            b = take(l["b"], (l["cout"],))
            if bn is None:
                c.bias.data.copy_(b)
            else:
                bn.weight.data.fill_(1)
                bn.bias.data.copy_(b)
                bn.running_mean.zero_()
                bn.running_var.fill_(1 - bn.eps)
            i += 1 + (bn is not None)

    def line_context(lc):
        l = next(it)
        lc.fc.weight.data.copy_(take(l["w"], lc.fc.weight.shape))
        lc.fc.bias.data.copy_(take(l["b"], (l["c"],)))

    seq(m.features)
    seq(m.collapse)
    if arch == "ctx":
        line_context(m.line1)
        seq(m.context)
        seq(m.wide)
        line_context(m.line2)
    else:
        seq(m.context)
    if isinstance(m.head, GatedHead):
        l, h = next(it), m.head
        for c, w, b in ((h.blank, "w", "b"), (h.proj, "proj_w", "proj_b"), (h.chars, "char_w", "char_b")):
            c.weight.data.copy_(take(l[w], c.weight.shape))
            c.bias.data.copy_(take(l[b], (c.weight.shape[0],)))
    else:
        seq(nn.Sequential(m.head))
    return {"model": m.state_dict(), "width": width, "arch": arch, "head": head, "rank": rank,
            "hstride": meta["stride"], "classes": classes, "alphabet": meta["alphabet"],
            "family": meta["family"], "labels": meta.get("labels"), "frozen_bn": True}


if __name__ == "__main__":
    ck = load_exported(sys.argv[1])
    flags = f"--family {ck['family']} --width {ck['width']:g} --arch {ck['arch']} --head {ck['head']} --hstride {ck['hstride']}"
    print(flags + (f" --rank {ck['rank']}" if ck["head"] == "gated" else ""))
