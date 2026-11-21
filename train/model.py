"""A convolution-only CTC line recognizer, built from operations a Go forward
pass can hand-write: 3x3 conv, depthwise conv, 1x1 conv, batch norm (folded
at export), ReLU, and a final linear layer per time step.

Input (N, 1, 32, W) ink map. Output (N, T, C) log-probabilities, T = W // 4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dw_block(cin, cout, stride, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LineContext(nn.Module):
    """Adds what the whole line looks like to every time step: the mean of each
    channel over the line's valid steps, through a 1x1 layer and ReLU, added
    back. It gives a lone glyph a reference for the line's x-height and cap
    height, which case (o/O, w/W) and l/1/I need. A Go port needs one mean and
    one small matrix multiply."""

    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x, lengths=None):
        if lengths is None:
            m = x.mean(dim=3, keepdim=True)
        else:
            t = x.shape[3]
            mask = (torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None]).to(x.dtype)
            m = (x * mask[:, None, None, :]).sum(dim=3, keepdim=True) / lengths.to(x.device).clamp(min=1).to(x.dtype)[:, None, None, None]
        return x + torch.relu(self.fc(m))


class GatedHead(nn.Module):
    """A head for alphabets of thousands of characters. A one-output layer
    gives each step's blank logit b. Where a character is likely, a low-rank
    pair of layers gives the character logits z. The log-probabilities are
    log P(blank) = log sigmoid(b) and log P(c) = log sigmoid(-b) +
    log softmax(z)_c, so CTC trains it as usual, and inference computes z only
    at the steps where b < 0."""

    def __init__(self, cin, classes, rank):
        super().__init__()
        self.blank = nn.Conv2d(cin, 1, 1)
        self.proj = nn.Conv2d(cin, rank, 1)
        self.chars = nn.Conv2d(rank, classes - 1, 1)

    def forward(self, x):
        b = self.blank(x)
        z = self.chars(self.proj(x))
        return torch.cat([F.logsigmoid(b), F.logsigmoid(-b) + z.log_softmax(1)], 1)


class Recognizer(nn.Module):
    """hstride is the input columns per output step, 4 or 2. At 4, scripts
    with narrow letters or stacked marks get under two steps a character, and
    CTC drops letters, so those scripts read at 2. The weights are the same at
    either stride."""

    def __init__(self, classes, width=1.0, arch="base", head="full", rank=128, hstride=4):
        super().__init__()
        if hstride not in (2, 4):
            raise ValueError(f"hstride {hstride}, want 4 or 2")
        self.arch, self.head_kind, self.hstride = arch, head, hstride
        c = [max(8, int(round(v * width))) for v in (16, 32, 64, 64, 128, 128, 192)]
        self.features = nn.Sequential(
            nn.Conv2d(1, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),  # 16 x W/2
            dw_block(c[0], c[1], 1),                # 16 x W/2
            dw_block(c[1], c[2], 2 if hstride == 4 else (2, 1)),  # 8 x W/hstride
            dw_block(c[2], c[3], 1),                # 8 x W/hstride
            dw_block(c[3], c[4], (2, 1)),           # 4 x W/hstride
            dw_block(c[4], c[5], 1),                # 4 x W/hstride
        )
        # collapse the 4 remaining rows: depthwise 4x1, then 1x1
        self.collapse = nn.Sequential(
            nn.Conv2d(c[5], c[5], (4, 1), groups=c[5], bias=False), nn.BatchNorm2d(c[5]), nn.ReLU(inplace=True),
            nn.Conv2d(c[5], c[6], 1, bias=False), nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
        )
        # context along the line: depthwise 1x5, then 1x1
        self.context = nn.Sequential(
            nn.Conv2d(c[6], c[6], (1, 5), 1, (0, 2), groups=c[6], bias=False), nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
            nn.Conv2d(c[6], c[6], 1, bias=False), nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
        )
        if arch == "ctx":
            self.line1 = LineContext(c[6])
            # a second, dilated context block: depthwise 1x5 at dilation 3 reaches 13 steps
            self.wide = nn.Sequential(
                nn.Conv2d(c[6], c[6], (1, 5), 1, (0, 6), dilation=(1, 3), groups=c[6], bias=False),
                nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
                nn.Conv2d(c[6], c[6], 1, bias=False), nn.BatchNorm2d(c[6]), nn.ReLU(inplace=True),
            )
            self.line2 = LineContext(c[6])
        self.head = GatedHead(c[6], classes, rank) if head == "gated" else nn.Conv2d(c[6], classes, 1)

    def forward(self, x, lengths=None):
        x = self.collapse(self.features(x))
        if self.arch == "ctx":
            x = self.line1(x, lengths)
            x = self.context(x)
            x = self.line2(self.wide(x), lengths)
        else:
            x = self.context(x)
        out = self.head(x).squeeze(2).permute(0, 2, 1)
        return out if self.head_kind == "gated" else out.log_softmax(-1)


def macs(model, width):
    """Exact multiply-adds of one forward pass at input (1, 1, 32, width)."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            total += out.numel() * (m.in_channels // m.groups) * kh * kw

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 32, width))
    for h in hs:
        h.remove()
    return total


if __name__ == "__main__":
    for arch in ("base", "ctx"):
        for w in (0.8, 0.85, 1.0):
            m = Recognizer(96, w, arch)
            params = sum(p.numel() for p in m.parameters())
            print(f"{arch} width {w}: params {params/1e3:.0f}k ({params*4/1e6:.2f} MB f32), "
                  f"MAC per input px {macs(m, 400)/(32*400):.0f}")
