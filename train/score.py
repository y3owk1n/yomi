"""Scores predictions against evlines/labels.jsonl.

A prediction file is JSON lines {"id":..., "scale":1|2, "text":...}.
Word accuracy: ground-truth words reproduced exactly (case and punctuation
included) in order, by a longest-common-subsequence alignment of the line's
words. CER: character edit distance over ground-truth characters.

Usage: python -I score.py LABELS PRED [PRED...]
"""
import collections
import json
import re
import sys

_DICT = {w.strip().lower() for w in open("/usr/share/dict/words")}
_NATURAL = re.compile(
    r"^(\d+|\d+/\d+/\d+|\d+:\d\d(AM|PM)?|v\d+\.\d+\.\d+|[$#@%+-]\d+|(~/|/|\./)[a-z/._]+)$"
)


def natural(word):
    """A dictionary word (with edge punctuation), number, date, time, version
    or path, as opposed to the random strings the eval also holds."""
    core = word.strip(".,:;!?()\"'[")
    return core.lower() in _DICT or bool(_NATURAL.match(word))


def lcs(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            m[i][j] = m[i + 1][j + 1] + 1 if a[i] == b[j] else max(m[i + 1][j], m[i][j + 1])
    return m[0][0]


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(labels, preds):
    res = {}
    for scale in (1, 2):
        c = collections.defaultdict(collections.Counter)
        for lab in labels:
            p = preds.get((lab["id"], scale))
            if p is None:
                continue
            gt = lab["words"]
            pw = p.split()
            key = ("mono" if lab["mono"] else "sans", "<=12" if lab["size"] <= 12 else ">12")
            for k in (key, ("all", "")):
                c[k]["words"] += len(gt)
                c[k]["hit"] += lcs(gt, pw)
                c[k]["lines"] += 1
                c[k]["exact"] += " ".join(pw) == " ".join(gt)
                c[k]["chars"] += len(lab["text"])
                c[k]["edits"] += edit(lab["text"], " ".join(pw))
            nat = [w for w in gt if natural(w)]
            c["nat"]["words"] += len(nat)
            c["nat"]["hit"] += lcs(nat, [w for w in pw if w in nat])
        if c:
            res[scale] = c
    return res


labels = [json.loads(l) for l in open(sys.argv[1])]
for path in sys.argv[2:]:
    preds = {}
    for l in open(path):
        j = json.loads(l)
        preds[(j["id"], j["scale"])] = j["text"]
    for scale, c in score(labels, preds).items():
        a, nat = c[("all", "")], c.pop("nat")
        print(f"{path.split('/')[-1]:24} {scale}x  word {a['hit']/a['words']:.1%}  natural {nat['hit']/nat['words']:.1%}"
              f"  line {a['exact']/a['lines']:.1%}  CER {a['edits']/a['chars']:.2%}  | " + "  ".join(
                  f"{k[0]}{k[1]} {v['hit']/v['words']:.1%}" for k, v in sorted(c.items()) if k[0] != "all"))
