"""Downloads the open evaluation set into data/, on any platform: 127 pages
in 42 languages built from Firefox's interface strings (MPL-2.0) and drawn in
open fonts, published as a release of the yomi repository. evalall.sh scores
it with the other sets. NOTICE.md in data/evals-open-v1/ lists its sources
and licences.

Usage: python fetch_evals.py
"""
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request

NAME = "yomi-evals-open-v1.tar.gz"
URL = f"https://github.com/y3owk1n/yomi/releases/download/evals-open-v1/{NAME}"
SHA256 = "c0dfd108ae1648b21eb624dd6f866aba6676b449a251a45d6901789b3e9234f2"

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data, exist_ok=True)
path = os.path.join(data, NAME)
if not os.path.exists(path):
    print("downloading", URL)
    with urllib.request.urlopen(URL) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(path + ".part", path)
with open(path, "rb") as f:
    got = hashlib.sha256(f.read()).hexdigest()
if got != SHA256:
    os.remove(path)
    sys.exit(f"{NAME}: sha256 {got}, want {SHA256}. Removed it, run again.")
with tarfile.open(path) as t:
    # The hash above checked the archive. Python without extraction filters
    # (before 3.12) extracts it as it is.
    t.extractall(data, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
root = os.path.join(data, "evals-open-v1")
for d in sorted(os.listdir(root)):
    if d.startswith("eval-open-"):
        dst = os.path.join(data, d)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(root, d), dst)
        print(dst)
