"""Reads evaluation pages with PaddleOCR (rapidocr on ONNX Runtime), for the
baseline and as the offline teacher. Training and evaluation only: nothing of
PaddleOCR ships. Writes the same JSON lines as `yomi bench`, so pagescore.py
scores both.

It reads each page with the recognizer for the page's language, PaddleOCR's
best case, since yomi has to pick its model itself. It leaves out pages in a
language PaddleOCR has no recognizer for.

Usage: uv run --group teacher python paddle.py EVALDIR > out.jsonl
"""
import json
import os
import sys
import time
import unicodedata

import cv2
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

V4, V5, V6 = OCRVersion.PPOCRV4, OCRVersion.PPOCRV5, OCRVersion.PPOCRV6
LATIN = (V5, LangRec.LATIN, ModelType.MOBILE)
DEVANAGARI = (V5, LangRec.DEVANAGARI, ModelType.MOBILE)
CHINESE = (V6, LangRec.CH, ModelType.SMALL)
MODELS = {
    **{lang: LATIN for lang in ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt_BR", "ro", "sv", "tr", "vi")},
    "ru": (V5, LangRec.ESLAV, ModelType.MOBILE), "uk": (V5, LangRec.ESLAV, ModelType.MOBILE),
    "el": (V5, LangRec.EL, ModelType.MOBILE),
    "zh_CN": CHINESE, "zh_TW": CHINESE, "ja": CHINESE, "ko": (V5, LangRec.KOREAN, ModelType.MOBILE),
    "ar": (V5, LangRec.ARABIC, ModelType.MOBILE),
    "hi": DEVANAGARI, "mr": DEVANAGARI, "ne": DEVANAGARI,
    "ta": (V5, LangRec.TA, ModelType.MOBILE), "te": (V5, LangRec.TE, ModelType.MOBILE),
    "kn": (V4, LangRec.KA, ModelType.MOBILE),
    "th": (V5, LangRec.TH, ModelType.MOBILE),
}
engines = {}


def engine(key):
    if key not in engines:
        ver, lang, kind = key
        engines[key] = RapidOCR(params={
            "Global.use_cls": False, "Global.log_level": "error", "Global.return_word_box": True,
            "Rec.ocr_version": ver, "Rec.lang_type": lang, "Rec.model_type": kind,
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        })
    return engines[key]


def logical(text):
    """Puts a right-to-left word in reading order. Word boxes carry such words
    in drawn order."""
    if any(unicodedata.bidirectional(c) in ("R", "AL") for c in text):
        return text[::-1]
    return text


def words(res, scale):
    out = []
    for line in res.word_results or ():
        for text, score, quad in line:
            text = logical(text)
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            x0, y0 = min(xs) / scale, min(ys) / scale
            out.append([text, round(x0, 1), round(y0, 1), round(max(xs) / scale - x0, 1),
                        round(max(ys) / scale - y0, 1), round(float(score), 3)])
    return out


d = sys.argv[1]
for page in open(os.path.join(d, "pages.txt")).read().split():
    key = MODELS.get(page.split("-")[0])
    if key is None:
        continue
    eng = engine(key)
    for scale in (1, 2):
        img = cv2.imread(os.path.join(d, f"{page}@{scale}x.png"))
        if img is None:
            continue
        eng(img)  # warm
        t = time.perf_counter()
        res = eng(img)
        ms = (time.perf_counter() - t) * 1000
        print(json.dumps({"page": page, "scale": scale, "first_ms": ms, "median_ms": ms, "words": words(res, scale)},
                         ensure_ascii=False), flush=True)
