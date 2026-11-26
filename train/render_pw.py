"""Renders each page in DIR/pages.txt in headless Chromium, on macOS, Linux or
Windows, as render.swift does with WebKit on a Mac: NAME@2x.png at two pixels
per CSS pixel, NAME@1x.png as its 2x2 box filter, and NAME.json with each
word's text and box in CSS pixels. A page whose fonts did not all load fails
the run, so no page is drawn in a font of the machine rendering it.

Usage: python render_pw.py DIR
"""
import json
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

W, H = 1280, 800


def main():
    d = os.path.abspath(sys.argv[1])
    with open(os.path.join(d, "pages.txt")) as f:
        names = f.read().split()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--allow-file-access-from-files", "--font-render-hinting=none"])
        page = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
        for name in names:
            page.goto("file://" + os.path.join(d, name + ".html"))
            page.evaluate("document.fonts.ready")
            failed = page.evaluate("[...document.fonts].filter(f => f.status === 'error').map(f => f.family)")
            if failed:
                sys.exit(f"{name}: fonts did not load: {failed}")
            words = page.evaluate("__words()")
            two = os.path.join(d, name + "@2x.png")
            page.screenshot(path=two)
            img = Image.open(two).convert("RGB")
            img.reduce(2).save(os.path.join(d, name + "@1x.png"))
            with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
                f.write(words)
            print(f"{name}: {img.width}x{img.height}, {len(json.loads(words))} words")
        browser.close()


if __name__ == "__main__":
    main()
