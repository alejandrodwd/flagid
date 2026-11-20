"""Render the app's image assets from the flag SVGs in ../flagid (read-only).

Writes:
  FlagLog/Assets.xcassets/Flags/<code>.imageset   one PNG per flag SVG (namespace "Flags")
  FlagLog/Assets.xcassets/AppIcon.appiconset      a simple generated app icon
  TestImages/*.png                                photo-like test images for the simulator
  FlagLogTests/Resources/*.png                    the same test images, for the unit tests

Needs cairosvg and Pillow. The flagid virtualenv has both:
  ../flagid/.venv/bin/python -B scripts/render_assets.py

Only the public-domain flag SVGs are used. Nothing is read from flagid/data/real_val/.
"""
from __future__ import annotations

import io
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFilter

APP = Path(__file__).resolve().parent.parent
FLAG_SVGS = APP.parent / "flagid" / "data" / "flags" / "world"
CATALOG = APP / "FlagLog" / "Assets.xcassets"
FLAGS_DIR = CATALOG / "Flags"
TEST_IMAGES = APP / "TestImages"
TEST_RESOURCES = APP / "FlagLogTests" / "Resources"

THUMB_HEIGHT = 160  # px; the detail view shows flags at most ~120 pt tall

# Test images: (file stem, flag codes, placement). Ireland is included so the "ie/ci" group shows.
# "small" puts a small flag off-centre, to try tap-to-focus.
TEST_CASES = [
    ("ireland", "ie", "center"),
    ("france", "fr", "center"),
    ("japan", "jp", "center"),
    ("brazil", "br", "center"),
    ("germany", "de", "tilted"),
    ("canada", "ca", "tilted"),
    ("south_africa", "za", "center"),
    ("nepal", "np", "center"),
    ("mexico_small", "mx", "small"),
]


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def render_svg(svg: Path, height: int) -> Image.Image:
    png = cairosvg.svg2png(url=str(svg), output_height=height)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def render_flags() -> int:
    if FLAGS_DIR.exists():
        shutil.rmtree(FLAGS_DIR)
    FLAGS_DIR.mkdir(parents=True)
    write_json(FLAGS_DIR / "Contents.json",
               {"info": {"author": "xcode", "version": 1}, "properties": {"provides-namespace": True}})
    svgs = sorted(FLAG_SVGS.glob("*.svg"))
    for svg in svgs:
        code = svg.stem
        imageset = FLAGS_DIR / f"{code}.imageset"
        imageset.mkdir()
        img = render_svg(svg, THUMB_HEIGHT)
        img.save(imageset / f"{code}.png", optimize=True)
        write_json(imageset / "Contents.json", {
            "images": [{"filename": f"{code}.png", "idiom": "universal"}],
            "info": {"author": "xcode", "version": 1},
        })
    return len(svgs)


def render_app_icon() -> None:
    size = 1024
    icon = Image.new("RGB", (size, size), (24, 74, 140))
    d = ImageDraw.Draw(icon)
    # flag pole
    d.rounded_rectangle([250, 180, 292, 860], radius=20, fill=(235, 235, 240))
    # waving flag: three horizontal bands on a sine-shaped cloth
    bands = [(230, 60, 60), (245, 245, 245), (40, 160, 90)]
    top, h, left, w = 200, 390, 292, 500
    for x in range(w):
        dy = 22 * math.sin(x / w * 2 * math.pi)
        for i, colour in enumerate(bands):
            y0 = top + dy + i * h / 3
            d.line([(left + x, y0), (left + x, y0 + h / 3)], fill=colour)
    iconset = CATALOG / "AppIcon.appiconset"
    iconset.mkdir(parents=True, exist_ok=True)
    icon.save(iconset / "AppIcon.png")
    write_json(iconset / "Contents.json", {
        "images": [{"filename": "AppIcon.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
        "info": {"author": "xcode", "version": 1},
    })


def background(rng: random.Random, w: int, h: int) -> Image.Image:
    """Sky-like vertical gradient with a darker 'ground' band and a little noise."""
    top = (rng.randint(90, 140), rng.randint(150, 190), rng.randint(210, 245))
    bottom = (rng.randint(180, 220), rng.randint(200, 225), rng.randint(220, 240))
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    ground = int(h * rng.uniform(0.8, 0.9))
    d.rectangle([0, ground, w, h], fill=(rng.randint(60, 90), rng.randint(80, 110), rng.randint(50, 70)))
    noise = Image.effect_noise((w, h), 8).convert("RGB")
    return Image.blend(img, noise, 0.06)


def make_test_image(code: str, placement: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    w, h = 1200, 1600  # portrait, like a phone photo
    img = background(rng, w, h)
    short = min(w, h)
    flag_h = int(short * (0.14 if placement == "small" else 0.45))
    flag = render_svg(FLAG_SVGS / f"{code}.svg", flag_h)
    if placement == "tilted":
        flag = flag.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, expand=True)
    flag = flag.filter(ImageFilter.GaussianBlur(0.8))
    if placement == "small":
        x, y = int(w * 0.68), int(h * 0.22)
    else:
        x, y = (w - flag.width) // 2, int(h * 0.42) - flag.height // 2
    # flag pole on the left edge of the flag
    ImageDraw.Draw(img).rectangle([x - max(4, flag_h // 30), y - flag_h // 6, x, int(h * 0.88)],
                                  fill=(200, 200, 205))
    img.paste(flag, (x, y), flag)
    return img


def render_test_images() -> None:
    for d in (TEST_IMAGES, TEST_RESOURCES):
        d.mkdir(parents=True, exist_ok=True)
    for i, (stem, code, placement) in enumerate(TEST_CASES):
        img = make_test_image(code, placement, seed=1000 + i)
        img.save(TEST_IMAGES / f"{stem}.jpg", quality=90)
        img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(
            TEST_RESOURCES / f"{stem}.jpg", quality=90)


def main() -> None:
    if not FLAG_SVGS.is_dir():
        sys.exit(f"flag SVGs not found at {FLAG_SVGS}")
    n = render_flags()
    render_app_icon()
    render_test_images()
    print(f"{n} flag assets, app icon, {len(TEST_CASES)} test images")


if __name__ == "__main__":
    main()
