"""Preview grids for the cloth-sim shape library.

    python src/cloth_preview.py data/cloth_test_shapes --out outputs/cloth_preview
-> shapes.png (bare white cloth, labelled with sim params) and flags.png (flags mapped on).
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from augment import make_background
from cloth_shapes import load_shape, map_flag, paste_shape
from config import CFG

PREVIEW_FLAGS = ["ie", "ci", "it", "fr", "nl", "de", "th", "in", "us", "es", "np", "br", "gr", "my", "jp", "sa"]


def load_flag(iso2):
    im = cv2.imread(str(CFG.paths.flags_png / f"{iso2}.png"), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)


def label(p):
    if not p:
        return ""
    wind = "no wind" if p["wind"] == 0 else f"wind {p['wind']:.0f}"
    return (f"{p['attach']} pole {p['pole_angle']:.0f}°\n{wind} {p['width']:.1f}m "
            f"st{p['stiffness']:.1f}{' BACK' if p['back'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes")
    ap.add_argument("--out", default=str(CFG.paths.outputs / "cloth_preview"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = sorted(Path(args.shapes).glob("*.npz"))
    shapes = [load_shape(p) for p in paths]
    params = [json.loads(str(s["params"])) for s in shapes]

    # 1) bare cloth: shows the simulated drape + HDRI lighting, on grey
    cols = 8
    rows = int(np.ceil(len(shapes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    white = np.full((8, 8, 4), 255, np.uint8)
    for ax, s, p in zip(axes.flat, shapes, params):
        layer, _ = map_flag(rng, s, white)
        bg = np.full(layer.shape[:2] + (3,), 0.35, np.float32)
        ax.imshow(bg * (1 - layer[..., 3:]) + layer[..., :3] * layer[..., 3:])
        ax.set_title(label(p), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "shapes.png", dpi=80)
    plt.close(fig)

    # 2) flags mapped onto random shapes, pasted on random backgrounds (training-style)
    bgs = sorted(CFG.paths.backgrounds.glob("*.jpg"))
    bg_loader = lambda r: cv2.imread(str(bgs[int(r.integers(len(bgs)))]))[..., ::-1]  # noqa: E731
    cols = 8
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.25), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, ax in enumerate(axes.flat[: args.n]):
        iso = PREVIEW_FLAGS[i % len(PREVIEW_FLAGS)]
        si = int(rng.integers(len(shapes)))
        upside = rng.random() < 0.05
        layer, k = map_flag(rng, shapes[si], load_flag(iso), upside_down=upside)
        bg = make_background(rng, bg_loader, CFG.aug)
        img, _ = paste_shape(rng, bg, layer, shapes[si], k, rng.uniform(0.15, 0.9), 12, 0.2)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{iso}  s{si}{' UPSIDE' if upside else ''}{' back' if params[si].get('back') else ''}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "flags.png", dpi=80)
    plt.close(fig)
    print(f"saved {out}/shapes.png and {out}/flags.png")


if __name__ == "__main__":
    main()
