"""Render an 8x8 grid of random synthetic samples (with labels) for visual inspection.

Usage: python src/sample_grid.py [--seed N] [--out outputs/sample_grid.png]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CFG
from dataset import SyntheticFlags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(CFG.paths.outputs / "sample_grid.png"))
    ap.add_argument("--shapes", default=None, help="cloth-sim shapes dir (default: config)")
    args = ap.parse_args()

    ds = SyntheticFlags(samples_per_class=1, seed=args.seed, shapes_dir=args.shapes)
    idxs = np.random.default_rng(args.seed).choice(len(ds), args.n, replace=False)
    cols = int(np.ceil(np.sqrt(args.n)))
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    t0 = time.time()
    for ax, i in zip(axes.flat, idxs):
        img, cls = ds.render(int(i))
        ax.imshow(img)
        ax.set_title(f"{ds.labels[cls]['iso2']}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[args.n:]:
        ax.axis("off")
    dt = (time.time() - t0) / args.n * 1000
    plt.tight_layout()
    CFG.paths.outputs.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=80)
    print(f"saved {args.out}  ({dt:.1f} ms/sample, single process)")


if __name__ == "__main__":
    main()
