"""Stage 3: evaluate a checkpoint on real photos (data/real_val/<iso2>/), or on a synthetic
split with a different seed if no real photos exist (overestimates real performance!).

Headline metric is top-4 (the app shows a live top-4 list and the user picks the flag).

Usage:
    python src/evaluate.py [--ckpt outputs/<run>/best.pt] [--synthetic]
Writes outputs/<run>/eval_<source>/: report.json, predictions.csv, errors.png (flags missing from top-4)
"""
import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG, IMAGENET_MEAN, IMAGENET_STD
from dataset import RealPhotos, SyntheticFlags, center_crop_resize, lookalike_groups, to_tensor
from model import get_device, load_checkpoint

TOPKS = (1, 3, 4, 5, 10)
HEADLINE_K = 4
MAX_ERROR_TILES = 48
CONF_THRESHOLDS = (0.3, 0.5, 0.7)  # for the no-flag-photo confidence report


@torch.no_grad()
def predict(model, ds, device):
    """Returns probs [N, C] (numpy) and labels [N]."""
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    probs, labels = [], []
    for x, y in dl:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def predict_files(model, files, device):
    xs = [to_tensor(center_crop_resize(cv2.imread(str(f), cv2.IMREAD_COLOR)[..., ::-1], CFG.aug.out_size))
          for f in files]
    if not xs:
        return np.zeros((0, 0))
    return torch.softmax(model(torch.stack(xs).to(device)), 1).cpu().numpy()


def summarize(probs, labels, class_names):
    rank = np.argsort(-probs, 1)
    pos = np.argmax(rank == labels[:, None], 1)        # 0-based rank of the true class
    conf = probs.max(1)
    correct = pos == 0
    per_class = {}
    for c in sorted(set(labels.tolist())):
        m = labels == c
        per_class[class_names[c]] = {"n": int(m.sum()), "top1": float(correct[m].mean()),
                                     f"top{HEADLINE_K}": float((pos[m] < HEADLINE_K).mean()),
                                     "median_rank": float(np.median(pos[m]) + 1)}
    pairs = Counter((class_names[t], class_names[p]) for t, p in zip(labels, rank[:, 0]) if t != p)
    return {
        "n": int(len(labels)),
        "num_classes_present": len(per_class),
        **{f"top{k}": float((pos < k).mean()) for k in TOPKS},
        "calibration": {
            "mean_conf_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
        },
        "per_class": per_class,
        "confused_pairs_top1": [{"true": t, "pred": p, "count": n} for (t, p), n in pairs.most_common(20)],
    }


def no_flag_report(model, device, class_names, flag_conf):
    """For a model without a no_flag class: how confident is it on real no-flag photos?
    (Useful for the app: e.g. only show the top-4 list when confidence is above some level.)"""
    d = CFG.paths.real_val / CFG.data.no_flag_label
    if CFG.data.no_flag_label in class_names or not d.is_dir():
        return None
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in RealPhotos.EXTS)
    p = predict_files(model, files, device)
    if len(p) == 0:
        return None
    c = p.max(1)
    return {"n": len(files), "mean_top1_conf": float(c.mean()), "mean_top1_conf_on_flag_photos": float(flag_conf.mean()),
            "share_above": {str(t): float((c >= t).mean()) for t in CONF_THRESHOLDS},
            "flag_photos_share_above": {str(t): float((flag_conf >= t).mean()) for t in CONF_THRESHOLDS}}


def denorm(x):
    img = x.numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def save_error_grid(ds, probs, labels, class_names, path):
    """Images whose true flag is NOT in the top-4 (the failures that matter for the app)."""
    rank = np.argsort(-probs, 1)
    miss = np.where(~(rank[:, :HEADLINE_K] == labels[:, None]).any(1))[0][:MAX_ERROR_TILES]
    if len(miss) == 0:
        return False
    cols = min(8, len(miss))
    rows = int(np.ceil(len(miss) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, miss):
        ax.imshow(denorm(ds[int(i)][0]))
        top = " ".join(class_names[k] for k in rank[i, :HEADLINE_K])
        ax.set_title(f"{class_names[labels[i]]}: {top}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return True


def print_report(r, source):
    print(f"\n=== {source}: {r['n']} flag photos, {r['num_classes_present']} countries present "
          f"(model ranks all classes) ===")
    print("  " + "   ".join(f"top-{k}: {r[f'top{k}']:.1%}" for k in TOPKS) + f"   <- headline: top-{HEADLINE_K}")
    g = r.get("grouped")
    if g:
        print("  " + "   ".join(f"top-{k}: {g[f'top{k}']:.1%}" for k in TOPKS)
              + f"   <- lookalike groups as one entry ({', '.join(g['groups'])})")
    c = r["calibration"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"  mean top-1 confidence  correct: {fmt(c['mean_conf_correct'])}   wrong: {fmt(c['mean_conf_wrong'])}")
    print(f"\nper-country (worst top-{HEADLINE_K} first):  top-{HEADLINE_K}   top-1   median rank")
    for name, v in sorted(r["per_class"].items(), key=lambda kv: (kv[1][f"top{HEADLINE_K}"], kv[1]["top1"])):
        print(f"  {name:8s}                {v[f'top{HEADLINE_K}']:6.1%}  {v['top1']:6.1%}   {v['median_rank']:5.1f}   (n={v['n']})")
    print("\nmost confused at top-1 (true -> predicted):")
    for p in r["confused_pairs_top1"][:12]:
        print(f"  {p['true']:8s} -> {p['pred']:8s} x{p['count']}")
    nf = r.get("no_flag_photos")
    if nf:
        print(f"\nno-flag photos ({nf['n']}): mean top-1 confidence {nf['mean_top1_conf']:.3f} "
              f"(flag photos: {nf['mean_top1_conf_on_flag_photos']:.3f})")
        for t in CONF_THRESHOLDS:
            print(f"  share with top-1 conf >= {t}:  no-flag {nf['share_above'][str(t)]:6.1%}   "
                  f"flags {nf['flag_photos_share_above'][str(t)]:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.paths.outputs / CFG.train.backbone / "best.pt"))
    ap.add_argument("--synthetic", action="store_true", help="force synthetic split")
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    class_names = [l["iso2"] for l in ckpt["labels"]]

    ds = None if args.synthetic else RealPhotos()
    if ds is not None and len(ds) > 0:
        source = "real_val"
        # flag photos only (no_flag photos are reported separately for models without that class)
        nf_idx = class_names.index(CFG.data.no_flag_label) if CFG.data.no_flag_label in class_names else None
        ds.items = [(f, c) for f, c in ds.items if c != nf_idx]
    else:
        source = "synthetic"
        print("\n" + "!" * 78 + "\nWARNING: no real photos in data/real_val - evaluating on a SYNTHETIC split "
              "(seed differs from\ntraining). This heavily OVERESTIMATES real-world accuracy.\n" + "!" * 78)
        ds = SyntheticFlags(CFG.train.val_samples_per_class, seed=CFG.train.val_seed + 1)

    probs, labels = predict(model, ds, device)
    if CFG.data.no_flag_label in class_names:  # older models: rank among countries only
        probs = probs.copy()
        probs[:, class_names.index(CFG.data.no_flag_label)] = -1
    report = summarize(probs, labels, class_names)
    if source == "real_val":
        report["no_flag_photos"] = no_flag_report(model, device, class_names, probs.max(1))
    report.update(source=source, checkpoint=str(args.ckpt), checkpoint_epoch=ckpt["epoch"])
    # lookalike groups (CFG.data.lookalike_groups) counted as one entry, as the app shows them
    groups, M, c2g = lookalike_groups(ckpt["labels"])
    gpos = np.argmax(np.argsort(-(probs @ M), 1) == c2g[labels][:, None], 1)
    report["grouped"] = {"groups": [g["iso2"] for g in groups if len(g["members"]) > 1],
                         **{f"top{k}": float((gpos < k).mean()) for k in TOPKS}}
    print_report(report, source)

    out = Path(args.ckpt).parent / f"eval_{source}"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "true", "rank_of_true"] + [f"top{k + 1}" for k in range(HEADLINE_K)])
        rank = np.argsort(-probs, 1)
        for i, t in enumerate(labels):
            item = str(ds.items[i][0].relative_to(CFG.paths.real_val)) if source == "real_val" else i
            w.writerow([item, class_names[t], int(np.argmax(rank[i] == t)) + 1]
                       + [f"{class_names[k]}:{probs[i, k]:.2f}" for k in rank[i, :HEADLINE_K]])
    has_errors = save_error_grid(ds, probs, labels, class_names, out / "errors.png")
    print(f"\nsaved {out}/report.json, predictions.csv" + (", errors.png (not in top-4)" if has_errors else ""))


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
