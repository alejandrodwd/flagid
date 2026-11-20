"""Add-a-region test: how well are flags ranked that a model was (or wasn't) trained on?

Two ways to score a model:
  head         the model's own softmax over the classes it was trained on
  fingerprint  backbone features (L2-normalized) compared by cosine similarity to one reference
               fingerprint per flag = mean feature of `--proto-renders` synthetic renders of it.
               This needs no training for new flags: that is the "add a region by data file" path.

Test images: a synthetic split (different seed from training/val and from the fingerprint renders)
for every class in --labels, plus the real photos in data/real_val (countries only so far).
Groups are reported separately: flags new in --labels vs. the base set (data/labels.json).

Usage:
    python src/region_test.py --labels data/labels_world_es.json \
        --ckpt outputs/mnv3_v3/best.pt outputs/mnv3_v3_es/best.pt
"""
import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CFG
from dataset import RealPhotos, SyntheticFlags, load_labels, to_tensor
from model import get_device, load_checkpoint

KS = (1, 4, 10)


class Synth(Dataset):
    def __init__(self, spc, seed, labels_path):
        self.ds = SyntheticFlags(spc, seed=seed, labels_path=labels_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, c = self.ds.render(i)
        return to_tensor(img), c


@torch.no_grad()
def run(model, ds, device):
    """Returns (normalized backbone features [N,D], head probs [N,C], labels [N])."""
    dl = DataLoader(ds, batch_size=128, num_workers=CFG.train.num_workers)
    feats, probs, ys = [], [], []
    for x, y in dl:
        f = model.backbone(x.to(device))
        feats.append(F.normalize(f, dim=1).cpu())
        probs.append(torch.softmax(model.head(f), 1).cpu())
        ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(probs).numpy(), torch.cat(ys).numpy()


def topk(scores, labels):
    rank = np.argsort(-scores, 1)
    pos = np.argmax(rank == labels[:, None], 1)
    return {f"top{k}": float((pos < k).mean()) for k in KS}, rank, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--proto-renders", type=int, default=40)
    ap.add_argument("--test-per-class", type=int, default=20)
    ap.add_argument("--out", default=str(CFG.paths.outputs / "region_test.json"))
    args = ap.parse_args()

    device = get_device()
    labels = load_labels(args.labels)
    ids = [l["iso2"] for l in labels]
    base = {l["iso2"] for l in load_labels()}
    new_mask = np.array([c not in base for c in ids])
    C = len(ids)

    proto_ds = Synth(args.proto_renders, CFG.train.val_seed + 101, args.labels)
    test_ds = Synth(args.test_per_class, CFG.train.val_seed + 202, args.labels)
    real_ds = RealPhotos()
    report = {"labels": args.labels, "classes": C, "new_classes": int(new_mask.sum()),
              "proto_renders": args.proto_renders, "test_per_class": args.test_per_class, "models": {}}

    for ck in args.ckpt:
        model, meta = load_checkpoint(ck, device)
        m_ids = [l["iso2"] for l in meta["labels"]]
        print(f"\n=== {ck}  (trained on {len(m_ids)} classes) ===")
        pf, _, py = run(model, proto_ds, device)
        protos = np.zeros((C, pf.shape[1]), np.float32)
        for c in range(C):
            v = pf[py == c].mean(0)
            protos[c] = v / np.linalg.norm(v)
        res = {}
        for set_name, ds in (("synthetic", test_ds), ("real", real_ds)):
            tf, tp, ty = run(model, ds, device)
            # head scores mapped into this label space (classes the model doesn't know get -1)
            head = np.full((len(ty), C), -1.0, np.float32)
            for j, c in enumerate(m_ids):
                if c in ids:
                    head[:, ids.index(c)] = tp[:, j]
            fp = tf @ protos.T
            for method, scores in (("head", head), ("fingerprint", fp)):
                for grp, m in (("new", new_mask[ty]), ("base", ~new_mask[ty])):
                    if m.sum() == 0:
                        continue
                    acc, rank, pos = topk(scores[m], ty[m])
                    key = f"{set_name}/{grp}/{method}"
                    res[key] = {**acc, "n": int(m.sum())}
                    if grp == "new":
                        yy = ty[m]
                        res[key]["per_flag_top4"] = {ids[c]: float((pos[yy == c] < 4).mean()) for c in sorted(set(yy))}
                        res[key]["confused_top1"] = [f"{ids[a]}->{ids[b]} x{n}" for (a, b), n in Counter(
                            (int(a), int(b)) for a, b in zip(yy, rank[:, 0]) if a != b).most_common(10)]
                    print(f"  {key:30s} n={int(m.sum()):5d}  " + "  ".join(f"top-{k} {acc[f'top{k}']:6.1%}" for k in KS))
        report["models"][ck] = res

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
