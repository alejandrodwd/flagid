"""Stage 2: train on on-the-fly synthetic flags, validate on a fixed-seed synthetic split.

Usage:
    python src/train.py                                   # full run (config defaults)
    python src/train.py --epochs 1 --samples-per-class 4  # smoke test
    python src/train.py --backbone efficientnet_b0
    python src/train.py --labels data/labels_world_es.json --init-from outputs/mnv3_v3/best.pt --epochs 6
        # region update: continue from an earlier model; known classes keep their head rows
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import SyntheticFlags
from model import FlagNet, get_device


def amp_works(model, device, num_classes):
    """Try one fp16 autocast forward/backward step on this device; return True if it works."""
    if device.type != "mps":
        return False
    try:
        # full step incl. GradScaler (fp16 needs loss scaling), on a throwaway optimizer
        x = torch.randn(2, 3, CFG.aug.out_size, CFG.aug.out_size, device=device)
        scaler = torch.amp.GradScaler("mps")
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            loss = model(x).float().logsumexp(1).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        model.zero_grad(set_to_none=True)
        return bool(torch.isfinite(loss))
    except Exception as e:  # noqa: BLE001
        print(f"mixed precision unavailable on mps ({e.__class__.__name__}: {e}); using fp32")
        return False


def init_from(model, path, class_ids):
    """Copy backbone + the head rows of classes the old checkpoint already knew (matched by id)."""
    old = torch.load(path, map_location="cpu", weights_only=False)
    assert old["backbone"] == model.backbone_name, "init-from checkpoint has a different backbone"
    sd = {k: v for k, v in old["model"].items() if not k.startswith("head.")}
    model.load_state_dict(sd, strict=False)
    old_idx = {l["iso2"]: i for i, l in enumerate(old["labels"])}
    lin = model.head[-1]
    with torch.no_grad():
        for i, c in enumerate(class_ids):
            if c in old_idx:
                lin.weight[i] = old["model"]["head.1.weight"][old_idx[c]]
                lin.bias[i] = old["model"]["head.1.bias"][old_idx[c]]
    print(f"init from {path}: backbone + {sum(c in old_idx for c in class_ids)} known classes, "
          f"{sum(c not in old_idx for c in class_ids)} new")


def lr_lambda_factory(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return f


@torch.no_grad()
def validate(model, loader, device, criterion, use_amp):
    model.eval()
    n = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x).float()
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def plot_history(hist, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in hist]
    a1.plot(ep, [h["train_loss"] for h in hist], label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], label="val (synthetic)")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss (label-smoothed CE)"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, [h["train_acc"] for h in hist], label="train")
    a2.plot(ep, [h["val_acc"] for h in hist], label="val (synthetic)")
    a2.set_xlabel("epoch"); a2.set_ylabel("top-1 accuracy"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    t = CFG.train
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=t.epochs)
    ap.add_argument("--samples-per-class", type=int, default=t.samples_per_class_per_epoch)
    ap.add_argument("--val-samples-per-class", type=int, default=t.val_samples_per_class)
    ap.add_argument("--backbone", default=t.backbone)
    ap.add_argument("--batch-size", type=int, default=t.batch_size)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--run-name", default=None, help="subfolder of outputs/ (default: backbone name)")
    ap.add_argument("--labels", default=None, help="label list (default: data/labels.json)")
    ap.add_argument("--init-from", default=None, help="checkpoint to continue from (classes matched by id)")
    args = ap.parse_args()

    torch.manual_seed(t.seed)
    device = get_device()
    out_dir = CFG.paths.outputs / (args.run_name or args.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SyntheticFlags(args.samples_per_class, seed=None, labels_path=args.labels)
    val_ds = SyntheticFlags(args.val_samples_per_class, seed=t.val_seed, labels_path=args.labels)
    nc = train_ds.num_classes
    loader_kw = dict(batch_size=args.batch_size, num_workers=t.num_workers, persistent_workers=t.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = FlagNet(nc, backbone=args.backbone)
    if args.init_from:
        init_from(model, args.init_from, [l["iso2"] for l in train_ds.labels])
    model = model.to(device)
    use_amp = t.amp and not args.no_amp and amp_works(model, device, nc)
    print(f"device={device} amp={use_amp} classes={nc} train/epoch={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": t.lr_backbone},
         {"params": model.head.parameters(), "lr": t.lr_head}],
        weight_decay=t.weight_decay,
    )
    steps_per_epoch = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(args.epochs * steps_per_epoch, int(t.warmup_epochs * steps_per_epoch)))
    scaler = torch.amp.GradScaler("mps", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)

    hist, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x).float()
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        val_loss, val_acc = validate(model, val_dl, device, criterion, use_amp)
        h = dict(epoch=epoch, train_loss=loss_sum / n, train_acc=correct / n, val_loss=val_loss,
                 val_acc=val_acc, lr_backbone=opt.param_groups[0]["lr"], seconds=time.time() - t0)
        hist.append(h)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}), flush=True)

        ckpt = dict(model=model.state_dict(), backbone=args.backbone, num_classes=nc, epoch=epoch,
                    val_acc=val_acc, labels=train_ds.labels)
        torch.save(ckpt, out_dir / "last.pt")
        if val_acc > best:
            best = val_acc
            torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
        plot_history(hist, out_dir / "training_curves.png")
    print(f"done. best synthetic val top-1 = {best:.4f}; checkpoints + curves in {out_dir}")


if __name__ == "__main__":
    # Python 3.12's multiprocessing resource tracker can block forever in waitpid() at interpreter
    # exit after DataLoader workers were used on macOS. Checkpoints are saved each epoch, so exit hard
    # (also after an exception, which is printed first).
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
