"""Stage 4: export a checkpoint to Core ML (ML Program, iOS 16+) for Vision.

- Input: 224x224 RGB image ("image"). Pixels are scaled to [0,1] by the ImageType; the exact
  per-channel ImageNet mean/std normalization is folded into the graph (ImageType only has one
  scale for all channels).
- Output: Vision classifier (classLabel + classLabelProbs, softmax included), labels = ISO alpha-2.
  Lookalike groups (CFG.data.lookalike_groups) are merged inside the graph: their probabilities
  are added into one entry labelled e.g. "ie/ci". Metadata: "names" (label -> display name) and
  "groups" (group label -> member codes). All accuracy numbers below are on these grouped labels.
- Variants: fp16, int8 weights, 6-bit and 4-bit palettized weights. Each is compared with the
  fp32 PyTorch model on a synthetic split and on the real photos; the smallest one whose top-1
  drops by less than ExportConfig.max_top1_drop on both, and whose top-1 agrees with PyTorch on
  >= parity_min_agreement of parity_images real photos, is saved as FlagID.mlpackage.
  Parity skips near-ties (PyTorch top-1 vs top-2 margin < parity_min_margin), where fp16 rounding
  legitimately flips the order; raw agreement is reported too. An fp32 Core ML conversion is
  also checked against PyTorch (must agree on every parity image: proves the conversion is exact).

Usage: python src/export_coreml.py [--ckpt outputs/mnv3_v3/best.pt]
Writes outputs/<run>/coreml/: FlagID.mlpackage (chosen), export_report.json
(+ FlagID_<variant>.mlpackage for every variant if ExportConfig.keep_variants)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import coremltools as ct
import cv2
import coremltools.optimize as cto
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import CFG, IMAGENET_MEAN, IMAGENET_STD
from dataset import RealPhotos, SyntheticFlags, center_crop_resize, lookalike_groups, to_tensor
from model import load_checkpoint

PROBS_NAME = "classLabelProbs"
TOPK = 4


class Deployable(nn.Module):
    """[0,1] RGB in -> ImageNet normalization -> FlagNet -> softmax -> sum lookalike groups (@ M)."""

    def __init__(self, net, group_matrix):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("M", torch.from_numpy(group_matrix))

    def forward(self, x):
        return torch.softmax(self.net((x - self.mean) / self.std), dim=1) @ self.M


class SynthUint8(Dataset):
    """Deterministic synthetic images as uint8 HxWx3 (rendered in DataLoader workers)."""

    def __init__(self, spc, seed):
        self.ds = SyntheticFlags(spc, seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, c = self.ds.render(i)
        return torch.from_numpy(np.ascontiguousarray(img)), c


def load_eval_images():
    """Returns {"synthetic": (uint8 [N,224,224,3], labels), "real": (...)}."""
    spc = CFG.train.val_samples_per_class
    dl = DataLoader(SynthUint8(spc, CFG.train.val_seed + 1), batch_size=64,
                    num_workers=CFG.train.num_workers)
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in dl])
    sets = {"synthetic": (np.concatenate(xs), np.concatenate(ys))}
    real = RealPhotos()
    if real.items:
        imgs = [center_crop_resize(cv2.imread(str(f), cv2.IMREAD_COLOR)[..., ::-1], CFG.aug.out_size)
                for f, _ in real.items]
        sets["real"] = (np.stack(imgs), np.array([c for _, c in real.items]))
    else:
        print("WARNING: no real photos in data/real_val; comparing on the synthetic split only")
    return sets


@torch.no_grad()
def torch_probs(net, imgs):
    out = []
    for i in range(0, len(imgs), 128):
        x = torch.stack([to_tensor(im) for im in imgs[i:i + 128]])
        out.append(torch.softmax(net(x), 1))
    return torch.cat(out).numpy()


def coreml_probs(mlmodel, imgs, class_names):
    out = []
    for i in range(0, len(imgs), 64):
        res = mlmodel.predict([{"image": Image.fromarray(im)} for im in imgs[i:i + 64]])
        out += [[r[PROBS_NAME][c] for c in class_names] for r in res]
    return np.array(out)


def topk_acc(probs, labels, k):
    return float((np.argsort(-probs, 1)[:, :k] == labels[:, None]).any(1).mean())


def size_mb(path):
    return int(subprocess.check_output(["du", "-sk", str(path)]).split()[0]) / 1024


def convert(deploy, class_names, names, groups, precision=ct.precision.FLOAT16):
    traced = torch.jit.trace(deploy, torch.rand(1, 3, CFG.aug.out_size, CFG.aug.out_size))
    s = CFG.aug.out_size
    m = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, s, s), scale=1 / 255.0, bias=[0, 0, 0],
                             color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(class_names),
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, CFG.export.min_ios),
        compute_precision=precision,
    )
    # give the probabilities output a stable name (the converter names it after a graph variable)
    spec = m.get_spec()
    old = next(o.name for o in spec.description.output if o.name != "classLabel")
    if old != PROBS_NAME:
        ct.utils.rename_feature(spec, old, PROBS_NAME)
        m = ct.models.MLModel(spec, weights_dir=m.weights_dir)
    m.short_description = (f"Country flag classifier ({len(class_names)} entries: ISO 3166-1 alpha-2 codes; "
                           f"lookalike groups are joined with '/', e.g. 'ie/ci').")
    m.input_description["image"] = f"{s}x{s} RGB photo. Use Vision with imageCropAndScaleOption = .centerCrop."
    m.output_description["classLabel"] = "Most likely entry (ISO alpha-2 code, or a lookalike group like 'ie/ci')"
    m.output_description[PROBS_NAME] = "Probability for every entry (softmax; a group's is the sum of its members)"
    m.user_defined_metadata["names"] = json.dumps(names, ensure_ascii=False)
    m.user_defined_metadata["groups"] = json.dumps(groups)
    return m


def variants(fp16):
    th = 2048  # leave tiny tensors (biases, norms) alone
    yield "fp16", fp16
    yield "int8", cto.coreml.linear_quantize_weights(fp16, cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=th)))
    for bits in (6, 4):
        yield f"pal{bits}", cto.coreml.palettize_weights(fp16, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=th)))


def latency_ms(path, compute_units, n=100):
    m = ct.models.MLModel(str(path), compute_units=compute_units)
    x = {"image": Image.fromarray(np.random.default_rng(0).integers(0, 255, (CFG.aug.out_size,) * 2 + (3,),
                                                                    dtype=np.uint8))}
    for _ in range(10):
        m.predict(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.predict(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def ane_op_share(path):
    """Share of compute ops that Core ML plans to run on the Neural Engine (CPU_AND_NE)."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice
        m = ct.models.MLModel(str(path))  # keep alive: its compiled .mlmodelc is deleted with it
        compiled = m.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        ops = [op for f in plan.model_structure.program.functions.values() for op in f.block.operations
               if "const" not in op.operator_name]  # weights / dequantize constants, not compute
        on_ane = 0
        for op in ops:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            on_ane += u is not None and isinstance(u.preferred_compute_device, MLNeuralEngineComputeDevice)
        return on_ane, len(ops)
    except Exception as e:  # noqa: BLE001  (API differs across coremltools/macOS versions)
        print(f"  (compute plan unavailable: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.paths.outputs / "mnv3_v3" / "best.pt"))
    args = ap.parse_args()
    e = CFG.export

    net, ckpt = load_checkpoint(args.ckpt, "cpu")
    groups, M, c2g = lookalike_groups(ckpt["labels"])
    class_names = [g["iso2"] for g in groups]
    names = {g["iso2"]: g["name"] for g in groups}
    merged = {g["iso2"]: g["members"] for g in groups if len(g["members"]) > 1}
    print(f"{len(ckpt['labels'])} classes -> {len(class_names)} entries; lookalike groups: {', '.join(merged)}")
    out = Path(args.ckpt).parent / "coreml"
    out.mkdir(exist_ok=True)

    print("rendering comparison images ...")
    sets = load_eval_images()
    raw = {k: torch_probs(net, imgs) for k, (imgs, _) in sets.items()}
    ungrouped = {k: {"top1": topk_acc(raw[k], y, 1), f"top{TOPK}": topk_acc(raw[k], y, TOPK), "n": len(y)}
                 for k, (_, y) in sets.items()}
    sets = {k: (imgs, c2g[y]) for k, (imgs, y) in sets.items()}  # from here on: grouped labels
    ref = {k: p @ M for k, p in raw.items()}
    report = {"checkpoint": str(args.ckpt), "classes": len(ckpt["labels"]), "entries": len(class_names),
              "lookalike_groups": merged, "pytorch_fp32_ungrouped": ungrouped, "pytorch_fp32": {
        k: {"top1": topk_acc(ref[k], y, 1), f"top{TOPK}": topk_acc(ref[k], y, TOPK), "n": len(y)}
        for k, (_, y) in sets.items()}, "variants": {}}
    for k in sets:
        print(f"  pytorch {k}: top-1 {ungrouped[k]['top1']:.1%} -> {report['pytorch_fp32'][k]['top1']:.1%} grouped, "
              f"top-{TOPK} {ungrouped[k][f'top{TOPK}']:.1%} -> {report['pytorch_fp32'][k][f'top{TOPK}']:.1%} grouped")
    parity_set = "real" if "real" in sets else "synthetic"

    n = e.parity_images
    pref = ref[parity_set][:n]
    top2 = np.sort(pref, 1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) >= e.parity_min_margin

    def parity(p):
        same = p.argmax(1) == pref.argmax(1)
        return {"set": parity_set, "n": len(p), "n_decisive": int(decisive.sum()),
                "agreement": float(same[decisive].mean()), "raw_agreement": float(same.mean()),
                f"top{TOPK}_set_agreement": float(np.mean([set(a) == set(b) for a, b in zip(
                    np.argsort(-p, 1)[:, :TOPK], np.argsort(-pref, 1)[:, :TOPK])]))}

    print("converting ...")
    deploy = Deployable(net, M).eval()
    m32 = convert(deploy, class_names, names, merged, ct.precision.FLOAT32)
    m32 = ct.models.MLModel(m32.get_spec(), weights_dir=m32.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)
    p32 = coreml_probs(m32, sets[parity_set][0][:n], class_names)
    report["conversion_check_fp32"] = {**parity(p32), "max_abs_prob_diff": float(np.abs(p32 - pref).max())}
    print(f"  fp32 conversion check: top-1 agreement {report['conversion_check_fp32']['raw_agreement']:.1%}, "
          f"max |prob diff| {report['conversion_check_fp32']['max_abs_prob_diff']:.2e}")
    if report["conversion_check_fp32"]["raw_agreement"] < 1.0:
        print("WARNING: fp32 Core ML disagrees with PyTorch: the conversion itself is not exact")
    fp16 = convert(deploy, class_names, names, merged)
    for name, m in variants(fp16):
        path = out / f"FlagID_{name}.mlpackage"
        if path.exists():
            shutil.rmtree(path)
        m.save(str(path))
        r = {"size_mb": round(size_mb(path), 2)}
        for k, (imgs, y) in sets.items():
            p = coreml_probs(m, imgs, class_names)
            r[k] = {"top1": topk_acc(p, y, 1), f"top{TOPK}": topk_acc(p, y, TOPK),
                    "top1_drop": report["pytorch_fp32"][k]["top1"] - topk_acc(p, y, 1),
                    "agree_top1_with_pytorch": float((p.argmax(1) == ref[k].argmax(1)).mean()),
                    "max_abs_prob_diff": float(np.abs(p - ref[k]).max())}
        r["parity"] = parity(coreml_probs(m, sets[parity_set][0][:n], class_names))
        r["ok"] = (all(r[k]["top1_drop"] < e.max_top1_drop for k in sets)
                   and r["parity"]["agreement"] >= e.parity_min_agreement)
        report["variants"][name] = r
        print(f"  {name:5s} {r['size_mb']:6.2f} MB  " + "  ".join(
            f"{k}: top1 {r[k]['top1']:.1%} (drop {r[k]['top1_drop']:+.1%}) top{TOPK} {r[k][f'top{TOPK}']:.1%}"
            for k in sets) + f"  parity {r['parity']['agreement']:.0%} (raw {r['parity']['raw_agreement']:.0%})"
            f"  {'OK' if r['ok'] else 'rejected'}")

    ok = [k for k, r in report["variants"].items() if r["ok"]]
    chosen = min(ok or ["fp16"], key=lambda k: report["variants"][k]["size_mb"])
    if not ok:
        print("WARNING: no variant met the accuracy/parity limits; keeping fp16")
    final = out / "FlagID.mlpackage"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(out / f"FlagID_{chosen}.mlpackage", final)
    report["chosen"] = chosen

    print("timing (median of 100 single-image predictions from Python, includes ~1 ms overhead) ...")
    report["latency_ms"] = {}
    for name in report["variants"]:
        p = out / f"FlagID_{name}.mlpackage"
        lat = {cu: round(latency_ms(p, getattr(ct.ComputeUnit, cu)), 2) for cu in ("CPU_AND_NE", "ALL", "CPU_ONLY")}
        report["latency_ms"][name] = lat
        print(f"  {name:5s} " + "  ".join(f"{cu}: {v:.2f} ms" for cu, v in lat.items()))
    share = ane_op_share(final)
    if share:
        report["ane_ops"] = {"on_neural_engine": share[0], "total": share[1]}
        print(f"  {chosen}: {share[0]}/{share[1]} compute ops planned on the Neural Engine")

    if not e.keep_variants:
        for name in report["variants"]:
            shutil.rmtree(out / f"FlagID_{name}.mlpackage")
    (out / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    c = report["variants"][chosen]
    print(f"\nchosen: {chosen} -> {final}  size {c['size_mb']:.2f} MB (limit 25 MB)  "
          f"latency CPU_AND_NE {report['latency_ms'][chosen]['CPU_AND_NE']:.2f} ms (limit 10 ms)  "
          f"parity {c['parity']['agreement']:.0%} on {c['parity']['n_decisive']} of {c['parity']['n']} "
          f"{c['parity']['set']} images that aren't near-ties (raw {c['parity']['raw_agreement']:.0%})")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
