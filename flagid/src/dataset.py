"""Torch datasets: on-the-fly synthetic flags (train / synthetic val) and real photos (eval)."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augment import synth_flag_photo, synth_no_flag
from cloth_shapes import load_shape
from config import CFG, IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, np.float32)
_STD = np.array(IMAGENET_STD, np.float32)


def load_labels(path=None):
    """Returns list where index = class id, item = {'iso2', 'name'}."""
    raw = json.loads(Path(path or CFG.paths.labels).read_text())
    return [raw[str(i)] for i in range(len(raw))]


def lookalike_groups(labels):
    """Merges CFG.data.lookalike_groups into single entries (in label order, at the position of the
    first member). Returns (groups [{'iso2': 'ie/ci', 'name': 'Ireland / Côte d'Ivoire',
    'members': ['ie', 'ci']}, ...], M float32 [C, G] so that grouped_probs = probs @ M,
    class_to_group int [C])."""
    names = {l["iso2"]: l["name"] for l in labels}
    of = {}
    for g in CFG.data.lookalike_groups:
        members = [c for c in g if c in names]
        if len(members) > 1:
            of.update({c: members for c in members})
    groups, index, c2g = [], {}, np.zeros(len(labels), int)
    for i, l in enumerate(labels):
        members = of.get(l["iso2"], [l["iso2"]])
        key = "/".join(members)
        if key not in index:
            index[key] = len(groups)
            groups.append({"iso2": key, "name": " / ".join(names[m] for m in members), "members": members})
        c2g[i] = index[key]
    M = np.zeros((len(labels), len(groups)), np.float32)
    M[np.arange(len(labels)), c2g] = 1
    return groups, M, c2g


def to_tensor(img_uint8):
    x = (img_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SyntheticFlags(Dataset):
    """Countries are balanced (samples_per_class each); no_flag gets
    ceil(samples_per_class * no_flag_fraction) items at the end of the index range.

    seed=None -> fresh randomness on every call (training: new variation each epoch).
    seed=int  -> item i is deterministic (synthetic validation split).
    """

    def __init__(self, samples_per_class, seed=None, aug_cfg=None, shapes_dir=None, labels_path=None):
        self.labels = load_labels(labels_path)
        self.num_classes = len(self.labels)
        self.no_flag_idx = next((i for i, l in enumerate(self.labels) if l["iso2"] == CFG.data.no_flag_label), None)
        self.flag_ids = [i for i in range(self.num_classes) if i != self.no_flag_idx]
        self.seed = seed
        self.cfg = aug_cfg or CFG.aug
        self.num_flag_items = samples_per_class * len(self.flag_ids)
        n_no_flag = math.ceil(samples_per_class * self.cfg.no_flag_fraction) if self.no_flag_idx is not None else 0
        self.length = self.num_flag_items + n_no_flag
        self.bg_paths = sorted(CFG.paths.backgrounds.glob("*.jpg"))
        assert self.bg_paths, "no backgrounds; run prepare_data.py backgrounds"
        # cloth-sim shapes are read from disk per sample (the library is too big to cache in RAM)
        self.shape_paths = sorted(Path(shapes_dir or CFG.paths.cloth_shapes).glob("*.npz"))
        if not self.shape_paths:
            print(f"NOTE: no cloth-sim shapes in {shapes_dir or CFG.paths.cloth_shapes}; procedural cloth only")
        self._flags = {}  # lazy per-worker cache of RGBA rasters

    def _shape(self, rng):
        if not self.shape_paths:
            return None
        return load_shape(self.shape_paths[int(rng.integers(len(self.shape_paths)))])

    def __len__(self):
        return self.length

    def _flag(self, cls):
        if cls not in self._flags:
            path = CFG.paths.flags_png / f"{self.labels[cls]['iso2']}.png"
            im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 3:
                im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            self._flags[cls] = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        return self._flags[cls]

    def _bg(self, rng):
        path = self.bg_paths[int(rng.integers(len(self.bg_paths)))]
        return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1]

    def render(self, idx):
        """Returns (uint8 HxWx3 image, class id)."""
        cls = self.flag_ids[idx % len(self.flag_ids)] if idx < self.num_flag_items else self.no_flag_idx
        rng = np.random.default_rng() if self.seed is None else np.random.default_rng([self.seed, idx])
        if cls == self.no_flag_idx:
            img = synth_no_flag(rng, self._bg, self.cfg)
        else:
            img = synth_flag_photo(rng, self._flag(cls), self._bg, self.cfg, self._shape)
        return img, cls

    def __getitem__(self, idx):
        img, cls = self.render(idx)
        return to_tensor(img), cls


def center_crop_resize(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    img = img[y:y + s, x:x + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class RealPhotos(Dataset):
    """data/real_val/<iso2>/*.jpg|png. Center-crop + resize to out_size (same as Vision's
    default centerCrop option on iOS). Unknown folder names are skipped with a warning."""

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # convert HEIC to JPEG first

    def __init__(self, root=None):
        self.labels = load_labels()
        iso_to_idx = {l["iso2"]: i for i, l in enumerate(self.labels)}
        root = Path(root or CFG.paths.real_val)
        self.items = []
        for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            if d.name.lower() not in iso_to_idx:
                print(f"WARNING: real_val folder '{d.name}' is not a known class, skipped")
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in self.EXTS:
                    self.items.append((f, iso_to_idx[d.name.lower()]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1]
        img = center_crop_resize(img, CFG.aug.out_size)
        return to_tensor(img), cls
