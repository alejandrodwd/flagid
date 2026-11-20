"""Cloth-sim shape library: convert Blender EXRs -> compact .npz shapes, and map any flag onto any
shape (one remap through the rendered UV coordinates, times the rendered lighting).

Usage:
    python src/cloth_shapes.py convert data/cloth_raw data/cloth_shapes
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SHAPE_MAX_SIDE = 384      # stored shapes are cropped to the cloth (and downscaled to this if larger);
                          # 256 gave visibly jaggy edges on close-ups
CROP_MARGIN = 0.04        # extra border around the cloth bbox (fraction of its size)
POLE_MIN_PIXELS = 20
# "Could a person still tell which flag this is?": share of the flag's surface that is visible,
# measured on a VIS_GRID of UV cells (a cell counts if it covers >= VIS_MIN_PIXELS pixels).
# Shapes below MIN_VISIBLE_DESIGN (edge-on views, crumpled rags) are dropped on conversion.
VIS_GRID = (12, 8)
VIS_MIN_PIXELS = 3
MIN_VISIBLE_DESIGN = 0.5


# ----------------------------------------------------------------------------- colour helpers
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ----------------------------------------------------------------------------- EXR -> npz
def read_exr(path):
    import OpenEXR  # only needed for conversion
    out = {}
    with OpenEXR.File(str(path), separate_channels=True) as f:
        for part in f.parts:
            for name, ch in part.channels.items():
                out[name.replace("ViewLayer.", "")] = np.asarray(ch.pixels, np.float32)
    return out


def stack(ch, layer, comps):
    return np.stack([ch[f"{layer}.{c}"] for c in comps], -1)


def visible_design(uv, cloth):
    """Fraction of UV grid cells (VIS_GRID) that are visible with at least VIS_MIN_PIXELS pixels."""
    m = cloth > 0.5
    gx, gy = VIS_GRID
    cu = np.clip((uv[..., 0][m] * gx).astype(int), 0, gx - 1)
    cv = np.clip((uv[..., 1][m] * gy).astype(int), 0, gy - 1)
    counts = np.bincount(cv * gx + cu, minlength=gx * gy)
    return float((counts >= VIS_MIN_PIXELS).mean())


def fit_pole(pole_mask, combined_rgb):
    """Fit a line to the visible pole. Returns finial end, direction toward the cut end, thickness,
    colour (linear) in full-render pixel coords, or None."""
    ys, xs = np.nonzero(pole_mask > 0.5)
    if len(xs) < POLE_MIN_PIXELS:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy], np.float32)
    t = (pts - [x0, y0]) @ d
    e1, e2 = np.array([x0, y0]) + d * t.min(), np.array([x0, y0]) + d * t.max()
    h, w = pole_mask.shape
    border = lambda p: min(p[0], p[1], w - 1 - p[0], h - 1 - p[1])  # noqa: E731
    if border(e1) < border(e2):  # e1 touches the frame edge -> it is the cut end
        e1, e2, d = e2, e1, -d
    # d now points from the finial end (e1) toward the cut end
    length = max(float(t.max() - t.min()), 1.0)
    return dict(finial=e1.astype(np.float32), direction=d, thickness=float(len(xs) / length),
                color=np.median(combined_rgb[pole_mask > 0.5], 0).astype(np.float32))


def convert_one(exr_path):
    ch = read_exr(exr_path)
    comb = stack(ch, "Combined", "RGBA")
    # Cycles light passes exclude the gloss colour; the actual specular term is (dir + ind) * colour
    gloss = (stack(ch, "Glossy Direct", "RGB") + stack(ch, "Glossy Indirect", "RGB")) * stack(ch, "Glossy Color", "RGB")
    uv = stack(ch, "uv", "RG")
    cloth, pole, back = ch["cloth.X"], ch["pole.X"], ch["back.X"]
    # AOVs are premultiplied by pixel coverage: un-premultiply so edge pixels index the right spot
    uv = np.clip(uv / np.maximum(cloth, 1e-3)[..., None], 0, 1)
    ys, xs = np.nonzero(cloth > 0.5)
    if len(xs) < 50:
        return None
    vis = visible_design(uv, cloth)  # measured at full render resolution
    pole_fit = fit_pole(pole, comb[..., :3])
    # crop to the cloth bbox (+margin); the pole beyond the crop is re-drawn from pole_fit later
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = int((x1 - x0) * CROP_MARGIN) + 1, int((y1 - y0) * CROP_MARGIN) + 1
    H, W = cloth.shape
    x0, x1, y0, y1 = max(0, x0 - mx), min(W, x1 + mx), max(0, y0 - my), min(H, y1 + my)
    s = min(1.0, SHAPE_MAX_SIDE / max(x1 - x0, y1 - y0))
    size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
    rs = lambda a, interp=cv2.INTER_AREA: cv2.resize(a[y0:y1, x0:x1], size, interpolation=interp)  # noqa: E731
    # UVs must not be averaged across folds -> nearest; masks/lighting -> area
    shape = dict(
        uv=rs(uv, cv2.INTER_NEAREST).astype(np.float16),
        cloth=rs(cloth).astype(np.float16),
        pole=rs(pole).astype(np.float16),
        shade=rs(comb[..., :3]).astype(np.float16),            # white cloth under the HDRI (+ pole)
        gloss=rs(cv2.GaussianBlur(gloss, (0, 0), 1.0)).astype(np.float16),
        back_frac=np.float32((back[cloth > 0.5] > 0.5).mean()),
        visible_design=np.float32(vis),
    )
    if pole_fit:
        shape["pole_finial"] = (pole_fit["finial"] - [x0, y0]) * s
        shape["pole_dir"] = pole_fit["direction"]
        shape["pole_thickness"] = np.float32(pole_fit["thickness"] * s)
        shape["pole_color"] = pole_fit["color"]
    return shape


def convert_dir(raw_dir, out_dir):
    from tqdm import tqdm
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {}
    pfile = raw_dir / "params.jsonl"
    if pfile.exists():
        for line in pfile.read_text().splitlines():
            p = json.loads(line)
            params[p["name"]] = p
    n_ok, rejected = 0, []
    for exr in tqdm(sorted(raw_dir.glob("*.exr")), desc="convert"):
        dst = out_dir / (exr.stem + ".npz")
        if dst.exists():
            n_ok += 1
            continue
        shape = convert_one(exr)
        if shape is None or shape["visible_design"] < MIN_VISIBLE_DESIGN:
            rejected.append((exr.stem, None if shape is None else round(float(shape["visible_design"]), 2)))
            continue
        shape["params"] = np.array(json.dumps(params.get(exr.stem, {})))
        np.savez_compressed(dst, **shape)
        n_ok += 1
    (out_dir / "rejected.json").write_text(json.dumps(rejected, indent=1))
    print(f"{n_ok} shapes in {out_dir}; rejected {len(rejected)} "
          f"(visible design < {MIN_VISIBLE_DESIGN} or empty), listed in rejected.json")


def load_shape(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- flag -> shape
def map_flag(rng, shape, flag_rgba_uint8, upside_down=False, exposure=(0.75, 1.05), gloss_gain=(0.4, 1.3)):
    """Returns float32 RGBA layer (sRGB, premultiplied-free) of the flag draped on the shape."""
    fh, fw = flag_rgba_uint8.shape[:2]
    flag = flag_rgba_uint8.astype(np.float32) / 255.0
    rgb_lin = srgb_to_linear(flag[..., :3])
    tex = np.dstack([rgb_lin, flag[..., 3:4]])
    uv = shape["uv"].astype(np.float32)
    u, v = uv[..., 0], uv[..., 1]
    if upside_down:  # rotate the design 180 deg on the cloth
        u, v = 1 - u, 1 - v
    map_x = (u * (fw - 1)).astype(np.float32)
    map_y = ((1 - v) * (fh - 1)).astype(np.float32)   # Blender v=0 is the bottom edge
    t = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cloth = shape["cloth"].astype(np.float32)
    pole = shape["pole"].astype(np.float32)
    shade = shape["shade"].astype(np.float32)
    gloss = shape["gloss"].astype(np.float32)
    # exposure: normalise so bright cloth regions land at a random level below clipping
    lum = shade[cloth > 0.5].mean(-1)
    k = rng.uniform(*exposure) / (np.percentile(lum, 95) + 1e-4) if lum.size else 1.0
    # Combined = diffuse light (white albedo) + specular -> recover diffuse, recolour it, add specular
    diffuse = np.clip(shade - gloss, 0, None)
    cloth_rgb = t[..., :3] * diffuse * k + gloss * k * rng.uniform(*gloss_gain)
    pole_rgb = shade * k                                 # Combined already holds the lit pole
    a_cloth = cloth * t[..., 3]
    alpha = np.clip(a_cloth + pole, 0, 1)
    rgb = (cloth_rgb * a_cloth[..., None] + pole_rgb * pole[..., None]) / np.maximum(alpha, 1e-4)[..., None]
    out = np.dstack([linear_to_srgb(rgb), alpha]).astype(np.float32)
    out[..., :3] = np.clip(out[..., :3], 0, 1)
    return out, k


def draw_pole_extension(img, finial, direction, thickness, color_srgb):
    """Pole from the finial end off to the frame edge (drawn *under* the flag layer)."""
    S = max(img.shape[:2])
    end = finial + direction * S * 4
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.line(mask, tuple(int(c) for c in finial), tuple(int(c) for c in end), 1.0,
             max(1, int(round(thickness))), cv2.LINE_AA)
    return img * (1 - mask[..., None]) + np.asarray(color_srgb, np.float32) * mask[..., None]


def paste_shape(rng, img_bg, layer, shape, k, scale_frac, rotation_deg, edge_crop_p):
    """Scale/rotate/place the flag layer (and pole extension) onto a square background."""
    S = img_bg.shape[0]
    h, w = layer.shape[:2]
    cloth = shape["cloth"].astype(np.float32)
    ys, xs = np.nonzero(cloth > 0.5)
    area = len(xs)
    s = np.sqrt(scale_frac * S * S / max(area, 1))      # cloth covers ~scale_frac of the frame
    ang = rng.uniform(-rotation_deg, rotation_deg)
    cx, cy = xs.mean(), ys.mean()
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, s)
    # where the cloth bbox lands -> choose a centre (sometimes partly out of frame)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float32)
    pc = corners @ M[:, :2].T + M[:, 2]
    lo, hi = pc.min(0), pc.max(0)
    slack = 0.4 * (hi - lo) if rng.random() < edge_crop_p else np.zeros(2)
    t = np.empty(2, np.float32)
    for i in range(2):
        a_min, a_max = -lo[i] - slack[i], S - hi[i] + slack[i]
        t[i] = rng.uniform(a_min, a_max) if a_max > a_min else (a_min + a_max) / 2
    M[:, 2] += t
    img = img_bg
    if "pole_finial" in shape:
        fin = M[:, :2] @ shape["pole_finial"] + M[:, 2]
        d = M[:, :2] @ shape["pole_dir"]
        d = d / (np.linalg.norm(d) + 1e-6)
        col = linear_to_srgb(np.clip(shape["pole_color"] * k, 0, 1))
        img = draw_pole_extension(img, fin, d, float(shape["pole_thickness"]) * s, col)
    warped = cv2.warpAffine(layer, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = warped[..., 3:4]
    q = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], np.float32) + t
    return img * (1 - a) + warped[..., :3] * a, q


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "convert":
        convert_dir(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
