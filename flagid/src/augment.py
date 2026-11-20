"""Synthetic 'photo of a flag' generator (numpy + OpenCV).

Every function takes an explicit numpy Generator `rng`, so the same code serves the
random training stream and the fixed-seed synthetic validation split.
Images are float32 RGB in [0, 1] unless noted.
"""
import cv2
import numpy as np

from cloth_shapes import map_flag, paste_shape
from config import AugmentConfig


def U(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def smooth_noise(rng, h, w, cells=(3, 8)):
    """Cheap Perlin-like noise: tiny random grid upsampled with bicubic. Range ~[-1, 1]."""
    c = int(rng.integers(*cells))
    grid = rng.uniform(-1, 1, (c, c)).astype(np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- backgrounds
def procedural_background(rng, size):
    kind = rng.integers(3)
    if kind == 0:  # sky gradient, sometimes with clouds
        top = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.45, 0.7), rng.uniform(0.75, 1.0)])
        bottom = np.clip(top + rng.uniform(0.1, 0.4), 0, 1)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
        img = np.broadcast_to(top * (1 - t) + bottom * t, (size, size, 3)).astype(np.float32).copy()
        if rng.random() < 0.6:
            clouds = np.clip(smooth_noise(rng, size, size, (4, 10)) * 1.5, 0, 1)[..., None]
            img = img * (1 - clouds * 0.8) + clouds * 0.8 * rng.uniform(0.85, 1.0)
        if rng.random() < 0.3:  # overcast grey
            g = img.mean(axis=2, keepdims=True)
            img = img * 0.3 + g * 0.7
    elif kind == 1:  # coloured low-frequency noise
        img = np.stack([smooth_noise(rng, size, size, (2, 12)) for _ in range(3)], -1) * 0.35
        img = img + rng.uniform(0.2, 0.7, 3)
    else:  # flat-ish wall with gradient + fine noise
        base = rng.uniform(0.1, 0.9, 3)
        grad = smooth_noise(rng, size, size, (2, 3))[..., None] * 0.15
        img = base + grad + rng.normal(0, 0.02, (size, size, 3))
    return np.clip(img, 0, 1).astype(np.float32)


def photo_background(rng, bg_rgb_uint8, size, zoom=(0.3, 1.0)):
    """Random crop (random scale) of a photo, resized to size x size."""
    h, w = bg_rgb_uint8.shape[:2]
    side = int(min(h, w) * rng.uniform(*zoom))
    side = max(side, 16)
    y = int(rng.integers(0, h - side + 1))
    x = int(rng.integers(0, w - side + 1))
    crop = bg_rgb_uint8[y:y + side, x:x + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    return crop.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- flag geometry
def cloth_warp(rng, flag, cfg: AugmentConfig, flipped):
    """Wave/fold displacement of an RGBA flag, with shading from the displacement.
    Returns padded RGBA image and the pad (flag rect starts at (pad, pad))."""
    h, w = flag.shape[:2]
    A = U(rng, cfg.wave_amplitude_frac) * h
    k = U(rng, cfg.wave_count)
    tilt = rng.uniform(-0.6, 0.6)
    phase = rng.uniform(0, 2 * np.pi)
    pad = int(np.ceil(A * 2.2)) + 2
    H, W = h + 2 * pad, w + 2 * pad
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = xs - pad, ys - pad
    xn = np.clip(x / w, 0, 1)
    # hoist (at the pole) moves less than the fly end
    hoist_side = (1 - xn) if flipped else xn
    envelope = 0.25 + 0.75 * hoist_side
    theta = 2 * np.pi * (k * xn + tilt * y / h) + phase
    d = np.sin(theta) + 0.35 * np.sin(2.3 * theta + 1.0) + 0.5 * smooth_noise(rng, H, W, (3, 6))
    if rng.random() < cfg.fold_p:  # sharp creases: |sin| has cusps where the cloth folds
        k2, tilt2 = U(rng, cfg.fold_count), rng.uniform(-1.0, 1.0)
        d += 0.6 * (1 - 2 * np.abs(np.sin(np.pi * (k2 * xn + tilt2 * y / h) + rng.uniform(0, np.pi))))
    d = (d * A * envelope).astype(np.float32)
    map_x = x + 0.3 * d
    map_y = y - d
    out = cv2.remap(flag, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # shading: surface slope along x (light from the side) + depth (troughs darker)
    slope = np.gradient(d, axis=1)
    slope_n = slope / (np.abs(slope).max() + 1e-6)
    depth_n = d / (np.abs(d).max() + 1e-6)
    s = U(rng, cfg.shading_strength)
    shade = np.clip(1 + s * (0.7 * slope_n - 0.3 * depth_n), *cfg.shade_clip)
    out[..., :3] *= shade[..., None]
    if rng.random() < cfg.specular_p:  # sheen on slopes facing the light (additive, so it washes colour)
        spec = U(rng, cfg.specular_strength) * np.clip(slope_n, 0, 1) ** 3
        out[..., :3] += spec[..., None] * out[..., 3:4]
    return np.clip(out, 0, None), pad


def cloth_look(rng, flag, cfg: AugmentConfig):
    """Dye/fading jitter and fine fabric texture on an RGBA cloth (in place-safe copy)."""
    flag = flag.copy()
    rgb = np.clip(flag[..., :3], 0, 1) ** U(rng, cfg.flag_gamma)
    rgb = rgb * (1 + rng.uniform(-cfg.flag_color_gain, cfg.flag_color_gain, 3))
    h, w = rgb.shape[:2]
    tex = cv2.GaussianBlur(rng.normal(0, U(rng, cfg.fabric_noise), (h, w)).astype(np.float32), (0, 0), 0.7)
    flag[..., :3] = np.clip(rgb * (1 + tex[..., None]), 0, 1)
    return flag


def draw_pole(rng, img, p_top, p_bottom, thickness):
    """Metallic/wooden pole along the line p_top -> p_bottom, extended to below the frame,
    with a small finial above the top. Drawn over the image."""
    S = img.shape[0]
    d = (p_bottom - p_top) / (np.linalg.norm(p_bottom - p_top) + 1e-6)
    top = p_top - d * thickness * rng.uniform(1, 4)
    bottom = p_bottom + d * S * 2
    base = np.array([rng.uniform(0.35, 0.9)] * 3) * (1 + rng.uniform(-0.12, 0.12, 3))
    mask = np.zeros((S, S), np.float32)
    hi = np.zeros((S, S), np.float32)
    pt = lambda v: tuple(int(round(c)) for c in v)  # noqa: E731
    cv2.line(mask, pt(top), pt(bottom), 1.0, max(1, int(thickness)), cv2.LINE_AA)
    perp = np.array([-d[1], d[0]]) * thickness * 0.2
    cv2.line(hi, pt(top + perp), pt(bottom + perp), 1.0, max(1, int(thickness * 0.3)), cv2.LINE_AA)
    cv2.circle(mask, pt(top), max(1, int(thickness * 0.9)), 1.0, -1, cv2.LINE_AA)
    color = base * (1 + 0.5 * hi[..., None])
    return img * (1 - mask[..., None]) + np.clip(color, 0, 1) * mask[..., None]


def place_flag(rng, flag_rgba, cfg: AugmentConfig, aspect):
    """Warp a (cloth-warped, padded) flag into the output frame with rotation/perspective.
    Returns RGBA layer of size out_size and the destination quad of the flag cloth."""
    S = cfg.out_size
    frac = U(rng, cfg.flag_scale)
    fh = np.sqrt(frac * S * S / aspect)
    fw = fh * aspect
    # destination quad around origin, jitter corners, rotate
    q = np.array([[-fw / 2, -fh / 2], [fw / 2, -fh / 2], [fw / 2, fh / 2], [-fw / 2, fh / 2]], np.float32)
    q += rng.uniform(-1, 1, (4, 2)).astype(np.float32) * cfg.perspective_strength * np.array([fw, fh], np.float32)
    a = np.deg2rad(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    q = q @ R.T
    # placement: normally fully inside the frame (if it fits), sometimes cut by the edge
    lo, hi = q.min(0), q.max(0)
    ext = hi - lo
    if rng.random() < cfg.edge_crop_p:
        slack = 0.4 * ext
    else:
        slack = np.zeros(2, np.float32)
    c = np.empty(2, np.float32)
    for i in range(2):
        a_min, a_max = -lo[i] - slack[i], S - hi[i] + slack[i]
        c[i] = rng.uniform(a_min, a_max) if a_max > a_min else (a_min + a_max) / 2
    q = q + c
    return q


def render_flag_layer(rng, flag_rgba, q, pad, S):
    h, w = flag_rgba.shape[:2]
    src = np.array([[pad, pad], [w - pad, pad], [w - pad, h - pad], [pad, h - pad]], np.float32)
    M = cv2.getPerspectiveTransform(src, q.astype(np.float32))
    layer = cv2.warpPerspective(flag_rgba, M, (S, S), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return layer


# ----------------------------------------------------------------------------- photometrics
def lighting(rng, img, cfg: AugmentConfig):
    S = img.shape[0]
    img = (img - 0.5) * U(rng, cfg.contrast) + 0.5
    img = img * U(rng, cfg.brightness)
    if rng.random() < cfg.gradient_light_p:
        ang = rng.uniform(0, 2 * np.pi)
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
        t = xs * np.cos(ang) + ys * np.sin(ang) + 0.5
        g0, g1 = rng.uniform(0.6, 1.2, 2)
        img = img * (g0 + (g1 - g0) * t)[..., None]
    if rng.random() < cfg.shadow_p:
        ang = rng.uniform(0, 2 * np.pi)
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
        side = (xs * np.cos(ang) + ys * np.sin(ang) - rng.uniform(-0.3, 0.3)) > 0
        mask = cv2.GaussianBlur(side.astype(np.float32), (0, 0), rng.uniform(1, 8))
        img = img * (1 - mask * rng.uniform(0.2, 0.55))[..., None]
    img = img * (1 + rng.uniform(-cfg.color_cast, cfg.color_cast, 3))
    img = np.clip(img, 0, 1).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # float: H in [0, 360)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_shift_deg, cfg.hue_shift_deg)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * U(rng, cfg.saturation), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def occlude(rng, img, q, cfg: AugmentConfig):
    """Pole, hand-like blob or rectangle over the flag, limited to ~occlusion_max_frac of its bbox."""
    S = img.shape[0]
    x0, y0 = np.clip(q.min(0), 0, S - 1)
    x1, y1 = np.clip(q.max(0), 0, S - 1)
    bw, bh = max(x1 - x0, 4), max(y1 - y0, 4)
    max_area = cfg.occlusion_max_frac * bw * bh
    mask = np.zeros((S, S), np.float32)
    kind = rng.integers(3)
    if kind == 0:  # pole along the hoist edge (or a random near-vertical line)
        p0, p1 = (q[0], q[3]) if rng.random() < 0.7 else (q[1], q[2])
        d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-6)
        a, b = p0 - d * S * 2, p1 + d * S * 2
        thick = int(rng.integers(2, max(3, int(min(bw, bh) * 0.08))))
        cv2.line(mask, tuple(int(v) for v in a), tuple(int(v) for v in b), 1.0, thick)
        color = np.array([rng.uniform(0.3, 0.8)] * 3) * (1 + rng.uniform(-0.1, 0.1, 3))
    elif kind == 1:  # hand / head-like blob: a few skin/dark ellipses
        color = np.array([0.85, 0.65, 0.5]) * rng.uniform(0.35, 1.1) if rng.random() < 0.7 else rng.uniform(0, 1, 3)
        r = np.sqrt(max_area / np.pi) * rng.uniform(0.5, 1.0)
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        for _ in range(int(rng.integers(1, 4))):
            center = (int(cx + rng.normal(0, r * 0.4)), int(cy + rng.normal(0, r * 0.4)))
            axes = (int(r * rng.uniform(0.4, 0.8)), int(r * rng.uniform(0.6, 1.0)))
            cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    else:  # rectangle (sign, branch, person...) with random colour
        area = max_area * rng.uniform(0.2, 1.0)
        ar = rng.uniform(0.3, 3.0)
        rw, rh = np.sqrt(area * ar), np.sqrt(area / ar)
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        cv2.rectangle(mask, (int(cx - rw / 2), int(cy - rh / 2)), (int(cx + rw / 2), int(cy + rh / 2)), 1.0, -1)
        color = rng.uniform(0, 1, 3)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.8)[..., None]
    shade = 1 + 0.15 * smooth_noise(rng, S, S)[..., None]
    return img * (1 - mask) + mask * np.clip(color * shade, 0, 1)


def degrade(rng, img, cfg: AugmentConfig):
    """Blur, noise and JPEG. Input float [0,1], output uint8."""
    if rng.random() < cfg.motion_blur_p:
        k = int(rng.integers(*cfg.motion_blur_ksize)) | 1
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kern = cv2.warpAffine(kern, rot, (k, k))
        img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))
    if rng.random() < cfg.gauss_blur_p:
        img = cv2.GaussianBlur(img, (0, 0), U(rng, cfg.gauss_blur_sigma))
    out = img * 255.0
    if rng.random() < cfg.noise_p:
        out = out + rng.normal(0, U(rng, cfg.noise_sigma), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < cfg.jpeg_p:
        q = int(rng.integers(*cfg.jpeg_quality))
        enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)


# ----------------------------------------------------------------------------- top level
def make_background(rng, bg_loader, cfg: AugmentConfig, zoom=(0.3, 1.0)):
    if rng.random() < cfg.procedural_bg_p:
        return procedural_background(rng, cfg.out_size)
    return photo_background(rng, bg_loader(rng), cfg.out_size, zoom)


def composite_cloth(rng, cloth_rgba_uint8, img_bg, cfg: AugmentConfig, aspect=None, allow_flip=True):
    """Warp a cloth (flag or distractor) like a waving flag and paste it onto img_bg.
    Returns (float image, destination quad of the cloth)."""
    S = cfg.out_size
    h0, w0 = cloth_rgba_uint8.shape[:2]
    if aspect is None:
        aspect = w0 / h0
        if rng.random() < cfg.aspect_jitter_p:
            aspect = U(rng, cfg.aspect_range)
    flipped = allow_flip and rng.random() < cfg.hflip_p
    # working resolution ~ the size it will occupy in the frame (avoids aliasing)
    work_h = int(np.clip(S * np.sqrt(cfg.flag_scale[1] / aspect) * 1.3, 32, 512))
    work_w = int(np.clip(work_h * aspect, 32, 1024))
    cloth = cv2.resize(cloth_rgba_uint8, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if flipped:
        cloth = cloth[:, ::-1].copy()
    cloth = cloth_look(rng, cloth, cfg)
    if rng.random() < cfg.cloth_p:
        cloth, pad = cloth_warp(rng, cloth, cfg, flipped)
    else:
        pad = 0
    q = place_flag(rng, cloth, cfg, aspect)
    layer = render_flag_layer(rng, cloth, q, pad, S)
    alpha = cv2.GaussianBlur(layer[..., 3], (0, 0), U(rng, cfg.edge_soften_sigma))[..., None]
    img = img_bg * (1 - alpha) + np.clip(layer[..., :3], 0, 1) * alpha
    if rng.random() < cfg.pole_p:
        p_top, p_bot = (q[1], q[2]) if flipped else (q[0], q[3])
        thick = rng.uniform(0.02, 0.06) * np.linalg.norm(q[3] - q[0]) + 1
        img = draw_pole(rng, img, p_top, p_bot, thick)
    return img, q


def synth_flag_photo(rng, flag_rgba_uint8, bg_loader, cfg: AugmentConfig, shape_loader=None):
    """flag_rgba_uint8: HxWx4 reference raster. bg_loader(rng) -> HxWx3 uint8 photo.
    shape_loader(rng) -> cloth-sim shape dict (see cloth_shapes.py) or None.
    Returns out_size x out_size x 3 uint8."""
    bg = make_background(rng, bg_loader, cfg)
    shape = shape_loader(rng) if (shape_loader is not None and rng.random() < cfg.cloth_sim_p) else None
    if shape is not None:  # v3: flag draped on a Blender-simulated cloth, lit by its HDRI
        layer, k = map_flag(rng, shape, flag_rgba_uint8, upside_down=rng.random() < cfg.cloth_upside_down_p,
                            exposure=cfg.cloth_exposure, gloss_gain=cfg.cloth_gloss_gain)
        img, q = paste_shape(rng, bg, layer, shape, k, U(rng, cfg.flag_scale),
                             cfg.cloth_paste_rotation_deg, cfg.edge_crop_p)
    else:  # procedural sine-wave cloth
        img, q = composite_cloth(rng, flag_rgba_uint8, bg, cfg)
    if rng.random() < cfg.occlusion_p:
        img = occlude(rng, img, q, cfg)
    img = lighting(rng, img, cfg)
    return degrade(rng, img, cfg)


def make_distractor(rng, bg_loader, cfg: AugmentConfig, h=256):
    """A non-flag cloth: plain fabric, awning-style stripes (>=6 alternating), check, or a photo patch."""
    w = int(h * U(rng, cfg.aspect_range))
    kind = rng.integers(4)
    if kind == 0:  # plain fabric
        rgb = np.ones((h, w, 3), np.float32) * rng.uniform(0, 1, 3)
    elif kind == 1:  # awning / towel stripes
        n = int(rng.integers(*cfg.distractor_stripes))
        c = rng.uniform(0, 1, (2, 3))
        vertical = rng.random() < 0.5
        coord = np.arange(w if vertical else h) * n // (w if vertical else h)
        band = c[coord % 2]
        rgb = np.broadcast_to(band[None] if vertical else band[:, None], (h, w, 3)).astype(np.float32)
    elif kind == 2:  # check / plaid
        n = int(rng.integers(4, 14))
        c = rng.uniform(0, 1, (2, 3))
        yy, xx = np.mgrid[0:h, 0:w]
        rgb = np.where(((yy * n // h + xx * n // h) % 2)[..., None] == 1, c[0], c[1]).astype(np.float32)
        if rng.random() < 0.5:  # plaid: blend in the lines instead of hard squares
            rgb = cv2.GaussianBlur(rgb, (0, 0), h / n / 4)
    else:  # patch of an unrelated photo (printed banner, towel, curtain...)
        rgb = photo_background(rng, bg_loader(rng), h)
        rgb = cv2.resize(rgb, (w, h))
    rgb = np.clip(rgb * (1 + 0.1 * smooth_noise(rng, h, w)[..., None]), 0, 1)
    return (np.dstack([rgb, np.ones((h, w), np.float32)]) * 255).astype(np.uint8)


def synth_no_flag(rng, bg_loader, cfg: AugmentConfig):
    """Background-only sample (photo crop, zoomed crop, or procedural), sometimes with a waving
    non-flag cloth and/or a bare pole, with the same photometrics as flag samples."""
    S = cfg.out_size
    zoom = (0.15, 1.0) if rng.random() < cfg.no_flag_crop_p else (0.6, 1.0)
    img = make_background(rng, bg_loader, cfg, zoom)
    q = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32) * rng.uniform(0.3, 1.0)
    if rng.random() < cfg.no_flag_distractor_p:
        img, q = composite_cloth(rng, make_distractor(rng, bg_loader, cfg), img, cfg)
    elif rng.random() < cfg.pole_p:  # bare pole: a pole alone must not mean "flag"
        x = rng.uniform(0.1, 0.9) * S
        top = np.array([x, rng.uniform(-0.2, 0.6) * S])
        img = draw_pole(rng, img, top, top + np.array([rng.uniform(-0.1, 0.1) * S, S]), rng.uniform(2, 8))
    if rng.random() < cfg.occlusion_p:
        img = occlude(rng, img, q, cfg)
    img = lighting(rng, img, cfg)
    return degrade(rng, img, cfg)
