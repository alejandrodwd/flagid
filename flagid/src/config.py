"""All hyperparameters and paths live here. Nothing else should hard-code numbers."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    data: Path = ROOT / "data"
    flags_svg: Path = ROOT / "data" / "flags_svg"
    flags_png: Path = ROOT / "data" / "flags_png"          # rasterized cache
    backgrounds: Path = ROOT / "data" / "backgrounds"
    real_val: Path = ROOT / "data" / "real_val"
    cloth_shapes: Path = ROOT / "data" / "cloth_shapes"   # converted Blender cloth-sim shapes (.npz)
    labels: Path = ROOT / "data" / "labels.json"
    downloads: Path = ROOT / "data" / "_downloads"
    outputs: Path = ROOT / "outputs"          # per-run folder: outputs/<backbone>/


@dataclass
class DataConfig:
    raster_width: int = 512           # SVG -> PNG width
    num_backgrounds: int = 3000       # how many COCO images to copy into data/backgrounds
    background_max_side: int = 640    # backgrounds are downscaled to this on copy
    no_flag_label: str = "no_flag"
    # v3: no_flag removed from the classifier. The app goal is "the right flag is in the live
    # top-4 list", and users point the camera at real flags; no_flag was the main error source.
    include_no_flag: bool = False
    # Lookalike groups: flags that can't be told apart even in a good photo (identical up to a
    # mirror image, the aspect ratio, or a colour shade smaller than real lighting variation).
    # They stay separate classes in training; evaluation and the Core ML export add up their
    # probabilities into one entry (label "ie/ci", name "Ireland / Côte d'Ivoire").
    # Flags that differ by an emblem (eg/ye, au/nz, ...) are NOT grouped: a closer photo resolves them.
    lookalike_groups: tuple = (("ie", "ci"), ("ml", "gn"), ("td", "ro"), ("id", "mc"))


@dataclass
class AugmentConfig:
    out_size: int = 224
    # flag geometry
    aspect_range: tuple = (1.0, 2.0)          # width / height of the flag cloth
    aspect_jitter_p: float = 0.7              # prob. of replacing the true aspect with a random one
    # cloth warp (v2: deeper folds + stronger shading after real-photo eval showed a sim-to-real gap)
    cloth_p: float = 0.9
    wave_amplitude_frac: tuple = (0.02, 0.12)  # displacement amplitude as fraction of flag height
    wave_count: tuple = (0.5, 3.5)             # number of wave periods across the flag width
    shading_strength: tuple = (0.3, 0.8)       # how much slope/depth darkens the cloth
    shade_clip: tuple = (0.2, 1.6)             # min/max multiplicative shading
    fold_p: float = 0.5                        # extra sharp creases (|sin| cusps)
    fold_count: tuple = (2.0, 6.0)
    specular_p: float = 0.5                    # sheen on the slopes facing the light
    specular_strength: tuple = (0.1, 0.35)
    fabric_noise: tuple = (0.01, 0.05)         # fine texture on the cloth (std, 0-1 units)
    flag_color_gain: float = 0.08              # per-channel dye/fading jitter on the cloth only (+-)
    flag_gamma: tuple = (0.8, 1.25)
    edge_soften_sigma: tuple = (0.3, 1.2)      # blur of the flag's alpha edge (px)
    pole_p: float = 0.5                        # flagpole at the hoist edge (also drawn in no_flag)
    # pose
    rotation_deg: float = 25.0
    perspective_strength: float = 0.15         # max corner jitter as fraction of size
    # placement
    flag_scale: tuple = (0.08, 1.00)           # fraction of frame area covered (v2: spec said 0.20; real
                                               # photos had distant flags well below that)
    edge_crop_p: float = 0.25                  # prob. of pushing flag partly out of frame
    # photometric
    brightness: tuple = (0.6, 1.3)
    contrast: tuple = (0.7, 1.3)
    gradient_light_p: float = 0.5
    shadow_p: float = 0.3
    color_cast: float = 0.06                   # per-channel gain jitter (+-)
    hue_shift_deg: float = 4.0                 # keep small: hue defines the class
    saturation: tuple = (0.75, 1.15)
    # degradations
    motion_blur_p: float = 0.2
    motion_blur_ksize: tuple = (3, 11)
    gauss_blur_p: float = 0.3
    gauss_blur_sigma: tuple = (0.3, 1.8)
    noise_p: float = 0.4
    noise_sigma: tuple = (2.0, 10.0)          # in 0-255 units
    jpeg_p: float = 0.5
    jpeg_quality: tuple = (30, 95)
    # occlusion
    occlusion_p: float = 0.35
    occlusion_max_frac: float = 0.30
    # flips (simulate seeing the flag from behind); rare by design
    hflip_p: float = 0.05
    # v3: cloth-sim shapes (Blender). The rest of the samples use the procedural sine-wave warp above,
    # kept for extra variety.
    cloth_sim_p: float = 0.75                  # share of flag samples drawn on a simulated shape
    cloth_upside_down_p: float = 0.05          # design rotated 180 deg on the cloth
    cloth_exposure: tuple = (0.75, 1.05)       # bright cloth (95th pct) lands here before photometrics
    cloth_gloss_gain: tuple = (0.4, 1.3)
    cloth_paste_rotation_deg: float = 12.0     # camera roll when pasting the rendered shape
    # background
    procedural_bg_p: float = 0.3               # vs. photo background
    # no_flag class
    no_flag_crop_p: float = 0.5                # random crop zoom for no_flag samples
    no_flag_fraction: float = 0.5              # no_flag samples per epoch relative to one country
    no_flag_distractor_p: float = 0.5          # paste a non-flag cloth (plain/awning stripes/check/photo)
    distractor_stripes: tuple = (6, 17)        # >=6 alternating 2-colour stripes: no national flag
                                               # looks like this without a canton/emblem


@dataclass
class TrainConfig:
    backbone: str = "mobilenetv3_large_100"   # also try "efficientnet_b0", "vit_small_patch14_dinov2"
    pretrained: bool = True
    dropout: float = 0.2
    epochs: int = 30
    samples_per_class_per_epoch: int = 200
    batch_size: int = 128
    lr_head: float = 1e-3
    lr_backbone: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True                           # tried on mps, falls back to fp32 if it fails
    num_workers: int = 8
    seed: int = 0
    val_seed: int = 12345                      # synthetic val split uses a different seed
    val_samples_per_class: int = 20


@dataclass
class ExportConfig:
    min_ios: str = "iOS16"
    parity_images: int = 100
    parity_min_agreement: float = 0.99
    max_top1_drop: float = 0.01
    # parity counts only images where PyTorch's top-1 leads its top-2 by at least this much:
    # on near-ties fp16 rounding legitimately flips the order (not a conversion error)
    parity_min_margin: float = 0.05
    keep_variants: bool = False   # keep FlagID_<variant>.mlpackage files next to the chosen FlagID.mlpackage


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


CFG = Config()

# ImageNet normalization (used by training and baked into the Core ML model)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
