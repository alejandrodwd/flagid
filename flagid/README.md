# flagid — flag recognition model (training → Core ML)

Classifies a photo into one of **195 countries** (193 UN members + observers Holy See `va`
and Palestine `ps`). The `no_flag` class of v1/v2 was removed in v3, and four lookalike pairs
are merged into one entry each in the Core ML export (191 entries). Class identifiers are lowercase
ISO 3166-1 alpha-2 codes (`fr`, `jp`, …). Trained entirely on synthetic photos generated on
the fly from reference flag rasters; evaluated on real photos you supply.

All hyperparameters live in [`src/config.py`](src/config.py).

**Current model:** `outputs/mnv3_v3/best.pt` and its Core ML export
`outputs/mnv3_v3/coreml/FlagID.mlpackage`. Earlier models (v1, v2), the regional top-up and the
cloth-sim test/calibration runs were deleted. The sections about them are kept as history;
their commands still re-create them. Kept data: `data/cloth_raw` (raw Blender output, the
source of `data/cloth_shapes`), `data/cloth_shapes` (used by training), backgrounds, flags,
real photos and `data/_downloads` (COCO, Poly Haven HDRIs, country-flags).

## Setup

```bash
brew install cairo                       # native lib needed by cairosvg
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.12 is used because coremltools does not support 3.14 yet. Torch is pinned to
2.7.0, the newest version coremltools 9.0 is tested against.

## Stage 1: Data

```bash
cd data/_downloads
git clone --depth 1 https://github.com/hampusborgos/country-flags.git
curl -L -O http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
cd ../..
.venv/bin/python src/prepare_data.py flags         # SVGs -> data/flags_svg, PNGs -> data/flags_png, labels.json
.venv/bin/python src/prepare_data.py backgrounds   # 3000 COCO images -> data/backgrounds (max side 640)
.venv/bin/python src/sample_grid.py                # -> outputs/sample_grid.png
```

`data/labels.json` maps class index → `{"iso2", "name"}`. Indices are the sorted ISO codes
0–194, and `no_flag` is index 195.

### Licensing
- **Flags**: [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags)
  (sourced from Wikimedia Commons). The repo states the flags are public domain
  (`"license": "PD"` in package.json). National flags may still have usage
  restrictions in some jurisdictions (e.g. commercial or misleading use).
- **Backgrounds**: [COCO val2017](https://cocodataset.org/#termsofuse) images are Flickr
  photos under various Creative Commons licenses. They are used for training only and are
  not shipped in the app.

### Synthetic generator ([`src/augment.py`](src/augment.py))
Runs inside the `Dataset`, so each epoch gets new random samples. For each sample:
1. Random aspect ratio from 1:1 to 2:1 (70% of the time; otherwise the true ratio).
2. Cloth warp: sinusoidal waves with a diagonal tilt, a second harmonic and smooth noise.
   Amplitude grows from the hoist to the fly end. Shading comes from the slope plus depth,
   so troughs are darker.
3. Perspective corner jitter (±15%) and in-plane rotation (±25°).
4. Paste onto a COCO crop (70%) or a procedural background (30%: sky gradient with clouds,
   coloured noise, or a wall). The flag covers 20–100% of the frame. 25% of samples push
   it partly out of frame.
5. Occlusion (35%): pole along the hoist edge, skin-tone hand/head blobs, or a coloured
   rectangle, up to about 30% of the flag's bounding box.
6. Lighting: brightness, contrast, a linear light gradient, a soft directional shadow,
   ±6% per-channel colour cast, saturation 0.75–1.15, and **hue shift of only ±4°**.
7. Motion blur, gaussian blur, gaussian noise, and JPEG quality 30–95.
8. **Horizontal flip only with p = 0.05**, to simulate seeing the flag from behind. Many
   flags are asymmetric, so flips are kept rare. Set `AugmentConfig.hflip_p = 0` to
   disable them.

`no_flag` samples are background-only (photo crops, some zoomed in down to 15% of the
image, or procedural backgrounds). They get the same occlusion, lighting and degradation
steps. The class is balanced like every other class.

## Stage 2: Training

```bash
.venv/bin/python src/train.py > outputs/train_mobilenetv3.log 2>&1      # ~1.5–2 h on an M1 Pro
.venv/bin/python src/train.py --epochs 1 --samples-per-class 10 --run-name smoke   # 1-minute smoke test
.venv/bin/python src/train.py --backbone efficientnet_b0                # other timm backbones
.venv/bin/python src/train.py --backbone vit_small_patch14_dinov2.lvd142m
```

Outputs go to `outputs/<backbone>/`: `best.pt` (best synthetic-val top-1), `last.pt`,
`history.json` and `training_curves.png`.

- [`src/model.py`](src/model.py): timm backbone (`num_classes=0`), then Dropout(0.2), then
  Linear. The head's input size is measured with a dummy forward pass, because timm's
  `num_features` reports 960 for MobileNetV3 while the pooled output is 1280.
- Loss is CE with label smoothing 0.1. The optimizer is AdamW, with lr 2e-4 for the
  backbone and 1e-3 for the head. The schedule is 1 epoch of linear warmup, then cosine
  decay. 30 epochs × 200 samples/class = 39,200 samples per epoch, batch size 128.
- **fp32 on MPS.** fp16 autocast runs, but torch 2.7's `GradScaler` fails on MPS (it needs
  float64). Without loss scaling, fp16 training is unsafe. `train.py` probes a full
  scaled step at startup and falls back to fp32 automatically.
- The validation used during training is synthetic (a fixed seed, 20 samples/class). It
  overestimates real-photo accuracy; Stage 3 measures that properly.
- Reference run (MobileNetV3-Large, M1 Pro, fp32): about 2 h, 220–480 s per epoch. Synthetic
  val top-1 is 98.1% after epoch 1 and 99.87% at best (epoch 24). The final loss of 0.857
  is the floor set by label smoothing: with 196 classes and ε = 0.1, the minimum CE is
  ≈ 0.85. The synthetic task is saturated, so only real photos can tell backbones or
  augmentation settings apart.
- `train.py` ends with `os._exit()`. On macOS, Python 3.12's multiprocessing resource
  tracker can hang forever at interpreter exit after DataLoader workers have been used.
  Checkpoints are written every epoch, so nothing is lost.

## Stage 3: Evaluation

```bash
.venv/bin/python src/evaluate.py                      # real photos in data/real_val/<iso2>/
.venv/bin/python src/evaluate.py --synthetic          # force the synthetic split (seed val_seed+1)
.venv/bin/python src/evaluate.py --ckpt outputs/efficientnet_b0/best.pt
```

Reports top-1/top-3, per-class top-1, the most confused (true → predicted) pairs, and
calibration (mean confidence when correct vs. wrong). The model always chooses among all
196 classes, even when `real_val` only covers some of them. Results are written to
`outputs/<run>/eval_<source>/`: `report.json`, `predictions.csv` (top-3 per image) and
`errors.png` (misclassified images). If `data/real_val` is empty, the script evaluates on
synthetic data and prints a prominent warning.

### Result: MobileNetV3, first run (126 real photos, 18 countries × 7)
| | top-1 | top-3 |
|---|---|---|
| synthetic val | 99.9% | – |
| real photos | **41.3%** | 70.6% |
| real photos, `no_flag` excluded at argmax | 59.5% | 71.4% |

70 of the 74 errors were predicted as `no_flag`, often with confidence 0.8–0.95, on clean,
clearly visible flags. Mean confidence was 0.69 when wrong and 0.66 when correct, so the
model is badly calibrated. Diagnosis: a sim-to-real gap. Synthetic flags look flat and
"pasted" (gentle waves, clean colours, no fabric texture), while `no_flag` consists of real
photos. The model partly learned "looks like a real photo → `no_flag`".

With 28 real no-flag photos added to `data/real_val/no_flag/` (154 images in total), the
same model rejected 96.4% of the no-flag photos but called 54% of the flag photos
`no_flag`.

### Generator v2 (the fix)
- **Cloth:** wave amplitude 2–12% of the flag height (was 1–6%), and extra sharp creases
  (`fold_p`). Shading strength is 0.3–0.8 (was 0.15–0.45), and sheen on slopes facing the
  light (`specular_p`).
- **Fabric look:** per-flag dye/fading jitter (±8% gain per channel, gamma 0.8–1.25; hue
  is still only ±4°), fine fabric-texture noise, and soft alpha edges.
- **Flagpole** at the hoist edge (p = 0.5).
- **Scale:** the flag covers **8**–100% of the frame. The spec said 20%, but distant flags in
  real photos were smaller than that.
- **`no_flag`:** half a country's share of samples per epoch (`no_flag_fraction`). Half of
  its samples get a waving non-flag cloth pasted in with the same pipeline as flags
  (plain fabric, awning stripes, check/plaid, or a photo patch). Some get a bare flagpole.
  Striped distractors always have ≥6 alternating stripes in 2 colours, and no national
  flag looks like that without a canton or emblem.
- `evaluate.py` now also reports, for flag photos, top-1 and the share called `no_flag`.
  For no-flag photos, it reports the share correctly rejected. It also shows an
  alternative: ignore the `no_flag` output and show a country only if its confidence is
  above a cutoff (0.3/0.5/0.7/0.9).

### Result: v2 vs v1 (154 real photos: 126 flags in 18 countries + 28 no-flag)
| | v1 | v2 (deleted) |
|---|---|---|
| flag photos top-1 | 41.3% | **63.5%** |
| flag photos top-3 (all 154 photos) | 70.6%* | 85.7% |
| flag photos called `no_flag` | 54.0% | 16.7% |
| no-flag photos rejected | 96.4% | 92.9% (2 of 28 missed) |
| flag top-1 if `no_flag` is ignored | 59.5% | 73.8% |
| mean confidence correct / wrong | 0.74 / 0.69 | 0.75 / 0.44 |

\*v1's 70.6% was measured on the 126 flag photos only, before the no-flag photos were added.

Synthetic val is still 99.8%. `best.pt` and `last.pt` differ by about one photo. The
remaining problems are Ireland and Italy (0/7 each, mostly → `no_flag`, plus `ie → ci` ×2),
France (2/7, scattered low-confidence guesses), and Germany/India/Netherlands/Thailand
(3/7 each). Many of the misses are clean flags against blue sky, so a realism gap remains.
These 154 photos have now guided two rounds of decisions, so treat them as
semi-held-out. A fresh untouched set is needed for a final unbiased number.

Train v2: `.venv/bin/python src/train.py --run-name mnv3_v2`, then
`.venv/bin/python src/evaluate.py --ckpt outputs/mnv3_v2/best.pt`. Generation now
takes about 27 ms per sample in a single process (it was about 14 ms).

## Cloth-simulation shape library (v3 generator)

Blender simulates blank cloths. The flags themselves are mapped on in Python, so the
library doesn't grow with the number of flags.

```bash
B=/Applications/Blender.app/Contents/MacOS/Blender
$B -b --factory-startup --python blender/cloth_sim.py -- --out data/cloth_test --n-sims 12 --frames-per-sim 2 --cams-per-frame 2
.venv/bin/python src/cloth_shapes.py convert data/cloth_test data/cloth_test_shapes
.venv/bin/python src/cloth_preview.py data/cloth_test_shapes      # -> outputs/cloth_preview/{shapes,flags}.png
```

- [`blender/cloth_sim.py`](blender/cloth_sim.py) samples each setup:
  - attachment: sleeve, grommets, or held at two corners;
  - pole angle: 60% at 0–10°, 25% at 30–60°, 15% at 80–90°;
  - wind: 20% zero, otherwise 300–20,000 (log-uniform);
  - fabric stiffness and aspect ratio;
  - HDRI and its rotation;
  - camera: 25% from behind, mostly looking up, focal length 24–85 mm.

  It simulates 110 frames, grabs settled frames, and renders a multilayer EXR for each with
  Combined, Glossy Dir/Ind/Color, and AOVs `uv`, `back`, `cloth` and `pole`. Every
  sampled parameter goes to `params.jsonl`.
  `--override '[{...}]'` fixes parameters for calibration sweeps.
- **Nominal size:** every cloth is simulated at 1.5 m. Blender's wind force scales with
  face area while gravity doesn't, so real-size flags would need re-tuning per size. The
  drape only depends on the ratios of wind, weight and stiffness, so "flag size" is
  covered by sampling those. Wind was calibrated with a sweep: 600–1,500 is half-lifted,
  4,000+ is flying, and 10,000 is flying with ripples.
- [`src/cloth_shapes.py`](src/cloth_shapes.py) converts each EXR into a cropped `.npz` file
  (at most 384 px, about 190 KB). It holds the UV map (un-premultiplied by coverage), the
  coverage masks, the diffuse and specular lighting, and a line fitted to the pole. The
  pole is re-drawn to the frame edge when the shape is pasted smaller than the frame.
  `map_flag()` puts any flag on any shape: remap through the UVs, multiply by the diffuse
  light (Combined − specular), add the specular, and apply a random exposure. Seen from
  behind, the design is mirrored automatically, because back faces use the same UVs.
  Upside-down flags are a 180° UV flip.
- Blender 5 writes **multi-part** EXRs, so read every part. Cycles' glossy *light* passes
  must be multiplied by the Glossy Color pass. Without that, the shine was 10–20× too
  strong and washed out every design.
- **v2 fixes (after reviewing the v1 test; the test shapes have since been deleted):**
  - In Blender 5.3, **a cloth with no collider or effector in the scene ignores gravity**.
    v1 only hung because the pole was a collider. That collider was also the cause of the
    crumpled, wrapped and frozen shapes, because the pinned hoist sits inside its collision
    margin. The pole is now visual only, and a wind field is always present (strength 1
    for "no wind").
  - Stiffer fabric (bending 0.3–5, tension 15–40). v1 crinkled like tissue paper.
  - Tilted poles (>20°) and corner-held flags get calm or light wind only: 50% none,
    otherwise 300–1,500.
  - Camera: elevation −30° to +12° (was −45° to +20°), azimuth ±45° around the
    front/back normal (was ±70°), and 20% of views from behind (was 25%).
  - Attachment: 65% sleeve, 20% grommets, 15% corners.
  - The converter drops shapes whose visible design (share of a 12×8 UV grid that is
    visible) is below 0.5. This is only a safety net for edge-on views: neither it nor a
    fold-edge measure reliably told crumpled shapes from fine ones.
- Test speed on an M1 Pro (Metal): about 5–30 s of simulation per setup plus about 1 s per
  384 px render. The first run compiles GPU kernels (about 3 min, then cached).

### Full shape library and v3 training
```bash
# HDRIs: 48 CC0 Poly Haven 2k .hdr files (clear midday 8, partly cloudy 8, overcast 7,
# morning-afternoon 7, sunrise-sunset 6, urban 5, indoor daylight 4, indoor artificial 2, night 1),
# picked by download count per category -> data/_downloads/polyhaven/hdri (list in chosen.json)
$B -b --factory-startup --python blender/cloth_sim.py -- --out data/cloth_raw --n-sims 400 \
   --frames-per-sim 6 --cams-per-frame 3 --seed 100 --hdri-dir data/_downloads/polyhaven/hdri
.venv/bin/python src/cloth_shapes.py convert data/cloth_raw data/cloth_shapes
.venv/bin/python src/train.py --run-name mnv3_v3
.venv/bin/python src/evaluate.py --ckpt outputs/mnv3_v3/best.pt
```
- **`no_flag` is removed** (`DataConfig.include_no_flag = False`). The app's goal is that
  the right flag appears in a live top-4 list, and users point the camera at real flags.
  `labels.json` now has 195 classes; the old file is kept as `labels_v2_with_no_flag.json`.
  Older checkpoints still evaluate, because they carry their own labels and `no_flag` was
  the last index.
- **Generator v3:** 75% of samples (`cloth_sim_p`) drape the flag on a random simulated
  shape: `map_flag`, then `paste_shape` onto a COCO or procedural background, then the
  same occlusion, lighting and degradation steps as before. The other 25% use the v2
  procedural warp, for variety. 5% of the cloth samples are upside down. Shapes are read
  from disk per sample; the library is too big to cache in RAM.
- **Evaluation:** top-4 is the headline metric; top-1/3/5 and the median rank of the true
  flag are also reported for each country. `errors.png` now shows photos whose flag is
  missing from the top 4. For a model without `no_flag`, the no-flag photos are reported
  separately: the model's top-1 confidence on them versus on flag photos, for picking an
  app-side "not sure" cutoff. Older models with `no_flag` have it removed from the
  ranking.
- v2 baseline under this metric: top-1 73.8%, top-3 84.1%, **top-4 85.7%**, top-5 88.1%.
  Worst countries: France and Italy (3/7 in the top 4, median rank 14) and India (3/7).

### Result: v3 (cloth-sim library) vs v2 (126 real flag photos, 18 countries × 7)
- **Batch:** 400 setups → 5,265 renders in about 2.2 h on an M1 Pro → 5,222 shapes
  (1.4 GB; 43 rejected by the visibility check). Mix: 62% sleeve, 22% grommets,
  16% corners; poles 69% upright, 20% at 30–60°, 11% at 80–90°; wind 32% zero,
  42% light, 26% strong; 20% of renders from behind.
- **Training:** 30 epochs × 39,000 samples, about 4 min per epoch, fp32 on MPS. Best
  synthetic val top-1 is 97.4%.

| `outputs/mnv3_v3/best.pt` | v2 | **v3** |
|---|---|---|
| top-1 | 73.8% | 76.2% |
| top-3 | 84.1% | 87.3% |
| **top-4** | **85.7%** | **91.3%** |
| top-5 | 88.1% | 93.7% |

- **Fixed:** France top-4 43% → 100% (median rank 14 → 2) and Italy 43% → 71% (median
  rank 14 → 1). The Netherlands, Germany and Ethiopia each gained a photo.
- **Still missing from the top 4 (11 photos):**
  - **India (4/7):** small, distant flags on tall poles. The model can't see the
    chakra and is unsure between Niger, Hungary and Seychelles (confidence under 0.3).
    Tap-to-focus (cropping) should help.
  - **Ireland (2/7):** predicted as Ivory Coast with high confidence. Ireland seen from
    behind *is* Ivory Coast, pixel for pixel, so they can't be told apart without
    context.
  - **Others:** Italian bunting (strings of many small flags), a stock photo of the
    Netherlands on a fence (rank 46), one Spanish flag held up inside a car, and one
    Thai flag predicted as the Netherlands.
- Spain and Ireland each lost one photo against v2. With 7 photos per country, a
  1-photo change is noise.
- **No-flag photos (28):** mean top-1 confidence is 0.12, versus 0.62 on flag photos.
  2/28 no-flag photos reach 0.3; 81% of flag photos do. A cutoff of about 0.3 works as
  an app-side "point at a flag" hint.
- **Caveat:** these 126 photos have been looked at for every decision since v1. A fresh
  held-out set would give an unbiased number.

## Stage 4: Core ML export
```bash
.venv/bin/python src/export_coreml.py --ckpt outputs/mnv3_v3/best.pt > outputs/export_coreml.log 2>&1   # ~5 min
```
Writes `outputs/mnv3_v3/coreml/`:
- `FlagID.mlpackage`: the chosen variant, ready to drop into Xcode;
- `export_report.json`: accuracy, parity, size and latency for each variant.

The other variants (`FlagID_{fp16,int8,pal6,pal4}.mlpackage`) are deleted after the
comparison unless `ExportConfig.keep_variants = True`.

**The model:**
- **Format:** ML Program for iOS 16+, traced with `torch.jit.trace`.
- **Input:** `image`, a 224×224 RGB `ImageType` with scale 1/255. The per-channel ImageNet
  mean/std normalization is folded into the graph, because `ImageType` has only one scale
  for all channels, so its own scale/bias can't express ImageNet std exactly.
- **Output:** a Vision classifier with 191 entries: the 195 countries, with the four
  lookalike groups below each merged into one entry. `classLabel` is an ISO alpha-2 code or
  a group label such as `ie/ci`. `classLabelProbs` holds the softmax probabilities; a
  group's probability is the sum of its members'. `VNCoreMLRequest` returns sorted
  `VNClassificationObservation`s: take the first 4.
- **Metadata (JSON):** `names` maps each label to its display name (`"ie/ci": "Ireland /
  Côte d'Ivoire"`), and `groups` maps each group label to its member codes.
- **In the app:** set `imageCropAndScaleOption = .centerCrop` to match training and
  evaluation.

**Picking a variant.** Each variant is compared with the fp32 PyTorch model on 3,900
synthetic images (seed `val_seed+1`) and on the real photos. The smallest variant wins if
its top-1 drops by less than 1 point on both sets and it passes the parity check:
- **Parity check:** top-1 matches PyTorch on ≥99% of the first 100 real photos.
- **Near-ties are skipped.** Photos where PyTorch's top-2 margin is under 0.05 (9 of 100)
  aren't counted. On those, fp16 rounding legitimately flips the order: every flip seen had
  a margin ≤ 0.017. The raw agreement is still reported.
- **Conversion check:** an fp32 Core ML export must match PyTorch on every photo (it does,
  with a max probability difference of 4e-6). This proves the conversion itself is exact.

### Lookalike groups
Some flags can't be told apart even in a good photo: identical up to a mirror image, the
aspect ratio, or a colour shade smaller than real lighting variation. These are listed in
`DataConfig.lookalike_groups`:

| group | why |
|---|---|
| `ie/ci` Ireland / Côte d'Ivoire | mirror images (identical seen from behind or hanging) |
| `ml/gn` Mali / Guinea | mirror images |
| `td/ro` Chad / Romania | blue shade only |
| `id/mc` Indonesia / Monaco | aspect ratio only |

- **They stay separate classes in training.** The export adds their probabilities
  inside the Core ML graph, and `evaluate.py` reports both strict and grouped top-k. To
  change a group, edit the config and re-export. No retraining is needed.
- **Flags that differ by an emblem are not grouped,** for example Egypt / Yemen (the most
  confused pair on synthetic images: Egypt → Yemen 25%), Australia / New Zealand or Jordan /
  Palestine. A closer photo resolves them, and grouping would hide real errors.
- **Check with data before adding a group:** only pairs still confused on good, close-up
  real photos (more than about a third of them).

### Result (`outputs/mnv3_v3/best.pt`, lookalike groups merged)
| variant | size | synthetic top-1 / top-4 | real top-1 / top-4 | parity (raw) | |
|---|---|---|---|---|---|
| PyTorch fp32, strict (no groups) | — | 97.6% / 99.6% | 76.2% / 91.3% | — | |
| PyTorch fp32, grouped | — | 97.7% / 99.6% | 81.0% / 92.9% | — | |
| fp16 | 8.68 MB | 97.7% / 99.6% | 82.5% / 92.9% | 100% (99%) | ok |
| **int8 weights** | **4.50 MB** | 97.8% / 99.6% | 80.2% / 92.1% | 100% (97%) | **chosen** |
| 6-bit palettized | 3.41 MB | 96.6% / 99.5% | 77.0% / 87.3% | 91% (86%) | rejected |
| 4-bit palettized | 2.34 MB | 25.5% / 49.6% | 16.7% / 32.5% | 21% (19%) | rejected |

Variants are scored on grouped labels. Real photos: 126 photos of 18 countries, so one
photo moves a number by 0.8 points. Grouped top-10 (PyTorch) is 96.8%.

- **Size:** 4.50 MB against a 25 MB budget.
- **Speed:** 0.72–0.83 ms per image on the M1 Pro's Neural Engine (`CPU_AND_NE`, median of
  100 predictions from Python, including Python overhead), against a 10 ms budget.
  CPU only: about 3 ms. Core ML plans 186 of 189 compute ops on the Neural Engine; the
  other 3 are the input scaling and the fp16 casts at the edges.
  iPhone Neural Engines are the same class of hardware, but check on a device with Xcode's
  Core ML Performance Report.
- **Palettization:** iOS 16 only supports one lookup table per tensor, which is too coarse
  for MobileNetV3's depthwise layers. Per-channel-group palettization needs iOS 18 and
  isn't needed at this size.
- **Warnings:** int8 quantization prints "invalid value encountered in divide" for a
  weight with an all-zero channel (scale 0). It is harmless: that channel stays zero, and
  accuracy is unchanged.

## Flag catalog (expansion beyond the 195 countries)
Not used for training yet: the training pipeline still reads `labels.json` and `data/flags_svg`.
```bash
.venv/bin/python src/fetch_flags.py world               # copy the 195 country flags into the catalog
.venv/bin/python src/fetch_flags.py regions es          # a country's regional flags (any ISO alpha-2)
.venv/bin/python src/fetch_flags.py preview es-regions  # check grid -> data/flags/preview/es-regions.png
```
```
data/flags/catalog.csv                one row per flag (columns below)
data/flags/world/<iso2>.svg           countries
data/flags/regional/<iso2>/<id>.svg   first-level regions, e.g. regional/es/es-an.svg
data/flags/preview/<pack>.png         check grids
```
- **Catalog columns:** id, name, category, country, iso_3166_2, wikidata, pack, path,
  source_file, source_url, license, author, attribution_required, lookalike_group.
- **Where regions come from:** Wikidata. The script takes the country's "contains
  administrative territorial entity" items (P150) that have an ISO 3166-2 code (P300) and
  a current flag (P41, best rank). Each SVG and its license and author come from
  Wikimedia Commons. IDs are lowercase ISO 3166-2 codes (`es-an`).
- **Licenses vary**, unlike the country flags. Spain alone has CC BY-SA 2.5–4.0, CC BY 3.0,
  CC0 and public domain. The app will need an attributions screen built from the catalog.
- **Variants:** Wikidata gives one version per region; Asturias, for example, came through
  in its gala version. Add other versions (with or without the coat of arms, for example)
  as extra rows in the same lookalike group.
- **Rate limits:** `upload.wikimedia.org` rate-limits scripted downloads hard. The script
  waits 5 s between files, backs off exponentially, saves the catalog after every file, and
  skips files that already exist. A run can be resumed, and SVGs downloaded by hand into
  `regional/<country>/<id>.svg` are picked up and registered on the next run.
- **Status, Spain:** all 19 downloaded (17 autonomous communities, plus Ceuta and Melilla).
  The last 5 needed a second run after the rate limit.

### Add-a-region test (Spain, 19 flags)
```bash
.venv/bin/python src/fetch_flags.py build-labels world es-regions --out data/labels_world_es.json
.venv/bin/python src/train.py --labels data/labels_world_es.json --init-from outputs/mnv3_v3/best.pt \
    --epochs 6 --run-name mnv3_v3_es                          # top-up: ~33 min on an M1 Pro
.venv/bin/python src/region_test.py --labels data/labels_world_es.json \
    --ckpt outputs/mnv3_v3/best.pt outputs/mnv3_v3_es/best.pt
```
- **Status:** experiment only. The top-up checkpoint `outputs/mnv3_v3_es/` was deleted;
  the commands above re-create it. Plan: perfect the 195-country model first, then add
  regional packs in one full retrain rather than chained top-ups. The Spanish SVGs, catalog
  rows and `data/labels_world_es.json` are kept; the results stay in
  `outputs/region_test_v3_es.json` and `outputs/train_mnv3_v3_es.log`.
- **Label lists:** `build-labels` keeps the 195 countries at their `labels.json` indices and
  appends new flags, rasterized to `data/flags_png/<id>.png`.
- **`--init-from`:** copies the backbone and the head rows of classes the old model knew
  (matched by id); new classes start from scratch.
- **[`src/region_test.py`](src/region_test.py)** scores each model two ways:
  - **head:** the model's own softmax;
  - **fingerprint:** the backbone's 1,280-d feature, L2-normalized, compared by cosine
    similarity with one reference per flag. The reference is the mean feature of 40
    synthetic renders, and computing one needs no training.
  
  Test images: 20 synthetic renders per class (a seed separate from training, validation
  and the references) plus the 126 real country photos. No grouping of identical
  lookalikes.

| model / scoring | Spanish, synthetic (380) top-1 / top-4 / top-10 | countries, synthetic (3,900) top-1 / top-4 | countries, real (126) top-1 / top-4 / top-10 |
|---|---|---|---|
| v3, head (doesn't know Spain) | 0 / 0 / 0 | 97.7% / 99.6% | 76.2% / 91.3% / 96.8% |
| **v3, fingerprint (no retraining)** | **92.4% / 100% / 100%** | 96.6% / 99.4% | 75.4% / 90.5% / 97.6% |
| v3 + 6-epoch top-up, head | 96.8% / 100% / 100% | 97.4% / 99.6% | 74.6% / 88.9% / 96.0% |
| v3 + 6-epoch top-up, fingerprint | 97.4% / 100% / 100% | 96.4% / 99.4% | 75.4% / 92.9% / 95.2% |

- **Adding a region without retraining works, on synthetic images.** A model that never
  saw a Spanish regional flag ranks all 19 in the top 4 from their fingerprints. Scoring
  countries by fingerprint costs ≤1 point against the head.
- **The top-up mainly helps top-1 among lookalikes** (92.4% → 97.4%). The remaining
  top-1 errors are the striped red-and-yellow family: Balearic Islands, Valencia and Aragon
  read as Catalonia. Other errors pair a regional flag with a similar country flag, such as
  Madrid / Türkiye and Castilla-La Mancha / Monaco.
- **Real country photos after the top-up:** head top-4 dropped from 91.3% to 88.9%
  (3 photos), while fingerprint top-4 rose to 92.9%. With 126 photos, ±3 photos is noise,
  but repeated top-ups could slowly drift. Keep `real_val` as a regression test for every
  update, and do a full retrain now and then.
- **Caveat:** no real Spanish photos yet. On countries, synthetic top-1 is about 97% while
  real photos give about 76%, so the Spanish numbers here are an upper bound.

### Design choices not in the spec (simplest option taken)
- Procedural backgrounds are generated on the fly instead of saved to `data/backgrounds/`.
- Training samples use fresh OS entropy, so each epoch differs. The synthetic validation
  split uses a fixed seed (`TrainConfig.val_seed`) for each item, so it is reproducible.
- Real photos are center-cropped to a square, then resized to 224. This matches Vision's
  `.centerCrop` option. HEIC files must be converted to JPEG first.
- COCO has no "flag" category, so a few backgrounds may contain small real flags. This adds
  some label noise to `no_flag`; it was accepted for simplicity.
- Data prep lives in `src/prepare_data.py`, and the grid script is `src/sample_grid.py`.
