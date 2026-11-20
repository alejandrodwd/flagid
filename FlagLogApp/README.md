# FlagLog (iOS app, v0)

A SwiftUI iPhone app that recognises country flags. It shows a live, ranked list from the
camera, or from a picked photo. Everything runs on the device: no network, no analytics, and
no photos are stored.

iOS 17+, Swift/SwiftUI, no third-party dependencies.

## Build and run

```bash
open FlagLog.xcodeproj
```

Pick the `FlagLog` scheme. To run on an iPhone, set your team under *Signing & Capabilities*
and, if needed, change the bundle id `com.flaglog.FlagLog`.

From the command line (simulator build and unit tests):

```bash
xcodebuild test -project FlagLog.xcodeproj -scheme FlagLog -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

**Simulator:** it has no camera, so the app says so and you test it with **Photo**. To put the
test images into the simulator's photo library:

```bash
xcrun simctl addmedia booted TestImages/*.jpg
```

## Using it

- **Live list:** the top 4 entries with flag, name and probability. **More** shows the top 10.
  The camera preview runs at 24 fps; the model runs on about 5 frames a second, and its results
  are averaged over about the last second so the list doesn't flicker.
- **Low confidence:** if the top entry is below 30%, a hint replaces the list.
- **Tap to focus:** tap the flag to classify only a square around it (40% of the frame's short
  side). Tap again, or press **Full frame**, to go back. This helps most when the flag is small.
- **Pinch** to zoom the camera.
- **Tap an entry** to freeze the frame and see the flag's details. A lookalike group such as
  Ireland / Côte d'Ivoire shows both flags.
- **Photo** classifies a picture from the library, with the same list and tap-to-focus.
  **Camera** goes back to the live view.

## Layout

| Path | What it is |
|---|---|
| `FlagLog/Core/` | Pure logic with unit tests: `FlagCatalog` (labels, names and groups read from the model metadata), `ProbabilitySmoother` (time-based EMA), `Ranker.rank` (the one ranking function), `CropGeometry` (tap-to-focus maths), `FrameGate` (frame rate and dropping), `AppConfig` (every threshold and constant). |
| `FlagLog/Services/` | `FlagClassifier` (Vision + Core ML), `CameraService` (AVFoundation, back camera, portrait BGRA frames), `LiveFrameProcessor` (model runs off the main thread about 5 times a second; the 24 fps preview never waits for it). |
| `FlagLog/ViewModel/` | `RecognitionViewModel`, the screen's state. |
| `FlagLog/Views/` | SwiftUI views. |
| `FlagLog/Model/FlagID.mlpackage` | The model. Xcode compiles it to `FlagID.mlmodelc`. |
| `FlagLog/Assets.xcassets/Flags/` | Flag PNGs, generated (see below). |
| `FlagLogTests/` | Unit tests, plus classifier tests on the generated images in `Resources/`. |
| `TestImages/` | Photo-like test images for the simulator. |
| `scripts/render_assets.py` | Renders the flag PNGs, the app icon and the test images. |

## Where the model comes from

`FlagLog/Model/FlagID.mlpackage` is copied unchanged from
`../flagid/outputs/mnv3_v3/coreml/FlagID.mlpackage`. It is a MobileNetV3 classifier trained on
synthetic images and exported to Core ML with int8 weights (4.5 MB). See `../flagid/README.md`.

- Input: a 224×224 RGB `image`. Vision scales and centre-crops each frame (`.centerCrop`).
- Output: probabilities for 191 entries. Labels are lowercase ISO 3166-1 alpha-2 codes, plus
  four lookalike groups (`ie/ci`, `ml/gn`, `td/ro`, `id/mc`).
- Metadata (`creatorDefined`): `names` (label → display name) and `groups`
  (group label → member codes), both JSON strings.

## Swapping in a re-exported model

Replace `FlagLog/Model/FlagID.mlpackage` with the new export (same file name) and rebuild.
That is the only change the app needs:

- Labels, names and groups are read from the model at runtime. No label is hard-coded.
- A label that has no name in the metadata gets the system's country name for a two-letter
  code, or the upper-cased code otherwise. A `/` in a label without a `groups` entry is split
  into members.
- If the new model covers flags that have no artwork yet, the list shows a grey placeholder
  with the code. To add artwork, put the SVGs next to the others and re-run the asset script:

```bash
../flagid/.venv/bin/python -B scripts/render_assets.py
```

  The script reads `../flagid/data/flags/world/*.svg` and needs `cairosvg` and Pillow (the
  flagid virtualenv has both). For regional packs, point `FLAG_SVGS` at the new folder.
  Asset names are the flag codes, in the `Flags/` namespace.
- `FlagCatalogTests.testBundledModelMetadata` checks the shipped model: entry count (191),
  groups, and that every code has artwork. Update the count when the model changes.

**Adding a prior later** (location, installed packs): set `Ranker.prior`, a function from label
to weight. `rank` multiplies the probabilities by it and renormalises, and the UI doesn't change.

## Tests

31 unit tests:

- **Metadata parsing:** names, groups, fallbacks, invalid JSON, and the bundled model.
- **Smoothing:** EMA maths, same result at any frame rate, settling time, reset.
- **Ranking:** order, ties, limit, prior, invalid values, the hint threshold, 4 and 10 rows.
- **Crop geometry** and the **frame gate.**
- **Classifier:** the real model on nine test images rendered from the SVGs. In the simulator,
  the eight large flags are all top-1 at 0.88–0.94, and Ireland comes out as the `ie/ci`
  group. The small, off-centre Mexican flag is top-1 at 0.81 on the full frame and 0.91 with
  a focus crop.

The test images are rendered from the public-domain flag SVGs. No real photos are used.

## Still to test on a real iPhone

The simulator has no camera, so none of the following has been run yet:

- The live preview, its portrait orientation, and that the camera frames line up with the
  preview, so the crop box sits where you tapped.
- The 24 fps preview, the ~5 model runs a second, and whether the list reacts fast enough (raise `inferenceFPS` or lower `smoothingTimeConstant` in `AppConfig` if not).
- Pinch to zoom.
- The frozen frame behind the detail sheet.
- The camera permission prompt and the "camera access is off" state.
- Pausing when the app goes to the background, and resuming.
- Accuracy on real flags, and whether the 0.3 hint threshold is right in practice.
