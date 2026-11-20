# Brief: FlagLog iOS app (v0, country flags)

Paste this into a new Claude Code session. `flagid/` is `~/Projects/flaglog/flagid/`; the app
goes next to it in `~/Projects/flaglog/FlagLogApp/`. If the session starts inside `flagid/`,
ask for access to that sibling folder.

## Goal
Build a SwiftUI iPhone app that identifies a flag live from the camera and shows a ranked
list of candidates. The model is finished and exported; this task is only the app around it.

## What already exists (read-only: do not modify anything in `flagid/`)
- **Model:** `flagid/outputs/mnv3_v3/coreml/FlagID.mlpackage` (int8 weights, 4.5 MB,
  ML Program, iOS 16+). Copy it into the Xcode project; do not re-export it.
  - Input `image`: 224×224 RGB. Normalization is inside the model. Use Vision
    (`VNCoreMLRequest`) with `imageCropAndScaleOption = .centerCrop`.
  - Outputs: `classLabel` (string) and `classLabelProbs` (dictionary label → probability,
    softmax, sums to 1). With Vision, read the `VNClassificationObservation` results.
  - **Labels** (191 entries) are lowercase ISO 3166-1 alpha-2 codes (`fr`, `jp`, ...). Four
    lookalike groups are merged into one entry, with codes joined by `/`: `ie/ci`, `ml/gn`,
    `td/ro`, `id/mc`. Their probability is already the sum of both flags.
  - **Model metadata** (`MLModel.modelDescription.metadata[.creatorDefinedKey]`):
    - `names`: JSON label → display name, e.g. `"ie/ci": "Ireland / Côte d'Ivoire"`;
    - `groups`: JSON group label → member codes, e.g. `"ie/ci": ["ie", "ci"]`.
  - Measured: about 0.8 ms per image on the Neural Engine (M1 Pro, from Python); 186 of 189
    ops run on the Neural Engine.
- **Flag artwork for the list:** `flagid/data/flags/world/<iso2>.svg` (195 country flags,
  public domain). Convert them to PNG assets or render the SVGs. A group entry shows both
  members' flags side by side.
- **Accuracy to expect** (126 real photos, 18 countries): with groups merged, top-1 80%,
  top-4 92%, top-10 about 97%. Main failure: the flag is small or far away in the frame.
  Tap-to-focus (cropping around the flag) recovered most of those misses in a test.
- **Confidence:** on photos with no flag, the top-1 probability is below 0.3 for 93% of
  photos and never above 0.5. On flag photos, it is ≥ 0.3 for 81%. The model has no
  "no flag" class, so the app has to decide when to show results.

## v0 features
1. **Live camera** (AVFoundation, back camera), full-screen preview, pinch to zoom.
2. **Live top-4 list** of flag thumbnail, name and probability. A **"More"** button expands it
   to the top 10. Smooth over frames (e.g. an exponential moving average of the probability
   dictionaries over about 0.5 s) so the list doesn't flicker. Run inference at about
   10–15 fps, off the main thread, and drop frames while busy.
3. **Low-confidence state:** when the smoothed top-1 probability is below 0.3, show a hint
   like "Point the camera at a flag" instead of the list. Put the threshold in one constant.
4. **Tap to focus:** a tap on the preview runs the model on a square crop centred on the tap
   (side = 40% of the frame's short side; one constant) instead of the full frame. Show the
   crop box, and switch back to full frame on a second tap or a button.
5. **Freeze/pick:** tapping a list entry freezes the frame and shows that flag's detail (name,
   code; both members for a group). No saving or history yet.
6. **Photo picker (debug + simulator):** choose an image from the photo library and classify
   it, with the same list and tap-to-focus. The simulator has no camera, so this is how the
   app is tested there.

## Design for later (don't build, but leave room)
- **Ranking priors:** keep the final ranking in one function `rank(probs) -> [Entry]`, so a
  location or pack prior can later multiply the probabilities without touching the UI.
- **Model swap:** the model will be retrained and re-exported with more flags (thousands,
  in regional packs). Read labels, names and groups from the model metadata at runtime;
  hard-code no labels. Replacing `FlagID.mlpackage` must be the only change needed.

## Constraints
- Swift, SwiftUI, iOS 17+, no third-party dependencies.
- Everything on device: no network, no analytics, no photos stored or uploaded. Add a
  camera usage description.
- Create the Xcode project in `~/Projects/flaglog/FlagLogApp/`. Keep it simple: a camera
  service, a classifier service (Vision), a view model, and views.
- **Never copy anything from `flagid/data/real_val/`** into the app or its tests (private
  photos). For tests, use renders of the flag SVGs or your own generated images.

## Done when
- It builds and runs in the iOS Simulator. Classifying a picked image of a flag shows the
  right flag in the top 4: test with a few flag PNGs rendered from the SVGs, including
  Ireland, so the `ie/ci` group shows.
- Unit tests cover metadata parsing (names, groups), the smoothing, and `rank`.
- A short `FlagLogApp/README.md` says how to build it, where the model comes from, and how
  to swap in a re-exported model.
- Live-camera behaviour can only be checked on a real iPhone. Say what still needs testing
  on a device.
