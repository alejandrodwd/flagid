import Foundation

/// Every tunable number in the app lives here.
enum AppConfig {
    /// Name of the Core ML model in the app bundle (`FlagID.mlpackage`, compiled to `FlagID.mlmodelc`).
    static let modelName = "FlagID"

    /// Below this smoothed top-1 probability the app shows a hint instead of the list.
    /// On photos with no flag the model's top-1 is below 0.3 for 93% of photos and never above 0.5.
    static let confidenceThreshold = 0.3

    /// Side of the tap-to-focus crop, as a fraction of the frame's short side.
    static let focusCropFraction = 0.4

    /// Camera frame rate: the preview on screen and the frames offered to the model.
    static let cameraFPS = 24.0

    /// How often the model runs on live frames. The preview is not affected; frames in between,
    /// or that arrive while the model is busy, are not classified.
    static let inferenceFPS = 5.0

    /// Time constant (seconds) of the exponential moving average over model results.
    /// The weight of a result halves every ~0.28 s, so the list reflects roughly the last second
    /// (about 5 results at 5 fps); one odd frame can't flip the top entry.
    static let smoothingTimeConstant = 0.4

    /// Rows shown in the list, collapsed and after "More".
    static let collapsedCount = 4
    static let expandedCount = 10

    /// Upper bound for pinch-to-zoom (the device's own maximum is lower on some phones).
    static let maxZoomFactor = 10.0

    /// Picked photos are downscaled to this long side before classification (the model sees 224 px).
    static let maxPhotoSide = 2048.0
}
