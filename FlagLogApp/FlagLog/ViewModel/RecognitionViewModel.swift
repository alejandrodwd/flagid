import Observation
import SwiftUI
import UIKit

/// State for the single recognition screen: live camera or a picked photo, the ranked list,
/// tap-to-focus and the frozen detail view.
@MainActor
@Observable
final class RecognitionViewModel {
    enum Source: Equatable {
        case camera
        case photo(UIImage)   // upright, downscaled
    }

    struct Frozen: Identifiable {
        let flag: RankedFlag
        /// Still frame shown behind the detail sheet (camera only).
        let image: UIImage?
        var id: String { flag.id }
    }

    // MARK: Observable state

    private(set) var source: Source = .camera
    private(set) var cameraStatus: CameraService.Status = .notConfigured
    private(set) var loadError: String?
    /// Pixel size of the current frame or photo (upright).
    private(set) var imageSize: CGSize = .zero
    /// Focus crop in normalised top-left image coordinates; nil = full frame.
    private(set) var focusRect: CGRect?
    private(set) var ranked: [RankedFlag] = []
    private(set) var isClassifyingPhoto = false
    var expanded = false
    var frozen: Frozen?

    var display: ResultsDisplay { ResultsDisplay.make(ranked: ranked, expanded: expanded) }
    var hasMoreThanCollapsed: Bool { ranked.count > AppConfig.collapsedCount }

    // MARK: Services

    let camera = CameraService()
    private let classifier: FlagClassifier?
    private let processor: LiveFrameProcessor?
    private var ranker: Ranker?
    private var smoother = ProbabilitySmoother()
    private var generation = 0
    private var photoTask: Task<Void, Never>?
    private var zoomAtPinchStart: CGFloat = 1

    init() {
        do {
            let classifier = try FlagClassifier()
            self.classifier = classifier
            ranker = Ranker(catalog: classifier.catalog)
            let processor = LiveFrameProcessor(classifier: classifier)
            self.processor = processor
            camera.onFrame = { [processor] buffer, time in processor.process(buffer, time: time) }
            processor.onResult = { [weak self] result in
                Task { @MainActor in self?.receive(result) }
            }
        } catch {
            classifier = nil
            processor = nil
            loadError = "Could not load the flag model: \(error.localizedDescription)"
        }
    }

    var catalog: FlagCatalog? { classifier?.catalog }

    // MARK: Lifecycle

    func startCamera() async {
        guard source == .camera, processor != nil else { return }
        cameraStatus = await camera.start()
    }

    func stopCamera() { camera.stop() }

    // MARK: Live results

    private func receive(_ result: LiveFrameProcessor.Result) {
        guard source == .camera, frozen == nil, result.generation == generation, let ranker else { return }
        imageSize = result.imageSize
        let smoothed = smoother.update(result.probabilities, at: result.time)
        ranked = ranker.rank(smoothed, limit: AppConfig.expandedCount)
    }

    // MARK: Tap to focus

    /// A tap on the preview focuses on a crop around the tap; a second tap returns to full frame.
    func handleTap(at point: CGPoint, viewSize: CGSize) {
        guard frozen == nil else { return }
        if focusRect != nil {
            setFocus(nil)
            return
        }
        guard imageSize != .zero,
              let p = CropGeometry.imagePoint(fromView: point, viewSize: viewSize, imageSize: imageSize,
                                              mode: contentMode) else { return }
        setFocus(CropGeometry.focusRect(center: p, imageSize: imageSize))
    }

    func clearFocus() { setFocus(nil) }

    var contentMode: CropGeometry.ContentMode { source == .camera ? .fill : .fit }

    private func setFocus(_ rect: CGRect?) {
        focusRect = rect
        smoother.reset()
        switch source {
        case .camera:
            if let processor { generation = processor.setRegion(rect) }
        case .photo(let image):
            classifyPhoto(image)
        }
    }

    // MARK: Zoom

    func pinchChanged(_ scale: CGFloat) {
        camera.setZoom(zoomAtPinchStart * scale)
    }

    func pinchEnded(_ scale: CGFloat) {
        camera.setZoom(zoomAtPinchStart * scale)
        zoomAtPinchStart = max(1, min(zoomAtPinchStart * scale, AppConfig.maxZoomFactor))
    }

    // MARK: Freeze / detail

    func select(_ flag: RankedFlag) {
        var still: UIImage?
        if source == .camera {
            processor?.setPaused(true)
            still = processor?.latestFrame?.makeUIImage()
        }
        frozen = Frozen(flag: flag, image: still)
    }

    func unfreeze() {
        frozen = nil
        if source == .camera { processor?.setPaused(false) }
    }

    // MARK: Photo picker

    func usePhoto(data: Data) {
        guard let image = UIImage(data: data)?.uprightDownscaled(maxSide: AppConfig.maxPhotoSide) else {
            loadError = "That file could not be read as an image."
            return
        }
        loadError = classifier == nil ? loadError : nil
        camera.stop()
        processor?.setPaused(true)
        source = .photo(image)
        imageSize = CGSize(width: image.cgImage?.width ?? 0, height: image.cgImage?.height ?? 0)
        ranked = []
        expanded = false
        setFocus(nil)
    }

    func returnToCamera() {
        photoTask?.cancel()
        source = .camera
        ranked = []
        imageSize = .zero
        isClassifyingPhoto = false
        setFocus(nil)
        processor?.setPaused(false)
        Task { await startCamera() }
    }

    private func classifyPhoto(_ image: UIImage) {
        guard let classifier, let ranker, let cgImage = image.cgImage else { return }
        photoTask?.cancel()
        isClassifyingPhoto = true
        let region = focusRect
        photoTask = Task {
            let probs = await Task.detached(priority: .userInitiated) {
                try? classifier.classify(cgImage: cgImage, region: region)
            }.value
            guard !Task.isCancelled, case .photo(let current) = source, current === image, focusRect == region else { return }
            isClassifyingPhoto = false
            ranked = ranker.rank(probs ?? [:], limit: AppConfig.expandedCount)
        }
    }
}

extension UIImage {
    /// Redraws the image upright (orientation .up) with its long side at most `maxSide` pixels.
    func uprightDownscaled(maxSide: CGFloat) -> UIImage {
        let pixelSize = CGSize(width: size.width * scale, height: size.height * scale)
        let factor = min(1, maxSide / max(pixelSize.width, pixelSize.height))
        let target = CGSize(width: (pixelSize.width * factor).rounded(), height: (pixelSize.height * factor).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: target, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
    }
}
