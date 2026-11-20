import CoreVideo
import Foundation

/// Classifies camera frames off the main thread at about `AppConfig.inferenceFPS`,
/// dropping frames while the model is busy.
final class LiveFrameProcessor: @unchecked Sendable {
    struct Result: Sendable {
        let probabilities: [String: Double]
        let time: TimeInterval
        let imageSize: CGSize
        /// Settings generation the frame was classified with; stale results are ignored.
        let generation: Int
    }

    private let classifier: FlagClassifier
    private let gate = FrameGate()
    private let queue = DispatchQueue(label: "flaglog.inference", qos: .userInitiated)
    private let lock = NSLock()
    private var region: CGRect?
    private var paused = false
    private var generation = 0
    private var latestBuffer: CVPixelBuffer?

    /// Called on a background queue.
    var onResult: (@Sendable (Result) -> Void)?

    init(classifier: FlagClassifier) {
        self.classifier = classifier
    }

    /// Sets the focus crop (nil = full frame) and returns the new generation.
    @discardableResult
    func setRegion(_ region: CGRect?) -> Int {
        lock.lock(); defer { lock.unlock() }
        self.region = region
        generation += 1
        return generation
    }

    func setPaused(_ paused: Bool) {
        lock.lock(); self.paused = paused; lock.unlock()
    }

    /// The last frame that was classified.
    var latestFrame: CVPixelBuffer? {
        lock.lock(); defer { lock.unlock() }
        return latestBuffer
    }

    func process(_ buffer: CVPixelBuffer, time: TimeInterval) {
        lock.lock()
        let paused = paused, region = region, generation = generation
        lock.unlock()
        guard !paused, gate.tryEnter(at: time) else { return }
        queue.async { [self] in
            defer { gate.leave() }
            guard let probs = try? classifier.classify(pixelBuffer: buffer, region: region) else { return }
            lock.lock(); latestBuffer = buffer; lock.unlock()
            onResult?(Result(probabilities: probs, time: time, imageSize: buffer.size, generation: generation))
        }
    }
}
