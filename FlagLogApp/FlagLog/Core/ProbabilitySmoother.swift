import Foundation

/// Exponential moving average of probability dictionaries over time.
///
/// The weight of the newest frame depends on the time since the previous frame,
/// `alpha = 1 - exp(-dt / timeConstant)`, so the smoothing is the same at any frame rate.
/// The averaged values still sum to 1 when every input does.
struct ProbabilitySmoother {
    let timeConstant: TimeInterval
    private(set) var current: [String: Double] = [:]
    private var lastTime: TimeInterval?

    init(timeConstant: TimeInterval = AppConfig.smoothingTimeConstant) {
        precondition(timeConstant > 0)
        self.timeConstant = timeConstant
    }

    /// Adds a frame taken at `time` (seconds, monotonic) and returns the smoothed probabilities.
    @discardableResult
    mutating func update(_ probs: [String: Double], at time: TimeInterval) -> [String: Double] {
        guard let lastTime, !current.isEmpty else {
            current = probs
            self.lastTime = time
            return current
        }
        let dt = max(0, time - lastTime)
        let alpha = 1 - exp(-dt / timeConstant)
        var next = [String: Double](minimumCapacity: max(current.count, probs.count))
        for (label, old) in current {
            next[label] = old + alpha * ((probs[label] ?? 0) - old)
        }
        for (label, p) in probs where current[label] == nil {
            next[label] = alpha * p
        }
        current = next
        self.lastTime = time
        return current
    }

    /// Forgets the history, e.g. when the crop or the source changes.
    mutating func reset() {
        current = [:]
        lastTime = nil
    }
}
