import Foundation

/// Lets a frame through only if the previous one has finished and enough time has passed.
/// Thread-safe; frames that are refused are simply dropped.
final class FrameGate: @unchecked Sendable {
    let minInterval: TimeInterval
    private let lock = NSLock()
    private var busy = false
    private var lastStart: TimeInterval?

    init(fps: Double = AppConfig.inferenceFPS) {
        minInterval = fps > 0 ? 1 / fps : 0
    }

    /// Call with the frame time; if it returns true, call `leave()` when the frame is done.
    func tryEnter(at time: TimeInterval) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if busy { return false }
        // Small tolerance so camera frames that arrive a hair early still count (24 fps camera -> 5 fps model).
        if let lastStart, time - lastStart < minInterval * 0.9 { return false }
        busy = true
        lastStart = time
        return true
    }

    func leave() {
        lock.lock(); busy = false; lock.unlock()
    }
}
