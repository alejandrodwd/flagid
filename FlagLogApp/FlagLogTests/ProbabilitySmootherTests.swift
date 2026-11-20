import XCTest
@testable import FlagLog

final class ProbabilitySmootherTests: XCTestCase {
    func testFirstFrameIsPassedThrough() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        XCTAssertEqual(s.update(["a": 0.7, "b": 0.3], at: 10), ["a": 0.7, "b": 0.3])
    }

    func testMovesTowardNewFrameByAlpha() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        s.update(["a": 1, "b": 0], at: 0)
        let out = s.update(["a": 0, "b": 1], at: 0.1)
        let alpha = 1 - exp(-0.1 / 0.25)
        XCTAssertEqual(out["a"]!, 1 - alpha, accuracy: 1e-12)
        XCTAssertEqual(out["b"]!, alpha, accuracy: 1e-12)
    }

    func testSumStaysOneAndNewLabelsAreAdded() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        s.update(["a": 0.6, "b": 0.4], at: 0)
        let out = s.update(["b": 0.5, "c": 0.5], at: 0.08)
        XCTAssertEqual(out.values.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertGreaterThan(out["c"]!, 0)
        XCTAssertLessThan(out["c"]!, 0.5)
    }

    func testIndependentOfFrameRate() {
        var slow = ProbabilitySmoother(timeConstant: 0.25)
        var fast = ProbabilitySmoother(timeConstant: 0.25)
        slow.update(["a": 1, "b": 0], at: 0)
        fast.update(["a": 1, "b": 0], at: 0)
        let target = ["a": 0.0, "b": 1.0]
        let one = slow.update(target, at: 0.2)
        fast.update(target, at: 0.1)
        let two = fast.update(target, at: 0.2)
        XCTAssertEqual(one["b"]!, two["b"]!, accuracy: 1e-12)
    }

    func testSettlesAfterAboutHalfASecond() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        s.update(["a": 1, "b": 0], at: 0)
        var out: [String: Double] = [:]
        for i in 1...6 { out = s.update(["a": 0, "b": 1], at: Double(i) / 12) }  // 0.5 s at 12 fps
        XCTAssertGreaterThan(out["b"]!, 0.85)
        XCTAssertGreaterThan(out["b"]!, out["a"]!)
    }

    func testSingleOutlierFrameDoesNotFlipTheTop() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        for i in 0..<6 { s.update(["fr": 0.8, "nl": 0.2], at: Double(i) / 12) }
        let out = s.update(["fr": 0.1, "nl": 0.9], at: 6.0 / 12)
        XCTAssertGreaterThan(out["fr"]!, out["nl"]!)
    }

    /// With the app's settings (few model runs a second), one bad result doesn't flip the top
    /// entry, and a real change takes over within about a second.
    func testAppSettingsAtInferenceRate() {
        var s = ProbabilitySmoother()  // AppConfig.smoothingTimeConstant
        let dt = 1 / AppConfig.inferenceFPS
        var t = 0.0
        for _ in 0..<5 { s.update(["fr": 0.8, "nl": 0.2], at: t); t += dt }
        var out = s.update(["fr": 0.1, "nl": 0.9], at: t)
        XCTAssertGreaterThan(out["fr"]!, out["nl"]!, "one outlier must not flip the top")
        for _ in 0..<Int(AppConfig.inferenceFPS) { t += dt; out = s.update(["fr": 0.1, "nl": 0.9], at: t) }
        XCTAssertGreaterThan(out["nl"]!, out["fr"]!, "a real change shows within ~1 s")
    }

    func testResetAndSameTimestamp() {
        var s = ProbabilitySmoother(timeConstant: 0.25)
        s.update(["a": 1], at: 1)
        XCTAssertEqual(s.update(["a": 0, "b": 1], at: 1)["a"], 1)  // dt = 0 changes nothing
        s.reset()
        XCTAssertTrue(s.current.isEmpty)
        XCTAssertEqual(s.update(["b": 1], at: 2), ["b": 1])
    }
}
