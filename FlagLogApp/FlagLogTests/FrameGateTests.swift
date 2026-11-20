import XCTest
@testable import FlagLog

final class FrameGateTests: XCTestCase {
    func testDropsWhileBusyAndTooSoon() {
        let gate = FrameGate(fps: 10)
        XCTAssertTrue(gate.tryEnter(at: 0))
        XCTAssertFalse(gate.tryEnter(at: 0.5), "busy")
        gate.leave()
        XCTAssertTrue(gate.tryEnter(at: 0.5))
        gate.leave()
        XCTAssertFalse(gate.tryEnter(at: 0.55), "too soon")
        XCTAssertTrue(gate.tryEnter(at: 0.6))
    }

    /// The app's settings: a 24 fps camera feeds the model about 5 times a second.
    func testAppCameraRateGivesAboutFiveModelRunsPerSecond() {
        let gate = FrameGate(fps: AppConfig.inferenceFPS)
        var accepted = 0
        for i in 0..<Int(AppConfig.cameraFPS * 4) {  // 4 s of camera frames
            if gate.tryEnter(at: Double(i) / AppConfig.cameraFPS) { accepted += 1; gate.leave() }
        }
        XCTAssertEqual(Double(accepted) / 4, AppConfig.inferenceFPS, accuracy: 0.5)
    }

    func testThirtyFpsCameraGivesAboutTwelveFps() {
        let gate = FrameGate(fps: 12)
        var accepted = 0
        for i in 0..<60 {  // 2 s of 30 fps frames, each finishing immediately
            if gate.tryEnter(at: Double(i) / 30) { accepted += 1; gate.leave() }
        }
        XCTAssertTrue((20...30).contains(accepted), "\(accepted)")
    }
}
