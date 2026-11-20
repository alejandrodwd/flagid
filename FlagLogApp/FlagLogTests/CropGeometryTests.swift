import XCTest
@testable import FlagLog

final class CropGeometryTests: XCTestCase {
    private let portrait = CGSize(width: 1080, height: 1920)

    func testFocusRectIsSquareInPixels() {
        let r = CropGeometry.focusRect(center: CGPoint(x: 0.5, y: 0.5), imageSize: portrait, fraction: 0.4)
        XCTAssertEqual(r.width * portrait.width, 432, accuracy: 1e-9)
        XCTAssertEqual(r.height * portrait.height, 432, accuracy: 1e-9)
        XCTAssertEqual(r.midX, 0.5, accuracy: 1e-9)
        XCTAssertEqual(r.midY, 0.5, accuracy: 1e-9)
    }

    func testFocusRectIsMovedInsideTheImage() {
        let r = CropGeometry.focusRect(center: CGPoint(x: 0.99, y: 0.01), imageSize: portrait, fraction: 0.4)
        XCTAssertEqual(r.maxX, 1, accuracy: 1e-9)
        XCTAssertEqual(r.minY, 0, accuracy: 1e-9)
        XCTAssertEqual(r.width * portrait.width, 432, accuracy: 1e-9)
    }

    func testVisionRegionFlipsY() {
        let r = CropGeometry.visionRegion(fromTopLeft: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4))
        XCTAssertEqual(r.minX, 0.1, accuracy: 1e-12)
        XCTAssertEqual(r.minY, 0.4, accuracy: 1e-12)
        XCTAssertEqual(r.width, 0.3, accuracy: 1e-12)
        XCTAssertEqual(r.height, 0.4, accuracy: 1e-12)
    }

    func testFillMappingCropsTheSides() {
        // 1080x1920 image filling a 400x600 view: scaled to 400 wide? No: fill uses the larger
        // scale, max(400/1080, 600/1920) = 0.3704, so the image is 400x711 and the top is cut.
        let view = CGSize(width: 400, height: 600)
        let frame = CropGeometry.displayedImageFrame(viewSize: view, imageSize: portrait, mode: .fill)
        XCTAssertEqual(frame.width, 400, accuracy: 1e-6)
        XCTAssertLessThan(frame.minY, 0)
        let center = CropGeometry.imagePoint(fromView: CGPoint(x: 200, y: 300), viewSize: view, imageSize: portrait, mode: .fill)!
        XCTAssertEqual(center.x, 0.5, accuracy: 1e-9)
        XCTAssertEqual(center.y, 0.5, accuracy: 1e-9)
    }

    func testFitMappingRoundTripAndOutside() {
        let view = CGSize(width: 400, height: 900)
        let image = CGSize(width: 1200, height: 1600)  // drawn 400x533, letterboxed
        XCTAssertNil(CropGeometry.imagePoint(fromView: CGPoint(x: 200, y: 50), viewSize: view, imageSize: image, mode: .fit))
        let p = CropGeometry.imagePoint(fromView: CGPoint(x: 300, y: 450), viewSize: view, imageSize: image, mode: .fit)!
        XCTAssertEqual(p.x, 0.75, accuracy: 1e-9)
        let rect = CropGeometry.focusRect(center: p, imageSize: image)
        let back = CropGeometry.viewRect(fromImage: rect, viewSize: view, imageSize: image, mode: .fit)
        XCTAssertEqual(back.midX, 300, accuracy: 1e-6)
        XCTAssertEqual(back.midY, 450, accuracy: 1e-6)
        XCTAssertEqual(back.width, back.height, accuracy: 1e-6)
    }
}
