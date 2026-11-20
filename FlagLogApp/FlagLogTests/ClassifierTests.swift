import XCTest
@testable import FlagLog

/// Runs the bundled model on photo-like test images rendered from the flag SVGs
/// (scripts/render_assets.py). No real photos are used.
final class ClassifierTests: XCTestCase {
    private static var classifier: FlagClassifier!

    override class func setUp() {
        super.setUp()
        classifier = try? FlagClassifier()
    }

    private func image(_ name: String) throws -> UIImage {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: name, withExtension: "jpg"), name)
        return try XCTUnwrap(UIImage(contentsOfFile: url.path)).uprightDownscaled(maxSide: AppConfig.maxPhotoSide)
    }

    private func top(_ name: String, region: CGRect? = nil, limit: Int = 4) throws -> [RankedFlag] {
        let classifier = try XCTUnwrap(Self.classifier)
        let cg = try XCTUnwrap(try image(name).cgImage)
        let probs = try classifier.classify(cgImage: cg, region: region)
        XCTAssertEqual(probs.count, classifier.catalog.count)
        XCTAssertEqual(probs.values.reduce(0, +), 1, accuracy: 1e-2)
        return Ranker(catalog: classifier.catalog).rank(probs, limit: limit)
    }

    func testFlagImagesAreInTopFour() throws {
        let cases = ["ireland": "ie/ci", "france": "fr", "japan": "jp", "brazil": "br", "germany": "de",
                     "canada": "ca", "south_africa": "za", "nepal": "np"]
        var top1 = 0
        for (file, label) in cases.sorted(by: { $0.key < $1.key }) {
            let ranked = try top(file)
            print("\(file): " + ranked.map { "\($0.id) \(String(format: "%.2f", $0.probability))" }.joined(separator: ", "))
            XCTAssertTrue(ranked.contains { $0.id == label }, "\(file): \(label) not in top 4")
            if ranked.first?.id == label { top1 += 1 }
        }
        XCTAssertGreaterThanOrEqual(top1, cases.count - 1, "top-1 hits")
    }

    func testIrelandShowsTheLookalikeGroup() throws {
        let first = try XCTUnwrap(try top("ireland").first)
        XCTAssertEqual(first.id, "ie/ci")
        XCTAssertEqual(first.entry.members, ["ie", "ci"])
        XCTAssertGreaterThanOrEqual(first.probability, AppConfig.confidenceThreshold)
    }

    /// The Mexican flag is small and off-centre; a focus crop on it finds it.
    func testFocusCropFindsSmallFlag() throws {
        let img = try image("mexico_small")
        let size = CGSize(width: img.cgImage!.width, height: img.cgImage!.height)
        // Flag centre in the 1200x1600 render: x ≈ 816 + 147, y ≈ 352 + 84.
        let rect = CropGeometry.focusRect(center: CGPoint(x: 963.0 / 1200, y: 436.0 / 1600), imageSize: size)
        let cropped = try top("mexico_small", region: rect)
        print("mexico focus: " + cropped.map { "\($0.id) \(String(format: "%.2f", $0.probability))" }.joined(separator: ", "))
        XCTAssertEqual(cropped.first?.id, "mx")
        let full = try top("mexico_small", limit: 1)
        print("mexico full frame: \(full.map { "\($0.id) \($0.probability)" })")
        XCTAssertGreaterThan(cropped[0].probability, full[0].id == "mx" ? full[0].probability : 0)
    }
}
