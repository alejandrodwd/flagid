import XCTest
@testable import FlagLog

final class RankingTests: XCTestCase {
    private var catalog: FlagCatalog!

    override func setUpWithError() throws {
        catalog = try FlagCatalog(
            metadata: ["names": #"{"fr": "France", "nl": "Netherlands", "ru": "Russia", "ie/ci": "Ireland / Côte d'Ivoire", "it": "Italy"}"#,
                       "groups": #"{"ie/ci": ["ie", "ci"]}"#])
    }

    func testSortsDescendingWithEntries() {
        let ranked = Ranker(catalog: catalog).rank(["fr": 0.1, "nl": 0.6, "ie/ci": 0.3])
        XCTAssertEqual(ranked.map(\.id), ["nl", "ie/ci", "fr"])
        XCTAssertEqual(ranked[0].probability, 0.6)
        XCTAssertEqual(ranked[1].entry.members, ["ie", "ci"])
        XCTAssertEqual(ranked[1].entry.name, "Ireland / Côte d'Ivoire")
    }

    func testTiesAreBrokenByLabelAndLimitApplies() {
        let ranked = Ranker(catalog: catalog).rank(["ru": 0.25, "fr": 0.25, "nl": 0.25, "it": 0.25], limit: 2)
        XCTAssertEqual(ranked.map(\.id), ["fr", "it"])
        XCTAssertTrue(Ranker(catalog: catalog).rank(["fr": 1], limit: 0).isEmpty)
    }

    func testPriorMultipliesAndRenormalises() {
        var ranker = Ranker(catalog: catalog)
        ranker.prior = { $0 == "fr" ? 4 : 1 }
        let ranked = ranker.rank(["fr": 0.2, "nl": 0.6, "ru": 0.2])
        // weights 0.8, 0.6, 0.2 -> /1.6
        XCTAssertEqual(ranked.map(\.id), ["fr", "nl", "ru"])
        XCTAssertEqual(ranked[0].probability, 0.5, accuracy: 1e-12)
        XCTAssertEqual(ranked.map(\.probability).reduce(0, +), 1, accuracy: 1e-12)
    }

    func testInvalidValuesAndUnknownLabels() {
        let ranked = Ranker(catalog: catalog).rank(["fr": .nan, "nl": -1, "xx": 0.5])
        XCTAssertEqual(ranked.first?.id, "xx")
        XCTAssertEqual(ranked.first?.entry.members, ["xx"])
        XCTAssertTrue(ranked.dropFirst().allSatisfy { $0.probability == 0 })
    }

    func testResultsDisplayThresholdAndCounts() {
        let probs = Dictionary(uniqueKeysWithValues: (0..<12).map { ("x\($0)", 0.01) } + [("fr", 0.5)])
        let ranked = Ranker(catalog: catalog).rank(probs)
        guard case .list(let collapsed) = ResultsDisplay.make(ranked: ranked, expanded: false, threshold: 0.3),
              case .list(let expanded) = ResultsDisplay.make(ranked: ranked, expanded: true, threshold: 0.3) else {
            return XCTFail("expected a list")
        }
        XCTAssertEqual(collapsed.count, AppConfig.collapsedCount)
        XCTAssertEqual(expanded.count, AppConfig.expandedCount)
        XCTAssertEqual(collapsed.first?.id, "fr")

        XCTAssertEqual(ResultsDisplay.make(ranked: ranked, expanded: false, threshold: 0.51), .hint)
        XCTAssertEqual(ResultsDisplay.make(ranked: [], expanded: false), .hint)
        let low = Ranker(catalog: catalog).rank(["fr": 0.29, "nl": 0.28, "ru": 0.28])
        XCTAssertEqual(ResultsDisplay.make(ranked: low, expanded: false, threshold: 0.3), .hint)
    }
}
