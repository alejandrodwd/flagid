import XCTest
@testable import FlagLog

final class FlagCatalogTests: XCTestCase {
    private let metadata = [
        "names": #"{"fr": "France", "jp": "Japan", "ie/ci": "Ireland / Côte d'Ivoire"}"#,
        "groups": #"{"ie/ci": ["ie", "ci"]}"#,
        "com.github.apple.coremltools.version": "9.0",
    ]

    func testParsesNamesAndGroups() throws {
        let catalog = try FlagCatalog(metadata: metadata, classLabels: ["jp", "ie/ci", "fr"])
        XCTAssertEqual(catalog.labels, ["fr", "ie/ci", "jp"])
        XCTAssertEqual(catalog.entry(for: "fr"), FlagEntry(label: "fr", name: "France", members: ["fr"]))
        let group = catalog.entry(for: "ie/ci")
        XCTAssertEqual(group.name, "Ireland / Côte d'Ivoire")
        XCTAssertEqual(group.members, ["ie", "ci"])
        XCTAssertTrue(group.isGroup)
        XCTAssertFalse(catalog.entry(for: "jp").isGroup)
    }

    func testLabelsFallBackToMetadataKeys() throws {
        let catalog = try FlagCatalog(metadata: metadata, classLabels: nil)
        XCTAssertEqual(catalog.labels, ["fr", "ie/ci", "jp"])
    }

    func testGroupLabelWithoutGroupsKeyIsSplit() throws {
        let catalog = try FlagCatalog(metadata: ["names": #"{"td/ro": "Chad / Romania"}"#])
        XCTAssertEqual(catalog.entry(for: "td/ro").members, ["td", "ro"])
    }

    func testMissingNameFallsBackToRegionNameOrCode() throws {
        let catalog = try FlagCatalog(metadata: metadata, classLabels: ["fr", "de", "us-ca"])
        XCTAssertEqual(catalog.entry(for: "de").name, Locale.current.localizedString(forRegionCode: "DE"))
        XCTAssertEqual(catalog.entry(for: "us-ca").name, "US-CA")
        // Unknown labels at runtime do not crash.
        XCTAssertEqual(catalog.entry(for: "zz-new").members, ["zz-new"])
        XCTAssertEqual(catalog.name(forCode: "ci"), Locale.current.localizedString(forRegionCode: "CI"))
    }

    func testInvalidJSONThrows() {
        XCTAssertThrowsError(try FlagCatalog(metadata: ["names": "{not json"])) { error in
            XCTAssertEqual(error as? FlagCatalog.ParseError, .invalidJSON(key: "names"))
        }
        XCTAssertThrowsError(try FlagCatalog(metadata: ["names": #"{"fr": "France"}"#, "groups": #"{"a": "b"}"#])) { error in
            XCTAssertEqual(error as? FlagCatalog.ParseError, .invalidJSON(key: "groups"))
        }
    }

    func testNoLabelsThrows() {
        XCTAssertThrowsError(try FlagCatalog(metadata: [:])) { error in
            XCTAssertEqual(error as? FlagCatalog.ParseError, .noLabels)
        }
    }

    /// The shipped model: labels, names and groups all come from its metadata.
    func testBundledModelMetadata() throws {
        let catalog = try FlagClassifier().catalog
        XCTAssertEqual(catalog.count, 191)
        for label in ["ie/ci", "ml/gn", "td/ro", "id/mc"] {
            XCTAssertTrue(catalog.labels.contains(label), label)
            XCTAssertEqual(catalog.entry(for: label).members.count, 2, label)
        }
        XCTAssertEqual(catalog.entry(for: "ie/ci").members, ["ie", "ci"])
        XCTAssertEqual(catalog.entry(for: "ie/ci").name, "Ireland / Côte d'Ivoire")
        XCTAssertEqual(catalog.entry(for: "fr").name, "France")
        XCTAssertFalse(catalog.labels.contains("ie"))
        // Every flag code has artwork in the asset catalog.
        for label in catalog.labels {
            for code in catalog.entry(for: label).members {
                XCTAssertNotNil(UIImage(named: "Flags/\(code)"), "missing flag image for \(code)")
            }
        }
    }
}
