import Foundation

/// One entry the model can output: a single flag (`"fr"`) or a lookalike group (`"ie/ci"`).
struct FlagEntry: Identifiable, Hashable, Sendable {
    /// The model's label, e.g. `"fr"` or `"ie/ci"`.
    let label: String
    /// Display name from the model metadata, e.g. `"Ireland / Côte d'Ivoire"`.
    let name: String
    /// Flag codes shown for this entry: `[label]` for a single flag, the members for a group.
    let members: [String]

    var id: String { label }
    var isGroup: Bool { members.count > 1 }
}

/// Labels, display names and lookalike groups, read from the model's metadata at runtime.
/// Nothing here is hard-coded, so a re-exported model with more flags needs no code change.
struct FlagCatalog: Sendable {
    static let namesKey = "names"
    static let groupsKey = "groups"
    static let groupSeparator: Character = "/"

    enum ParseError: Error, Equatable {
        case invalidJSON(key: String)
        case noLabels
    }

    /// Every label the model can output, sorted.
    let labels: [String]
    private let entries: [String: FlagEntry]
    private let names: [String: String]

    /// - Parameters:
    ///   - metadata: the model's creator-defined metadata (`names` and `groups` are JSON strings).
    ///   - classLabels: the model's class labels, if it is a classifier. When nil, the labels are
    ///     the keys of `names` and `groups`.
    init(metadata: [String: String], classLabels: [String]? = nil) throws {
        let names: [String: String] = try Self.decode(metadata[Self.namesKey], key: Self.namesKey) ?? [:]
        let groups: [String: [String]] = try Self.decode(metadata[Self.groupsKey], key: Self.groupsKey) ?? [:]

        let labels = classLabels ?? Array(Set(names.keys).union(groups.keys))
        guard !labels.isEmpty else { throw ParseError.noLabels }

        self.names = names
        self.labels = labels.sorted()
        var entries: [String: FlagEntry] = [:]
        for label in labels {
            entries[label] = Self.makeEntry(label: label, names: names, groups: groups)
        }
        self.entries = entries
    }

    var count: Int { labels.count }

    /// The entry for a model label. Unknown labels get a fallback entry instead of failing.
    func entry(for label: String) -> FlagEntry {
        entries[label] ?? Self.makeEntry(label: label, names: names, groups: [:])
    }

    /// Display name of one flag code, also for group members that have no entry of their own.
    func name(forCode code: String) -> String {
        names[code] ?? Self.fallbackName(for: code)
    }

    // MARK: - Helpers

    private static func decode<T: Decodable>(_ json: String?, key: String) throws -> T? {
        guard let json else { return nil }
        guard let data = json.data(using: .utf8), let value = try? JSONDecoder().decode(T.self, from: data) else {
            throw ParseError.invalidJSON(key: key)
        }
        return value
    }

    private static func makeEntry(label: String, names: [String: String], groups: [String: [String]]) -> FlagEntry {
        let members: [String]
        if let group = groups[label], !group.isEmpty {
            members = group
        } else if label.contains(groupSeparator) {
            members = label.split(separator: groupSeparator).map(String.init)
        } else {
            members = [label]
        }
        let name = names[label]
            ?? members.map { names[$0] ?? fallbackName(for: $0) }.joined(separator: " / ")
        return FlagEntry(label: label, name: name, members: members)
    }

    /// Localised country name for ISO 3166-1 alpha-2 codes, else the upper-cased code.
    static func fallbackName(for code: String) -> String {
        if code.count == 2, let name = Locale.current.localizedString(forRegionCode: code.uppercased()) {
            return name
        }
        return code.uppercased()
    }
}
