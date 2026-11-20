import Foundation

/// An entry with its (possibly prior-weighted) probability.
struct RankedFlag: Identifiable, Hashable, Sendable {
    let entry: FlagEntry
    let probability: Double
    var id: String { entry.id }
}

/// Turns model probabilities into the ranked list the UI shows.
/// This is the one place where the final ranking is decided.
struct Ranker: Sendable {
    let catalog: FlagCatalog

    /// Optional prior per label (e.g. from location or installed flag packs). When set, each
    /// probability is multiplied by `prior(label)` and the result is renormalised to sum to 1.
    /// Not used yet; the UI does not need to change when it is added.
    var prior: (@Sendable (String) -> Double)?

    init(catalog: FlagCatalog, prior: (@Sendable (String) -> Double)? = nil) {
        self.catalog = catalog
        self.prior = prior
    }

    /// Sorted by probability (highest first; ties by label), at most `limit` entries.
    func rank(_ probs: [String: Double], limit: Int = .max) -> [RankedFlag] {
        var weighted: [(label: String, p: Double)] = []
        weighted.reserveCapacity(probs.count)
        for (label, p) in probs {
            var w = p.isFinite ? max(0, p) : 0
            if let prior { w *= max(0, prior(label)) }
            weighted.append((label, w))
        }
        if prior != nil {
            let total = weighted.reduce(0) { $0 + $1.p }
            if total > 0 { weighted = weighted.map { ($0.label, $0.p / total) } }
        }
        weighted.sort { $0.p != $1.p ? $0.p > $1.p : $0.label < $1.label }
        return weighted.prefix(max(0, limit)).map {
            RankedFlag(entry: catalog.entry(for: $0.label), probability: $0.p)
        }
    }
}

/// What the results panel shows.
enum ResultsDisplay: Equatable {
    case hint
    case list([RankedFlag])

    /// Hint when there is nothing or the top-1 probability is below `threshold`.
    static func make(ranked: [RankedFlag], expanded: Bool,
                     threshold: Double = AppConfig.confidenceThreshold) -> ResultsDisplay {
        guard let top = ranked.first, top.probability >= threshold else { return .hint }
        return .list(Array(ranked.prefix(expanded ? AppConfig.expandedCount : AppConfig.collapsedCount)))
    }
}
