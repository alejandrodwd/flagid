import SwiftUI

/// Bottom panel: the ranked list, or a hint when the model is not confident.
struct ResultsPanel: View {
    @Bindable var model: RecognitionViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch model.display {
            case .hint:
                hint
            case .list(let flags):
                ForEach(Array(flags.enumerated()), id: \.element.id) { index, flag in
                    Button { model.select(flag) } label: { row(flag, rank: index + 1) }
                        .buttonStyle(.plain)
                }
                if model.hasMoreThanCollapsed {
                    Button(model.expanded ? "Less" : "More") {
                        withAnimation(.snappy) { model.expanded.toggle() }
                    }
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.top, 2)
                }
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .animation(.default, value: model.display == .hint)
    }

    private var hint: some View {
        HStack(spacing: 10) {
            Image(systemName: "flag.fill").font(.title3)
            Text(hintText).font(.headline)
            Spacer(minLength: 0)
            if model.isClassifyingPhoto { ProgressView() }
        }
        .padding(.vertical, 6)
    }

    private var hintText: String {
        switch model.source {
        case .camera:
            if model.cameraStatus != .running { return "Pick a photo of a flag" }
            return model.focusRect == nil ? "Point the camera at a flag" : "Tap the flag itself"
        case .photo:
            if model.isClassifyingPhoto { return "Looking…" }
            return model.focusRect == nil ? "No flag found. Tap a flag to zoom in" : "No flag here. Tap again for full frame"
        }
    }

    private func row(_ flag: RankedFlag, rank: Int) -> some View {
        HStack(spacing: 12) {
            EntryFlags(entry: flag.entry)
                .frame(width: 64, height: 28, alignment: .leading)
            Text(flag.entry.name)
                .font(rank == 1 ? .headline : .body)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 8)
            Text(flag.probability, format: .percent.precision(.fractionLength(0)))
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityHint("Shows details")
    }
}
