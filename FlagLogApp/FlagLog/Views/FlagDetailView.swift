import SwiftUI

/// Detail of a picked entry: name, code, and both members for a lookalike group.
struct FlagDetailView: View {
    let flag: RankedFlag
    let catalog: FlagCatalog?
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    EntryFlags(entry: flag.entry, height: flag.entry.isGroup ? 80 : 120)
                        .shadow(radius: 4)
                        .padding(.top, 8)
                    VStack(spacing: 4) {
                        Text(flag.entry.name).font(.title2.bold()).multilineTextAlignment(.center)
                        Text(flag.entry.label.uppercased()).font(.subheadline.monospaced()).foregroundStyle(.secondary)
                        Text("\(flag.probability, format: .percent.precision(.fractionLength(0))) match")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    if flag.entry.isGroup {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("These flags look alike; the model can't tell them apart.")
                                .font(.footnote).foregroundStyle(.secondary)
                            ForEach(flag.entry.members, id: \.self) { code in
                                HStack(spacing: 12) {
                                    FlagImage(code: code).frame(width: 60, height: 32)
                                    Text(catalog?.name(forCode: code) ?? FlagCatalog.fallbackName(for: code))
                                    Spacer()
                                    Text(code.uppercased()).font(.subheadline.monospaced()).foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding()
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                    }
                }
                .padding()
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done", action: onClose) }
            }
        }
    }
}
