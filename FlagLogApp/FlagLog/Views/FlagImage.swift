import SwiftUI
import UIKit

/// One flag from the `Flags` asset folder, or a placeholder with the code if there is no artwork.
struct FlagImage: View {
    let code: String

    var body: some View {
        if let image = UIImage(named: "Flags/\(code)") {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .overlay(RoundedRectangle(cornerRadius: 2).stroke(.white.opacity(0.25), lineWidth: 0.5))
        } else {
            RoundedRectangle(cornerRadius: 3)
                .fill(.gray.opacity(0.4))
                .aspectRatio(1.5, contentMode: .fit)
                .overlay(Text(code.uppercased()).font(.caption2.bold()).minimumScaleFactor(0.5))
        }
    }
}

/// An entry's flag, or its members' flags side by side for a lookalike group.
struct EntryFlags: View {
    let entry: FlagEntry
    var height: CGFloat = 26

    var body: some View {
        HStack(spacing: 3) {
            ForEach(entry.members, id: \.self) { FlagImage(code: $0).frame(maxHeight: height) }
        }
        .accessibilityHidden(true)
    }
}
