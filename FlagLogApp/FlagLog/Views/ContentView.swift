import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var model = RecognitionViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var pinchScale: CGFloat = 1
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                background
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
                    .contentShape(Rectangle())
                    .onTapGesture(coordinateSpace: .local) { model.handleTap(at: $0, viewSize: geo.size) }
                    .gesture(model.source == .camera ? pinch : nil)
                focusBox(viewSize: geo.size)
            }
            .ignoresSafeArea()
            .overlay(alignment: .top) { topBar.padding(.horizontal) }
            .overlay(alignment: .bottom) { bottomArea.padding(.horizontal).padding(.bottom, 8) }
        }
        .sheet(item: $model.frozen, onDismiss: model.unfreeze) { frozen in
            FlagDetailView(flag: frozen.flag, catalog: model.catalog, onClose: model.unfreeze)
                .presentationDetents([.medium, .large])
        }
        .task { await model.startCamera() }
        .onChange(of: scenePhase) { _, phase in
            guard model.source == .camera else { return }
            if phase == .active { Task { await model.startCamera() } } else if phase == .background { model.stopCamera() }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self) { model.usePhoto(data: data) }
                pickerItem = nil
            }
        }
    }

    // MARK: Background: camera, frozen frame, or photo

    @ViewBuilder private var background: some View {
        switch model.source {
        case .camera:
            ZStack {
                if model.cameraStatus == .running {
                    CameraPreview(session: model.camera.session)
                } else {
                    cameraUnavailable
                }
                if let still = model.frozen?.image {
                    Image(uiImage: still).resizable().scaledToFill()
                }
            }
        case .photo(let image):
            Image(uiImage: image).resizable().scaledToFit()
        }
    }

    private var cameraUnavailable: some View {
        VStack(spacing: 12) {
            Image(systemName: "video.slash").font(.largeTitle)
            Text(cameraMessage).multilineTextAlignment(.center)
            if model.cameraStatus == .denied {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }
            }
        }
        .foregroundStyle(.white.opacity(0.8))
        .padding(40)
    }

    private var cameraMessage: String {
        switch model.cameraStatus {
        case .denied: return "Camera access is off. Allow it in Settings, or pick a photo."
        case .unavailable: return "No camera on this device.\nPick a photo to try the model."
        case .failed: return "The camera could not be started. Pick a photo instead."
        case .notConfigured, .running: return "Starting camera…"
        }
    }

    @ViewBuilder private func focusBox(viewSize: CGSize) -> some View {
        if let rect = model.focusRect {
            let r = CropGeometry.viewRect(fromImage: rect, viewSize: viewSize, imageSize: model.imageSize,
                                          mode: model.contentMode)
            RoundedRectangle(cornerRadius: 8)
                .stroke(.yellow, lineWidth: 3)
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)
                .allowsHitTesting(false)
        }
    }

    private var pinch: some Gesture {
        MagnifyGesture()
            .onChanged { model.pinchChanged($0.magnification) }
            .onEnded { model.pinchEnded($0.magnification) }
    }

    // MARK: Controls

    private var topBar: some View {
        HStack {
            if case .photo = model.source {
                Button { model.returnToCamera() } label: { Label("Camera", systemImage: "camera") }
                    .buttonStyle(.borderedProminent).tint(.black.opacity(0.5))
            }
            Spacer()
            if model.focusRect != nil {
                Button { model.clearFocus() } label: { Label("Full frame", systemImage: "arrow.up.left.and.arrow.down.right") }
                    .buttonStyle(.borderedProminent).tint(.black.opacity(0.5))
            }
        }
    }

    private var bottomArea: some View {
        VStack(spacing: 10) {
            if let error = model.loadError {
                Text(error).font(.footnote).padding(10)
                    .background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 10))
            }
            HStack {
                Spacer()
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Label("Photo", systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.borderedProminent).tint(.black.opacity(0.5))
            }
            if model.loadError == nil || model.catalog != nil {
                ResultsPanel(model: model)
            }
        }
    }
}
