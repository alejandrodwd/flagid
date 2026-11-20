import CoreML
import CoreVideo
import ImageIO
import Vision

/// Runs the flag model through Vision. Safe to call from any one background queue at a time.
final class FlagClassifier: @unchecked Sendable {
    enum LoadError: Error, LocalizedError {
        case modelNotFound(String)
        case noMetadata

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let name): return "\(name).mlmodelc is missing from the app bundle."
            case .noMetadata: return "The model has no names/groups metadata."
            }
        }
    }

    let catalog: FlagCatalog
    private let model: VNCoreMLModel

    /// Loads `<AppConfig.modelName>.mlmodelc` from `bundle` (Xcode compiles the .mlpackage).
    convenience init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: AppConfig.modelName, withExtension: "mlmodelc") else {
            throw LoadError.modelNotFound(AppConfig.modelName)
        }
        try self.init(compiledModelURL: url)
    }

    init(compiledModelURL url: URL) throws {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly   // no Neural Engine in the simulator
        #else
        config.computeUnits = .all
        #endif
        let mlModel = try MLModel(contentsOf: url, configuration: config)
        let description = mlModel.modelDescription
        guard let metadata = description.metadata[.creatorDefinedKey] as? [String: String] else {
            throw LoadError.noMetadata
        }
        catalog = try FlagCatalog(metadata: metadata, classLabels: description.classLabels as? [String])
        model = try VNCoreMLModel(for: mlModel)
    }

    /// Probabilities for every label. `region` is a normalised rect with a top-left origin in the
    /// upright image (nil = the whole frame); Vision crops it, then centre-crops to 224×224.
    func classify(cgImage: CGImage, orientation: CGImagePropertyOrientation = .up,
                  region: CGRect? = nil) throws -> [String: Double] {
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
        return try perform(handler, region: region)
    }

    func classify(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation = .up,
                  region: CGRect? = nil) throws -> [String: Double] {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)
        return try perform(handler, region: region)
    }

    private func perform(_ handler: VNImageRequestHandler, region: CGRect?) throws -> [String: Double] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .centerCrop
        if let region {
            request.regionOfInterest = CropGeometry.visionRegion(fromTopLeft: region)
        }
        try handler.perform([request])
        let observations = request.results as? [VNClassificationObservation] ?? []
        var probs = [String: Double](minimumCapacity: observations.count)
        for o in observations { probs[o.identifier] = Double(o.confidence) }
        return probs
    }
}
