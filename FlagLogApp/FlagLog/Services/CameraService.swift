import AVFoundation
import CoreImage
import UIKit

/// Back camera capture. Delivers portrait pixel buffers to `onFrame` on a background queue.
final class CameraService: NSObject, @unchecked Sendable {
    enum Status: Equatable { case notConfigured, running, denied, unavailable, failed }

    let session = AVCaptureSession()
    /// Called on the video queue for every frame; keep it cheap (the frame gate drops most).
    var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)?

    private let sessionQueue = DispatchQueue(label: "flaglog.camera.session")
    private let videoQueue = DispatchQueue(label: "flaglog.camera.video", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?
    private var configured = false

    /// Asks for permission if needed, configures the session once and starts it.
    func start() async -> Status {
        guard AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) != nil else {
            return .unavailable   // e.g. the simulator
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: break
        case .notDetermined:
            guard await AVCaptureDevice.requestAccess(for: .video) else { return .denied }
        default: return .denied
        }
        return await withCheckedContinuation { continuation in
            sessionQueue.async {
                if !self.configured {
                    let status = self.configure()
                    guard status == .running else { continuation.resume(returning: status); return }
                    self.applyFrameRate(AppConfig.cameraFPS)
                    self.configured = true
                }
                if !self.session.isRunning { self.session.startRunning() }
                continuation.resume(returning: .running)
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Current zoom factor and the range pinch-to-zoom may use.
    var zoomFactor: CGFloat { device?.videoZoomFactor ?? 1 }

    func setZoom(_ factor: CGFloat) {
        sessionQueue.async {
            guard let device = self.device else { return }
            let upper = min(device.maxAvailableVideoZoomFactor, AppConfig.maxZoomFactor)
            let clamped = min(max(factor, device.minAvailableVideoZoomFactor), upper)
            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = clamped
                device.unlockForConfiguration()
            } catch {}
        }
    }

    /// Locks the camera to `fps` if the active format supports it (otherwise keeps the default).
    /// Called after the session preset is committed, which would otherwise reset it.
    private func applyFrameRate(_ fps: Double) {
        guard let device,
              device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
                  $0.minFrameRate <= fps && fps <= $0.maxFrameRate
              }),
              (try? device.lockForConfiguration()) != nil else { return }
        let duration = CMTime(value: 1, timescale: CMTimeScale(fps.rounded()))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.unlockForConfiguration()
    }

    private func configure() -> Status {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return .unavailable
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .high
        guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            return .failed
        }
        session.addInput(input)

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return .failed }
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90   // portrait buffers, so Vision orientation is .up
        }

        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            device.unlockForConfiguration()
        }
        self.device = device
        return .running
    }
}

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(buffer, time)
    }
}

extension CVPixelBuffer {
    /// A still copy of the frame, for the frozen view.
    func makeUIImage() -> UIImage? {
        let image = CIImage(cvPixelBuffer: self)
        guard let cg = CIContext().createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg)
    }

    var size: CGSize { CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self)) }
}
