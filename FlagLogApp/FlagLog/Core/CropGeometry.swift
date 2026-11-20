import CoreGraphics

/// Coordinate maths for tap-to-focus. Image rects and points are normalised to 0...1 with the
/// origin at the top left of the upright image.
enum CropGeometry {
    enum ContentMode { case fill, fit }

    /// Square crop centred on `center`, with side `fraction` × the image's short side, moved
    /// (not shrunk) to stay inside the image.
    static func focusRect(center: CGPoint, imageSize: CGSize,
                          fraction: Double = AppConfig.focusCropFraction) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return CGRect(x: 0, y: 0, width: 1, height: 1) }
        let side = min(imageSize.width, imageSize.height) * fraction
        let w = side / imageSize.width, h = side / imageSize.height
        let x = min(max(center.x - w / 2, 0), 1 - w)
        let y = min(max(center.y - h / 2, 0), 1 - h)
        return CGRect(x: x, y: y, width: w, height: h)
    }

    /// Vision's `regionOfInterest` uses a bottom-left origin.
    static func visionRegion(fromTopLeft rect: CGRect) -> CGRect {
        CGRect(x: rect.minX, y: 1 - rect.maxY, width: rect.width, height: rect.height)
    }

    /// Where the image is drawn inside a view of `viewSize` (aspect fill or fit, centred).
    static func displayedImageFrame(viewSize: CGSize, imageSize: CGSize, mode: ContentMode) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return CGRect(origin: .zero, size: viewSize) }
        let sx = viewSize.width / imageSize.width, sy = viewSize.height / imageSize.height
        let scale = mode == .fill ? max(sx, sy) : min(sx, sy)
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(x: (viewSize.width - size.width) / 2, y: (viewSize.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    /// Normalised image point under a view point, or nil if the point is outside the image.
    static func imagePoint(fromView point: CGPoint, viewSize: CGSize, imageSize: CGSize,
                           mode: ContentMode) -> CGPoint? {
        let frame = displayedImageFrame(viewSize: viewSize, imageSize: imageSize, mode: mode)
        guard frame.width > 0, frame.height > 0, frame.contains(point) else { return nil }
        return CGPoint(x: (point.x - frame.minX) / frame.width, y: (point.y - frame.minY) / frame.height)
    }

    /// View rect of a normalised image rect.
    static func viewRect(fromImage rect: CGRect, viewSize: CGSize, imageSize: CGSize,
                         mode: ContentMode) -> CGRect {
        let frame = displayedImageFrame(viewSize: viewSize, imageSize: imageSize, mode: mode)
        return CGRect(x: frame.minX + rect.minX * frame.width, y: frame.minY + rect.minY * frame.height,
                      width: rect.width * frame.width, height: rect.height * frame.height)
    }
}
