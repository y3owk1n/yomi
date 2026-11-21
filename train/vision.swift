// Reads evaluation pages with Apple's Vision framework, the baseline yomi is
// measured against on macOS. Same output as `yomi bench`: one JSON line per
// frame with the words, their boxes in logical pixels, and the first and
// median read times (the image decoded before timing).
//
// Usage: ./vision DIR fast|accurate auto|en-US REPS > out.jsonl
//   auto lets Vision detect each line's language, as Neru's accurate read does.
import CoreGraphics
import Foundation
import ImageIO
import Vision

let dir = CommandLine.arguments[1]
let accurate = CommandLine.arguments[2] == "accurate"
let auto = CommandLine.arguments[3] == "auto"
let reps = Int(CommandLine.arguments[4])!
let pages = try! String(contentsOfFile: "\(dir)/pages.txt", encoding: .utf8).split(separator: "\n").map(String.init)

func decoded(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let ctx = CGContext(data: nil, width: img.width, height: img.height, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: img.width, height: img.height))
    return ctx.makeImage()
}

func read(_ img: CGImage) -> (Double, [VNRecognizedTextObservation]) {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = accurate ? .accurate : .fast
    req.usesLanguageCorrection = false
    if auto { req.automaticallyDetectsLanguage = true } else { req.recognitionLanguages = ["en-US"] }
    let t = DispatchTime.now().uptimeNanoseconds
    try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
    return (Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6, req.results ?? [])
}

for page in pages {
    for scale in [1, 2] {
        guard let img = decoded("\(dir)/\(page)@\(scale)x.png") else { continue }
        let W = Double(img.width) / Double(scale), H = Double(img.height) / Double(scale)
        let (first, obs) = read(img)
        let times = (0..<reps).map { _ in read(img).0 }.sorted()
        var words: [[Any]] = []
        for o in obs {
            guard let cand = o.topCandidates(1).first else { continue }
            let s = cand.string
            var i = s.startIndex
            while i < s.endIndex {
                while i < s.endIndex, s[i] == " " { i = s.index(after: i) }
                guard i < s.endIndex else { break }
                var e = i
                while e < s.endIndex, s[e] != " " { e = s.index(after: e) }
                if let b = try? cand.boundingBox(for: i..<e)?.boundingBox {
                    words.append([String(s[i..<e]), b.minX * W, (1 - b.maxY) * H, b.width * W, b.height * H, cand.confidence])
                }
                i = e
            }
        }
        let rec: [String: Any] = ["page": page, "scale": scale, "first_ms": first,
                                  "median_ms": times.isEmpty ? first : times[times.count / 2], "words": words]
        print(String(data: try! JSONSerialization.data(withJSONObject: rec), encoding: .utf8)!)
        fflush(stdout)
    }
}
