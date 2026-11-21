// Renders each page in pages.txt with WebKit at the display's backing scale
// (2x on Retina), saves NAME@2x.png and NAME.json (word text and boxes in CSS
// pixels, i.e. 1x points). Usage: ./render DIR
import AppKit
import WebKit

let dir = CommandLine.arguments[1]
let names = try! String(contentsOfFile: "\(dir)/pages.txt", encoding: .utf8)
    .split(separator: "\n").map(String.init)
let W = 1280, H = 800

final class R: NSObject, WKNavigationDelegate {
    let web: WKWebView
    let win: NSWindow
    var i = 0
    override init() {
        win = NSWindow(contentRect: NSRect(x: -5000, y: 0, width: W, height: H), styleMask: .borderless, backing: .buffered, defer: false)
        web = WKWebView(frame: NSRect(x: 0, y: 0, width: W, height: H))
        super.init()
        win.contentView = web
        win.orderBack(nil)
        web.navigationDelegate = self
    }
    func next() {
        guard i < names.count else { exit(0) }
        let url = URL(fileURLWithPath: "\(dir)/\(names[i]).html")
        web.loadFileURL(url, allowingReadAccessTo: URL(fileURLWithPath: dir))
    }
    func webView(_ w: WKWebView, didFinish _: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { self.capture() }
    }
    func capture() {
        let id = names[i]
        web.evaluateJavaScript("__words()") { gt, err in
            if let err { print("js \(id): \(err)"); exit(1) }
            self.web.takeSnapshot(with: WKSnapshotConfiguration()) { img, err in
                guard let img, let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                    print("snap \(id): \(String(describing: err))"); exit(1)
                }
                let rep = NSBitmapImageRep(cgImage: cg)
                try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "\(dir)/\(id)@2x.png"))
                try! (gt as! String).write(toFile: "\(dir)/\(id).json", atomically: true, encoding: .utf8)
                print("\(id): \(cg.width)x\(cg.height)")
                self.i += 1
                self.next()
            }
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let r = R()
r.next()
app.run()
