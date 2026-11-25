// Renders each page in DIR/pages.txt (or LIST) with WebKit at the display's
// backing scale (2x on Retina) and saves NAME@2x.png, NAME.json (word text and
// boxes in CSS pixels, i.e. 1x points) and NAME-bare@2x.png, the same page with
// all text hidden, which pagecrops.py diffs to tell which words show. Pages
// with a bare render already are skipped. Usage: ./pagerender DIR [LIST]
import AppKit
import WebKit

let dir = CommandLine.arguments[1]
let list = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "\(dir)/pages.txt"
let names = try! String(contentsOfFile: list, encoding: .utf8)
    .split(separator: "\n").map(String.init)
    .filter { !FileManager.default.fileExists(atPath: "\(dir)/\($0)-bare@2x.png") }
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
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.capture() }
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
                let hide = "for (const e of document.querySelectorAll('*')) { const cs = getComputedStyle(e); if (cs.webkitBackgroundClip === 'text') e.style.background = 'none'; e.style.setProperty('color', 'transparent', 'important'); e.style.setProperty('-webkit-text-fill-color', 'transparent', 'important'); e.style.setProperty('-webkit-text-stroke-width', '0', 'important'); e.style.setProperty('text-shadow', 'none', 'important'); } 1"
                self.web.evaluateJavaScript(hide) { _, _ in
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.web.takeSnapshot(with: WKSnapshotConfiguration()) { img2, _ in
                            if let img2, let cg2 = img2.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                                let rep2 = NSBitmapImageRep(cgImage: cg2)
                                try! rep2.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "\(dir)/\(id)-bare@2x.png"))
                            }
                            print("\(id): \(cg.width)x\(cg.height)")
                            self.i += 1
                            self.next()
                        }
                    }
                }
            }
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let r = R()
r.next()
app.run()
