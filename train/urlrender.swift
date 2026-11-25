// Renders each "NAME URL" line of URLS in an offscreen WKWebView at the
// display's backing scale, waits for it to settle, and saves DIR/NAME@2x.png
// and DIR/NAME.json (visible DOM words and boxes, from words.js). Pages
// already in DIR are skipped. Usage: ./urlrender URLS DIR
import AppKit
import WebKit

let dir = CommandLine.arguments[2]
let entries = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
    .split(separator: "\n").map { $0.split(separator: " ").map(String.init) }.filter { $0.count == 2 }
let js = try! String(contentsOfFile: "words.js", encoding: .utf8)
let W = 1280, H = 800

final class R: NSObject, WKNavigationDelegate {
    let web: WKWebView
    let win: NSWindow
    var i = 0
    var done = false
    override init() {
        win = NSWindow(contentRect: NSRect(x: -5000, y: 0, width: W, height: H), styleMask: .borderless, backing: .buffered, defer: false)
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .nonPersistent()
        web = WKWebView(frame: NSRect(x: 0, y: 0, width: W, height: H), configuration: cfg)
        web.customUserAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
        super.init()
        win.contentView = web
        win.orderBack(nil)
        web.navigationDelegate = self
    }
    func next() {
        guard i < entries.count else { exit(0) }
        done = false
        let name = entries[i][0]
        if FileManager.default.fileExists(atPath: "\(dir)/\(name)@2x.png") { i += 1; next(); return }
        web.load(URLRequest(url: URL(string: entries[i][1])!, timeoutInterval: 20))
        let mine = i
        // A page that never finishes loading is captured as it stands.
        DispatchQueue.main.asyncAfter(deadline: .now() + 25) { if self.i == mine && !self.done { self.settle() } }
    }
    func webView(_ w: WKWebView, didFinish _: WKNavigation!) { settle() }
    func webView(_ w: WKWebView, didFail _: WKNavigation!, withError e: Error) { skip("\(e)") }
    func webView(_ w: WKWebView, didFailProvisionalNavigation _: WKNavigation!, withError e: Error) { skip("\(e)") }
    func skip(_ why: String) {
        guard !done else { return }
        done = true
        print("\(entries[i][0]): failed \(why.prefix(120))")
        i += 1
        next()
    }
    func settle() {
        guard !done else { return }
        done = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 4) { self.capture() }
    }
    func capture() {
        let id = entries[i][0]
        web.evaluateJavaScript(js + "; window.scrollTo(0,0); __words()") { gt, err in
            guard let gt = gt as? String else { print("js \(id): \(String(describing: err))"); self.i += 1; self.next(); return }
            self.web.takeSnapshot(with: WKSnapshotConfiguration()) { img, err in
                if let img, let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    let rep = NSBitmapImageRep(cgImage: cg)
                    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "\(dir)/\(id)@2x.png"))
                    try! gt.write(toFile: "\(dir)/\(id).json", atomically: true, encoding: .utf8)
                    print("\(id): \(cg.width)x\(cg.height)")
                } else {
                    print("snap \(id): \(String(describing: err))")
                }
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
